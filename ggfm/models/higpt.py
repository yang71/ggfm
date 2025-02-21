import os
import json
import logging
import glob
import math
import re
import html
import gzip
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache, cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable, Any, TypeVar
from omegaconf import OmegaConf
from urllib.parse import urlparse
from torch import Tensor
from torch_geometric.typing import Adj, EdgeType, NodeType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import CrossEntropyLoss, Parameter
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    degree,
    softmax,
    add_remaining_self_loops,
)
from torch_geometric.utils.hetero import construct_bipartite_edge_index
from torch_scatter import scatter_add

import ftfy


"""Special tokens used in the model"""
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def is_url(url_or_filename):
    """
    Check if a string is a URL.
    
    Args:
        url_or_filename (str): String to check
        
    Returns:
        bool: True if string is a URL, False otherwise
    """
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def get_abs_path(path):
    """
    Get absolute path from a potentially relative path.
    
    Args:
        path (str): Input path
        
    Returns:
        str: Absolute path
    """
    return os.path.abspath(os.path.expanduser(path))

@lru_cache()
def default_bpe():
    """
    Get default path to BPE vocabulary file.
    
    Returns:
        str: Path to BPE vocabulary file
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")

@lru_cache()
def bytes_to_unicode():
    """
    Convert bytes to unicode characters.
    
    Returns:
        dict: Mapping from bytes to unicode characters
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """
    Get all adjacent pairs of characters from a word.
    
    Args:
        word (tuple): Word as tuple of characters
        
    Returns:
        set: Set of character pairs
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    """
    Basic text cleaning using ftfy.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    """
    Clean whitespace in text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized whitespace
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class BaseModel(nn.Module):
    """
    Base class for all models in HiGPT.
    
    Provides common functionality for model loading, optimization and evaluation.
    """

    def __init__(self):
        """Initialize the base model."""
        super().__init__()

    @property
    def device(self):
        """Get the device where model parameters are stored."""
        return list(self.parameters())[0].device

    def load_checkpoint(self, url_or_filename):
        """
        Load model weights from a checkpoint file.
        
        Args:
            url_or_filename (str): Path or URL to checkpoint
            
        Returns:
            LoaderOutput: Results of loading the checkpoint
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)
        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)
        return msg

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Create a model instance from pretrained weights.
        
        Args:
            model_type (str): Type/name of the pretrained model
            
        Returns:
            BaseModel: Model instance initialized with pretrained weights
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)
        return model

    @classmethod
    def default_config_path(cls, model_type):
        """
        Get the default configuration file path for a model type.
        
        Args:
            model_type (str): Type/name of the model
            
        Returns:
            str: Path to the configuration file
            
        Raises:
            AssertionError: If model_type is not recognized
        """
        assert model_type in cls.PRETRAINED_MODEL_CONFIG_DICT, "Unknown model type {}".format(model_type)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint based on configuration.
        
        Args:
            cfg (Config): Configuration object containing checkpoint paths
            **kwargs: Additional arguments for loading pretrained weights
            
        Raises:
            AssertionError: If required paths are missing in config
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert finetune_path is not None, "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            load_pretrained = cfg.get("load_pretrained", True)
            if load_pretrained:
                pretrain_path = cfg.get("pretrained", None)
                assert "Found load_finetuned is False, but pretrain_path is None."
                self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

    def before_training(self, **kwargs):
        pass

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        """
        Get parameters for optimizer with proper weight decay settings.
        
        Args:
            weight_decay (float): Weight decay factor
            lr_scale (float, optional): Learning rate scaling factor. Defaults to 1
            
        Returns:
            list: List of parameter groups with optimization settings
        """
        p_wd, p_non_wd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
        optim_params = [
            {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
        ]
        return optim_params

    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        """
        Calculate and format the total number of parameters.
        
        Args:
            return_str (bool, optional): Whether to return formatted string. Defaults to True
            
        Returns:
            Union[str, int]: Number of parameters as string (with M/K suffix) or integer
        """
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot

class LayerNorm(nn.LayerNorm):
    """
    Layer normalization module with fp16 support.
    
    Extends PyTorch's LayerNorm to properly handle float16 precision.
    """

    def forward(self, x: torch.Tensor):
        """
        Apply layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalized tensor in original dtype
        """
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class QuickGELU(nn.Module):
    """
    Fast approximation of GELU activation function.
    
    Uses sigmoid multiplication instead of error function for efficiency.
    """
    
    def forward(self, x: torch.Tensor):
        """
        Apply Quick GELU activation.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Activated tensor
        """
        return x * torch.sigmoid(1.702 * x)
    
class graph_transformer(nn.Module):
    """
    Graph Transformer model for processing graph structured data.
    
    Combines transformer architecture with graph neural network components
    to process graph node features and edge structure.
    """
    
    def __init__(self, args):
        """
        Initialize the graph transformer.
        
        Args:
            args: Configuration object containing model parameters including:
                - gnn_width: Width of GNN layers
                - gnn_layers: Number of GNN layers
                - gnn_heads: Number of attention heads
        """
        super().__init__()
        self.config = PretrainedConfig()
        self.gnn = Transformer(
            width=args.gnn_width,
            layers=args.gnn_layers,
            heads=args.gnn_heads
        )
        self.ln_post = LayerNorm(args.gnn_width)
        self.proj = nn.Parameter(torch.randn(args.gnn_width, args.gnn_output) / args.gnn_width ** 0.5)

    def forward(self, g):
        """
        Process input graph through the transformer.
        
        Args:
            g: Graph object containing:
                - graph_node: Node feature tensor
                - edge_index: Edge connectivity tensor
                
        Returns:
            torch.Tensor: Processed node features
        """
        x = g.graph_node
        edge_index = g.edge_index
        
        x = self.gnn(x)
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
            
        return x

def load_model(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: Optional[str] = None,
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    debug: bool = False,
):
    """
    Load a pretrained model from Hugging Face.
    
    Args:
        model_path (str): Path or name of model on HuggingFace
        device (str): Device to load model on ('cpu', 'cuda', 'mps')
        num_gpus (int): Number of GPUs to use
        max_gpu_memory (str, optional): Maximum GPU memory per device
        load_8bit (bool, optional): Whether to load in 8-bit precision. Defaults to False
        cpu_offloading (bool, optional): Whether to offload weights to CPU. Defaults to False
        debug (bool, optional): Whether to print debug info. Defaults to False
        
    Returns:
        AutoModelForCausalLM: Loaded model instance
        
    Raises:
        ValueError: If device is invalid
    """
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs["device_map"] = "sequential"
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
    else:
        raise ValueError(f"Invalid device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **kwargs
    )

    if (device == "cuda" and num_gpus == 1 and not cpu_offloading) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model


def add_model_args(parser):
    """Add model arguments to the parser."""
    group = parser.add_argument_group('model')
    group.add_argument(
        "--model-path",
        type=str,
        default="lmsys/vicuna-7b-v1.3",
        help="Path to the model weights or model name on Hugging Face."
    )
    group.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda",
        help="The device to run the model on."
    )
    group.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use."
    )
    group.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maximum GPU memory to use per GPU."
    )
    group.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load the model in 8-bit precision."
    )
    group.add_argument(
        "--cpu-offloading", 
        action="store_true",
        help="Offload model weights to CPU to save GPU memory."
    )
    return group

def get_gpu_memory(num_gpus):
    """Get available memory for each GPU."""
    import torch.cuda
    gpu_memory = []
    for i in range(num_gpus):
        with torch.cuda.device(i):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)  
            gpu_memory.append(total_memory)
    return gpu_memory

def maybe_zero_3(param, ignore_status=False, name=None):
    """Handle DeepSpeed ZeRO-3 params."""
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class ResidualAttentionBlock(nn.Module):
    """
    Transformer block with residual attention and MLP.
    
    Implements a standard transformer block with self-attention followed by MLP,
    with layer normalization and residual connections.
    """
    
    def __init__(self, d_model: int, n_head: int, act_layer: Callable = nn.GELU):
        """
        Initialize the residual attention block.
        
        Args:
            d_model (int): Hidden dimension size
            n_head (int): Number of attention heads
            act_layer (Callable, optional): Activation function. Defaults to GELU
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Compute self-attention.
        
        Args:
            x (torch.Tensor): Input tensor
            attn_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Self-attention output
        """
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through the block.
        
        Args:
            x (torch.Tensor): Input tensor
            attn_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Processed tensor
        """
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """
    Full transformer model with multiple attention blocks.
    
    Stacks multiple ResidualAttentionBlocks to form a complete transformer.
    """
    
    def __init__(self, width: int, layers: int, heads: int, act_layer: Callable = nn.GELU):
        """
        Initialize the transformer.
        
        Args:
            width (int): Hidden dimension size
            layers (int): Number of transformer layers
            heads (int): Number of attention heads per layer
            act_layer (Callable, optional): Activation function. Defaults to GELU
        """
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, act_layer=act_layer) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through the transformer.
        
        Args:
            x (torch.Tensor): Input tensor
            attn_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Processed tensor
        """
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x

class SimpleTokenizer(object):
    """
    Basic tokenizer implementation with BPE encoding.
    
    Implements byte-pair encoding (BPE) tokenization with support for special tokens
    and caching.
    """
    
    def __init__(self, bpe_path: str = default_bpe(), special_tokens=None):
        """
        Initialize the tokenizer.
        
        Args:
            bpe_path (str, optional): Path to BPE vocabulary file
            special_tokens (list, optional): Additional special tokens to add
        """
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        if not special_tokens:
            special_tokens = ["<start_of_text>", "<end_of_text>"]
        else:
            special_tokens = ["<start_of_text>", "<end_of_text>"] + special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]

    def bpe(self, token):
        """
        Apply byte-pair encoding to a token.
        
        Args:
            token (str): Input token
            
        Returns:
            str: BPE-encoded token
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """
        Encode text into token IDs.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of token IDs
        """
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        """
        Decode token IDs back to text.
        
        Args:
            tokens (list): List of token IDs
            
        Returns:
            str: Decoded text
        """
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace").replace("</w>", " ")
        return text

def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """
    Tokenize text(s) with special tokens and padding.
    
    Args:
        texts (Union[str, List[str]]): Input text or list of texts
        context_length (int, optional): Maximum sequence length. Defaults to 77
        
    Returns:
        torch.LongTensor: Tensor of token IDs with shape (batch_size, context_length)
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<start_of_text>"]
    eot_token = _tokenizer.encoder["<end_of_text>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

_tokenizer = SimpleTokenizer()



def gcn_conv(h, edge_index):
    """
    Basic Graph Convolutional Network convolution operation.
    
    Implements the standard GCN propagation rule with normalized adjacency matrix.
    
    Args:
        h (torch.Tensor): Node feature matrix
        edge_index (torch.Tensor): Graph connectivity in COO format
        
    Returns:
        torch.Tensor: Updated node features after convolution
    """
    N, node_feas = h.shape
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5) 
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    rows, cols = edge_index
    edge_msg = h[rows, :] * torch.unsqueeze(edge_weight, dim=-1)
    col_embeds = h[cols, :]
    tem = torch.zeros([N, node_feas]).to(edge_msg.device)
    rows = rows.to(edge_msg.device)
    h_prime = tem.index_add_(0, rows, edge_msg)
    return h_prime

class MPNN(nn.Module):
    """
    Message Passing Neural Network implementation.
    
    A general framework for graph neural networks that updates node representations
    via message passing between neighbors.
    
    Args:
        in_channels (int): Input feature dimension
        hidden_channels (int): Hidden layer dimension
        out_channels (int): Output feature dimension
        **kwargs: Additional arguments including:
            - dropout (float): Dropout rate
            - num_layers (int): Number of message passing layers
            - if_param (bool): Whether to use learnable parameters
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super(MPNN, self).__init__()
        self.config = PretrainedConfig()
        self.dropout = kwargs.get('dropout')
        self.num_layers = kwargs.get('num_layers')
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.if_param = kwargs.get('if_param')

        if self.if_param:
            self.fcs = nn.ModuleList([])
            self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
            for _ in range(self.num_layers - 2):
                self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias))
            self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias))
            self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters using Xavier initialization."""
        for mlp in self.fcs:
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, g, use_conv=True):
        """
        Forward pass through the MPNN.
        
        Args:
            g: Graph object containing node features and connectivity
            use_conv (bool, optional): Whether to use convolution. Defaults to True
            
        Returns:
            torch.Tensor: Updated node features
        """
        x = g.graph_node
        edge_index = g.edge_index
        try:
            device = self.parameters().__next__().device
        except:
            device = x.device
        x = x.to(device)
        edge_index = edge_index.to(device)

        for i in range(self.num_layers - 1):
            if self.if_param:
                x = x @ self.fcs[i].weight.t()
            if use_conv:
                x = gcn_conv(x, edge_index)
            if self.ff_bias and self.if_param:
                x = x + self.fcs[i].bias
            try:
                x = self.activation(self.bns(x))
            except:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.if_param:
            x = x @ self.fcs[-1].weight.t()
        if use_conv:
            x = gcn_conv(x, edge_index)
        if self.ff_bias and self.if_param:
            x = x + self.fcs[-1].bias
        return x

@dataclass
class MetaHGTConvCfg:
    """
    Configuration class for Meta Heterogeneous Graph Transformer Convolution.
    
    Attributes:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        heads (int): Number of attention heads
        dynamic (bool): Whether to use dynamic weight generation
    """
    in_channels: int
    out_channels: int
    heads: int
    dynamic: bool = True

class MetaHGTConv(MessagePassing):
    """
    Meta Heterogeneous Graph Transformer Convolution layer.
    
    Implements attention-based message passing for heterogeneous graphs with
    meta-learning capabilities for handling different types of nodes and edges.
    """
    
    def __init__(self, in_channels, out_channels, heads=1, dynamic=False, text_cfg=None, **kwargs):
        """
        Initialize the MetaHGTConv layer.
        
        Args:
            in_channels (int): Input feature dimension
            out_channels (int): Output feature dimension
            heads (int, optional): Number of attention heads. Defaults to 1
            dynamic (bool, optional): Whether to use dynamic weights. Defaults to False
            text_cfg: Text processing configuration
            **kwargs: Additional arguments
        """
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.config = PretrainedConfig()

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be divisible by the number of heads (got {heads})")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.kqv_lin = MetaHeteroDictLinear(text_cfg.width, self.in_channels,
                                        self.out_channels * 3, dynamic)

        self.out_lin = MetaHeteroDictLinear(text_cfg.width, self.out_channels, self.out_channels, dynamic)
        self.context_length = text_cfg.context_length

        dim = out_channels // heads

        self.k_rel = MetaHeteroLinear(text_cfg.width, dim, dim, dynamic)
        self.v_rel = MetaHeteroLinear(text_cfg.width, dim, dim, dynamic)

        self.skipTrans = nn.Linear(text_cfg.width, 1)
        self.p_relTrans = nn.Linear(text_cfg.width, heads)
        self.norm = nn.LayerNorm(self.out_channels, eps=1e-6)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """
        Concatenate features from different node types.
        
        Args:
            x_dict (Dict[str, Tensor]): Dictionary of node features by type
            
        Returns:
            Tuple[Tensor, Dict[str, int]]: Concatenated features and offset mapping
        """
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj], 
        edge_type_feas_dict: Dict[EdgeType, Tensor], 
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """
        Construct source node representations for attention.
        
        Args:
            k_dict (Dict[str, Tensor]): Key vectors by node type
            v_dict (Dict[str, Tensor]): Value vectors by node type
            edge_index_dict (Dict[EdgeType, Adj]): Edge indices by type
            edge_type_feas_dict (Dict[EdgeType, Tensor]): Edge type features
            
        Returns:
            Tuple[Tensor, Tensor, Dict[EdgeType, int]]: Processed key and value vectors with offsets
        """
        cumsum = 0
        num_edge_types = len(edge_index_dict.keys())
        H, D = self.heads, self.out_channels // self.heads

        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}

        edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(edge_index_dict.keys())
        }
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            edge_type_offset = edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        edge_feas_dict = {edge_types_map[k]: v for k, v in edge_type_feas_dict.items()}

        k = self.k_rel(ks, type_vec, edge_feas_dict).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec, edge_feas_dict).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    def _construct_p_rel(self, edge_type_feas_dict: Dict[EdgeType, Tensor]):
        """
        Construct relation-specific attention weights.
        
        Args:
            edge_type_feas_dict (Dict[EdgeType, Tensor]): Edge type features
            
        Returns:
            Dict[EdgeType, Tensor]: Processed attention weights for each edge type
        """
        p_rel = {k: self.p_relTrans(v).unsqueeze(0) for k, v in edge_type_feas_dict.items()}
        return p_rel

    def _construct_skip(self, node_type_feas_dict: Dict[EdgeType, Tensor]):
        """
        Construct skip connection weights.
        
        Args:
            node_type_feas_dict (Dict[EdgeType, Tensor]): Node type features
            
        Returns:
            Dict[EdgeType, Tensor]: Skip connection weights for each node type
        """
        skip = {k: self.skipTrans(v) for k, v in node_type_feas_dict.items()}
        return skip

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        data_type: str = 'dblp',
        node_type_feas_dict: Dict[NodeType, Tensor] = None,
        edge_type_feas_dict: Dict[EdgeType, Tensor] = None,
    ) -> Dict[NodeType, Optional[Tensor]]:
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        kqv_dict = self.kqv_lin(x_dict, node_type_feas_dict)

        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict, edge_type_feas_dict)
        p_rel = self._construct_p_rel(edge_type_feas_dict)
        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=p_rel)

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr,
                             size=None)

        dst_node_types = set([key[-1] for key in edge_index_dict.keys()])

        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        a_dict = self.out_lin({
            k: v if v is not None else v
            for k, v in out_dict.items()
        }, node_type_feas_dict)

        skip = self._construct_skip(node_type_feas_dict)

        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out = self.norm(out)
            out_dict[node_type] = out
            
        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        """
        Compute messages in the message passing framework.
        
        Implements the attention-based message computation between nodes.
        
        Args:
            k_j (Tensor): Key vectors of source nodes
            q_i (Tensor): Query vectors of target nodes
            v_j (Tensor): Value vectors of source nodes
            edge_attr (Tensor): Edge attributes
            index (Tensor): Target node indices
            ptr (Optional[Tensor]): Compressed sparse format pointer
            size_i (Optional[int]): Number of target nodes
            
        Returns:
            Tensor: Computed messages
        """
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        """
        Get string representation of the layer.
        
        Returns:
            str: Layer description with output channels and number of heads
        """
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')

class GNN(MessagePassing):
    """
    Graph Neural Network implementation.
    
    A basic GNN that uses message passing to update node representations.
    Includes learnable weight matrices and bias terms.
    
    Args:
        args: Configuration object containing:
            - gnn_hid (int): Hidden dimension size
            - gnn_input (int): Input feature dimension
            - gnn_output (int): Output feature dimension
    """
    
    def __init__(self, args, **kwargs):
        super(GNN, self).__init__(aggr='add', **kwargs)
        self.config = PretrainedConfig()
        self.vars = nn.ParameterList()

        w = nn.Parameter(torch.ones([args.gnn_hid, args.gnn_input]))
        torch.nn.init.xavier_uniform_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.gnn_hid)))

        w = nn.Parameter(torch.ones([args.gnn_output, args.gnn_hid]))
        torch.nn.init.xavier_uniform_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.gnn_output)))

    @staticmethod
    def norm(edge_index, num_nodes, improved=False, dtype=None):
        """
        Compute normalized edge weights.
        
        Args:
            edge_index (Tensor): Edge indices
            num_nodes (int): Number of nodes in graph
            improved (bool, optional): Whether to use improved normalization. Defaults to False
            dtype (torch.dtype, optional): Data type of weights
            
        Returns:
            Tuple[Tensor, Tensor]: Normalized edge indices and weights
        """
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, g, vars=None):
        device = self.parameters()[0].device
        g = g.to(device)
        
        edge_index = g.edge_index
        x = g.graph_node
        if vars is None:
            vars = self.vars
        improved = False

        w, b = vars[0], vars[1]
        edge_index, norm = self.norm(edge_index, x.size(self.node_dim), improved, x.dtype)
        x = self.propagate(edge_index, x=x, norm=norm)
        w = w.to(x.device)
        b = b.to(x.device)
        x = F.linear(x, w, b)
        x = F.leaky_relu(x)

        w, b = vars[2], vars[3]
        edge_index, norm = self.norm(edge_index, x.size(self.node_dim), improved, x.dtype)
        x = self.propagate(edge_index, x=x, norm=norm)
        w = w.to(x.device)
        b = b.to(x.device)
        x = F.linear(x, w, b)

        return x

    def parameters(self):
        return self.vars


@dataclass
class CLIPTextCfg:
    """
    Configuration class for CLIP text encoder.
    
    Attributes:
        context_length (int): Maximum sequence length for text
        vocab_size (int): Size of vocabulary
        width (int): Hidden dimension size
        heads (int): Number of attention heads
        layers (int): Number of transformer layers
    """
    context_length: int
    vocab_size: int
    width: int
    heads: int
    layers: int

@dataclass
class ClipOutputFeatures:
    """
    Data class for storing features extracted by CLIP model.
    
    Attributes:
        image_embeds (torch.FloatTensor, optional): Raw image embeddings
        image_embeds_proj (torch.FloatTensor, optional): Projected image embeddings
        text_embeds (torch.FloatTensor, optional): Raw text embeddings
        text_embeds_proj (torch.FloatTensor, optional): Projected text embeddings
    """
    image_embeds: Optional[torch.FloatTensor] = None
    image_embeds_proj: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    text_embeds_proj: Optional[torch.FloatTensor] = None

@dataclass
class ClipOutput:
    """
    Output class for CLIP model.
    
    Attributes:
        intermediate_output (ClipOutputFeatures, optional): Intermediate feature outputs
        logit_scale_exp (torch.FloatTensor, optional): Exponential of learnable temperature parameter
        loss (torch.FloatTensor, optional): Contrastive loss value
    """
    intermediate_output: Optional[ClipOutputFeatures] = None
    logit_scale_exp: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None

@dataclass
class HeteClipOutputFeatures:
    """
    Data class for storing features from heterogeneous CLIP model.
    
    Similar to ClipOutputFeatures but replaces image embeddings with graph embeddings
    for handling heterogeneous graph data.
    
    Attributes:
        graph_embeds (torch.FloatTensor, optional): Raw graph embeddings
        graph_embeds_proj (torch.FloatTensor, optional): Projected graph embeddings
        text_embeds (torch.FloatTensor, optional): Raw text embeddings
        text_embeds_proj (torch.FloatTensor, optional): Projected text embeddings
    """
    graph_embeds: Optional[torch.FloatTensor] = None
    graph_embeds_proj: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    text_embeds_proj: Optional[torch.FloatTensor] = None

class CLIP(BaseModel):
    """
    CLIP model adapted for graph-text contrastive learning.
    
    Implements a CLIP-style architecture that learns joint embeddings of 
    heterogeneous graphs and text descriptions.
    
    Args:
        embed_dim (int): Joint embedding dimension
        graph_cfg (MetaHGTConvCfg): Configuration for graph encoder
        text_cfg (CLIPTextCfg): Configuration for text encoder
        quick_gelu (bool, optional): Whether to use quick GELU activation. Defaults to False
    """
    def __init__(
        self,
        embed_dim: int,
        graph_cfg: MetaHGTConvCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
    ):
        super().__init__()

        self.tokenizer = tokenize
        self._loss = None

        if isinstance(graph_cfg, dict):
            graph_cfg = MetaHGTConvCfg(**graph_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        self.context_length = text_cfg.context_length
        act_layer = QuickGELU if quick_gelu else nn.GELU

        self.graph_encoder = MetaHGTConv(
            in_channels = graph_cfg.in_channels,
            out_channels = graph_cfg.out_channels,
            heads = graph_cfg.heads,
            dynamic = graph_cfg.dynamic,
            text_cfg = text_cfg
        )

        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=act_layer,
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, text_cfg.width))
        self.ln_final = LayerNorm(text_cfg.width)

        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.prompt_templates = openai_imagenet_template
        self.classifier = None

        self.init_parameters()

    @property
    def loss(self):
        """Get the contrastive loss function."""
        if self._loss is None:
            self._loss = HeteClipLoss()
        return self._loss

    def init_parameters(self):
        """Initialize model parameters with proper scaling."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """
        Build causal attention mask for transformer.
        
        Returns:
            torch.Tensor: Attention mask with upper triangular set to -inf
        """
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def encode_graph(self, graph: List[Dict[str, torch.Tensor]], des_order: List[List[str]]):
        """
        Encode heterogeneous graph inputs.
        
        Args:
            graph (List[Dict[str, torch.Tensor]]): List of graph dictionaries
            des_order (List[List[str]]): Node type ordering for each graph
            
        Returns:
            torch.Tensor: Graph embeddings
        """
        graph_list = []
        for graph_dict in graph:
            graph_list.append(self.graph_encoder(graph_dict.x_dict, graph_dict.edge_index_dict))
        graph_embeds = []
        assert len(graph_list) == len(des_order)
        for idx, order in enumerate(des_order):
            graph_embeds.extend([graph_list[idx][o] for o in order])
        graph_embeds = torch.cat(graph_embeds, dim=0)
        return graph_embeds

    def encode_text(self, text):
        """
        Encode text inputs through transformer.
        
        Args:
            text: Input text tokens
            
        Returns:
            torch.Tensor: Text embeddings
        """
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, samples):
        """
        Forward pass computing contrastive loss between graph and text.
        
        Args:
            samples: Dictionary containing:
                - graph (List[Dict]): Graph inputs
                - text_input (List[str]): Text inputs
                - des_order (List[List[str]]): Node type ordering
                
        Returns:
            ClipOutput: Model outputs including features and loss
        """
        graph: List[Dict] = samples.get("graph")
        text: List[str] = samples.get("text_input")
        des_order: List[List[str]] = samples.get("des_order")

        if text is not None:
            text = self.tokenizer(text, self.context_length).to(self.token_embedding.weight.device)

        if graph is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_graph(graph, des_order)

        graph_embeds = self.encode_graph(graph, des_order)
        graph_features = F.normalize(graph_embeds, dim=-1)

        text_embeds = self.encode_text(text)
        text_features = F.normalize(text_embeds, dim=-1)
        assert graph_features.shape == text_features.shape

        loss = self.loss(graph_features, text_features, self.logit_scale.exp())

        return ClipOutput(
            intermediate_output=HeteClipOutputFeatures(
                graph_embeds=graph_embeds,
                graph_embeds_proj=graph_features,
                text_embeds=text_embeds,
                text_embeds_proj=text_features,
            ),
            loss=loss,
            logit_scale_exp=self.logit_scale.exp(),
        )

    def extract_features(self, samples):
        """
        Extract features without computing loss.
        
        Similar to forward() but only returns embeddings without loss computation.
        
        Args:
            samples: Dictionary containing graph and text inputs
            
        Returns:
            HeteClipOutputFeatures: Extracted features
        """
        graph: List[Dict] = samples.get("graph")
        text: List[str] = samples.get("text_input")
        des_order: List[List[str]] = samples.get("des_order")

        if text is not None:
            text = self.tokenizer(text)

        if graph is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_graph(graph, des_order)

        graph_embeds = self.encode_graph(graph, des_order)
        graph_features = F.normalize(graph_embeds, dim=-1)

        text_embeds = self.encode_text(text)
        text_features = F.normalize(text_embeds, dim=-1)
        assert graph_features.shape == text_features.shape

        return HeteClipOutputFeatures(
            graph_embeds=graph_embeds,
            graph_embeds_proj=graph_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
        )



class GraphLlamaConfig(LlamaConfig):
    """
    Configuration class for GraphLLaMA model.
    
    Extends LlamaConfig to include graph-specific configuration options.
    """
    model_type = "GraphLlama"

class GraphPretrainConfig:
    """
    Configuration class for graph pre-training.
    
    A simple wrapper that converts dictionary config to object attributes.
    
    Args:
        dictionary (dict): Configuration dictionary
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def load_model_pretrained(model_name, pretrain_model_path):
    """
    Load a pretrained model from checkpoint.
    
    Args:
        model_name: Model class to instantiate
        pretrain_model_path (str): Path to pretrained model checkpoint
        
    Returns:
        Tuple[nn.Module, GraphPretrainConfig]: Loaded model and its configuration
        
    Raises:
        AssertionError: If config.json is missing
    """
    assert os.path.exists(os.path.join(pretrain_model_path, 'config.json')), 'config.json missing'
    with open(os.path.join(pretrain_model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    args = GraphPretrainConfig(config_dict)
    model = model_name(args)
    pkl_files = glob.glob(os.path.join(pretrain_model_path, '*.pkl'))
    state_dict = torch.load(pkl_files[0])
    if 'logit_scale' in state_dict.keys():
        state_dict.pop('logit_scale')
    print('loading graph pre train model')
    model.load_state_dict(state_dict)
    return model, args

def load_metahgt_pretrained(model_name, pretrain_model_path):
    """
    Load a pretrained MetaHGT model from checkpoint.
    
    Args:
        model_name: Should be MetaHGTConv class
        pretrain_model_path (str): Path to pretrained model checkpoint
        
    Returns:
        MetaHGTConv: Loaded model instance
        
    Raises:
        AssertionError: If config files are missing or model_name is incorrect
    """
    assert os.path.exists(os.path.join(pretrain_model_path, 'graph_config.json')), 'graph_config.json missing'
    with open(os.path.join(pretrain_model_path, 'graph_config.json'), 'r') as f:
        graph_config_dict = json.load(f)
    graph_cfg = MetaHGTConvCfg(**graph_config_dict)

    assert os.path.exists(os.path.join(pretrain_model_path, 'text_config.json')), 'text_config.json missing'
    with open(os.path.join(pretrain_model_path, 'text_config.json'), 'r') as f:
        text_config_dict = json.load(f)
    text_cfg = CLIPTextCfg(**text_config_dict)
    
    assert model_name == MetaHGTConv
    model = model_name(
        in_channels=graph_cfg.in_channels,
        out_channels=graph_cfg.out_channels,
        heads=graph_cfg.heads,
        dynamic=graph_cfg.dynamic,
        text_cfg=text_cfg,
    )

    pkl_files = glob.glob(os.path.join(pretrain_model_path, '*.ckpt'))
    state_dict = torch.load(pkl_files[0], map_location='cpu')['state_dict']
    print('loading graph pre train model ...')
    gnn_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.graph_encoder'):
            new_key = key.split('model.graph_encoder.')[1]
            gnn_state_dict[new_key] = value
    model.load_state_dict(gnn_state_dict, strict=False)
    return model

def transfer_param_tograph(clip_graph, gnn):
    """
    Transfer parameters from CLIP graph encoder to GNN.
    
    Args:
        clip_graph: Source CLIP model containing graph encoder
        gnn: Target GNN model
        
    Returns:
        nn.Module: GNN with transferred parameters
    """
    gnn_state_dict = clip_graph.gnn.state_dict()
    gnn.load_state_dict(gnn_state_dict)
    return gnn

class GraphLlamaModel(LlamaModel):
    """
    GraphLLaMA model that combines LLaMA with graph processing capabilities.
    
    Extends LlamaModel to handle graph inputs through various graph neural network
    architectures including MPNN, GCN, and graph transformers.
    
    Args:
        config (LlamaConfig): Model configuration
    """
    config_class = GraphLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(GraphLlamaModel, self).__init__(config)

        if hasattr(config, "graph_tower"):
            if config.graph_tower == 'MPNN':
                self.graph_tower = MPNN(
                    in_channels=config.graph_hidden_size,
                    hidden_channels=config.graph_hidden_size * 2,
                    out_channels=config.graph_hidden_size,
                    dropout=0.1,
                    num_layers=2,
                    if_param=False
                )
            elif config.graph_tower == "clip_gcn_arxiv":
                clip_graph, args = load_model_pretrained(CLIP, config.pretrain_graph_model_path)
                self.graph_tower = GNN(args)
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)
            elif config.graph_tower == "clip_gt":
                clip_graph, args = load_model_pretrained(CLIP, config.pretrain_graph_model_path)
                self.graph_tower = graph_transformer(args)
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)
            elif config.graph_tower == "clip_gt_arxiv":
                clip_graph, args = load_model_pretrained(CLIP, config.pretrain_graph_model_path)
                self.graph_tower = graph_transformer(args)
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)
            elif config.graph_tower == "clip_gt_arxiv_pub":
                clip_graph, args = load_model_pretrained(CLIP, config.pretrain_graph_model_path)
                self.graph_tower = graph_transformer(args)
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)

        if hasattr(config, "use_graph_proj"):
            self.graph_projector = nn.Linear(config.graph_hidden_size, config.hidden_size)

    def get_graph_tower(self):
        """
        Get the graph processing component.
        
        Returns:
            nn.Module: Graph neural network module
        """
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def initialize_graph_modules(self, graph_tower, graph_select_layer,
                               pretrain_graph_mlp_adapter=None, fsdp=None):
        """
        Initialize graph processing modules.
        
        Args:
            graph_tower (str): Type of graph neural network to use
            graph_select_layer (int): Which layer to select features from
            pretrain_graph_mlp_adapter (str, optional): Path to pretrained adapter weights
            fsdp (list, optional): FSDP configuration
        """
        self.config.graph_tower = graph_tower

        if not hasattr(self, 'graph_tower'):
            if self.config.graph_tower == 'MPNN':
                graph_tower = MPNN(
                    in_channels=self.config.graph_hidden_size,
                    hidden_channels=self.config.graph_hidden_size * 2,
                    out_channels=self.config.graph_hidden_size,
                    dropout=0.1,
                    num_layers=2,
                    if_param=False
                )
            elif self.config.graph_tower == "clip_gcn_arxiv":
                clip_graph, args = load_model_pretrained(CLIP, self.config.pretrain_graph_model_path)
                graph_tower = GNN(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
            elif self.config.graph_tower == "clip_gt":
                clip_graph, args = load_model_pretrained(CLIP, self.config.pretrain_graph_model_path)
                graph_tower = graph_transformer(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
            elif self.config.graph_tower == "clip_gt_arxiv":
                clip_graph, args = load_model_pretrained(CLIP, self.config.pretrain_graph_model_path)
                graph_tower = graph_transformer(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
            elif self.config.graph_tower == "clip_gt_arxiv_pub":
                clip_graph, args = load_model_pretrained(CLIP, self.config.pretrain_graph_model_path)
                graph_tower = graph_transformer(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
        else:
            graph_tower = self.graph_tower

        graph_tower.requires_grad_(False)

        if fsdp is not None and len(fsdp) > 0:
            self.graph_tower = [graph_tower]
        else:
            self.graph_tower = graph_tower

        self.config.use_graph_proj = True
        self.config.graph_select_layer = graph_select_layer

        if not hasattr(self, 'graph_projector'):
            self.graph_projector = nn.Linear(self.config.graph_hidden_size, self.config.hidden_size)

        if pretrain_graph_mlp_adapter is not None:
            graph_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
            self.graph_projector.load_state_dict({k.split('.')[-1]: v for k, v in graph_projector_weights.items()})

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass of the GraphLLaMA model.
        
        Processes both text and graph inputs through the model, combining them
        via graph-augmented attention mechanisms.
        
        Args:
            input_ids (torch.LongTensor, optional): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            past_key_values (List[torch.FloatTensor], optional): Cached key/values for faster inference
            inputs_embeds (torch.FloatTensor, optional): Pre-computed input embeddings
            use_cache (bool, optional): Whether to use past key/values
            output_attentions (bool, optional): Whether to output attention weights
            output_hidden_states (bool, optional): Whether to output all hidden states
            graph_data (Data, optional): Input graph data
            return_dict (bool, optional): Whether to return a dictionary output
            
        Returns:
            Union[Tuple, BaseModelOutputWithPast]: Model outputs including hidden states,
                attention weights and past key/values if requested
        """
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        graph_tower = self.get_graph_tower()
        if graph_tower is not None and (input_ids.shape[1] != 1 or self.training) and graph_data is not None:
            with torch.no_grad():
                if type(graph_data) is list:
                    graph_node_features = []
                    if type(graph_data[0]) is Data:
                        for g in graph_data:
                            node_forward_out = graph_tower(g)
                            graph_node_features.append(node_forward_out)
                    elif type(graph_data[0]) is dict:
                        for g_dict in graph_data:
                            node_forward_out_1 = graph_tower(g_dict['graph_1'])
                            node_forward_out_2 = graph_tower(g_dict['graph_2'])
                            graph_node_features.append(node_forward_out_1)
                            graph_node_features.append(node_forward_out_2)
                else:
                    raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')

            if type(graph_data) is list:
                graph_node_features = [self.graph_projector(node_feature) for node_feature in graph_node_features]
            else:
                raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')

            dummy_graph_features = torch.zeros(256, 128, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_graph_features = self.graph_projector(dummy_graph_features)

            new_input_embeds = []
            cur_graph_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == graph_tower.config.graph_patch_token).sum() == 0:
                    cur_input_embeds = cur_input_embeds + (0. * dummy_graph_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_graph_idx += 1
                    continue
                if graph_tower.config.use_graph_start_end:
                    cur_graph_features = graph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    if (cur_input_ids == graph_tower.config.graph_start_token).sum() != (cur_input_ids == graph_tower.config.graph_end_token).sum():
                        raise ValueError("The number of graph start tokens and graph end tokens should be the same.")
                    graph_start_tokens = torch.where(cur_input_ids == graph_tower.config.graph_start_token)[0]
                    for graph_start_token_pos in graph_start_tokens:
                        cur_graph_features = graph_node_features[cur_graph_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_graph_features.shape[0]
                        if cur_input_ids[graph_start_token_pos + num_patches + 1] != graph_tower.config.graph_end_token:
                            raise ValueError("The graph end token should follow the graph start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:graph_start_token_pos].detach(), cur_input_embeds[graph_start_token_pos:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:graph_start_token_pos + num_patches + 2], cur_input_embeds[graph_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:]), dim=0)
                        cur_graph_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_graph_features = graph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    if (cur_input_ids == graph_tower.config.graph_patch_token).sum() != num_patches:
                        raise ValueError("The number of graph patch tokens should be the same as the number of graph patches.")
                    masked_indices = torch.where(cur_input_ids == graph_tower.config.graph_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The graph patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_graph_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_graph_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_graph_idx += 1

            assert cur_graph_idx == len(graph_node_features)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(GraphLlamaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class GraphLlamaForCausalLM(LlamaForCausalLM):
    """
    GraphLLaMA model for causal language modeling.
    
    Extends LlamaForCausalLM to support graph-augmented language modeling by
    incorporating graph structure information into the generation process.
    
    Args:
        config (GraphLlamaConfig): Model configuration
    """
    config_class = GraphLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = GraphLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        """Get the underlying GraphLlamaModel."""
        return self.model

    def get_graph_tower(self):
        """Get the graph processing component."""
        return self.get_model().get_graph_tower()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            graph_data=graph_data
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepare inputs for text generation.
        
        Args:
            input_ids: Input token IDs
            past_key_values: Cached key/values from previous forward passes
            attention_mask: Attention mask
            inputs_embeds: Pre-computed input embeddings
            **kwargs: Additional arguments
            
        Returns:
            dict: Dictionary of model inputs
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph_data": [kwargs.get("graph_data", None)],
            }
        )
        return model_inputs

    def initialize_graph_tokenizer(self, use_graph_start_end, tokenizer, device,
                                 tune_graph_mlp_adapter=False, pretrain_graph_mlp_adapter=None):
        """
        Initialize tokenizer for graph inputs.
        
        Args:
            use_graph_start_end (bool): Whether to use special graph tokens
            tokenizer: Base tokenizer to extend
            device: Device to place new tokens on
            tune_graph_mlp_adapter (bool): Whether to tune graph MLP adapter
            pretrain_graph_mlp_adapter (str, optional): Path to pretrained adapter
        """
        vision_config = self.get_graph_tower().config
        vision_config.use_graph_start_end = use_graph_start_end
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if use_graph_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.graph_start_token, vision_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_graph_mlp_adapter:
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_graph_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]


AutoConfig.register("GraphLlama", GraphLlamaConfig)
AutoModelForCausalLM.register(GraphLlamaConfig, GraphLlamaForCausalLM)


class HeteroLlamaConfig(LlamaConfig):
    """
    Configuration class for HeteroLLaMA model.
    
    Extends LlamaConfig to include configuration options for heterogeneous graph processing.
    """
    model_type = "HeteroLlama"

class HeteroLlamaModel(LlamaModel):
    """
    HeteroLLaMA model that handles heterogeneous graph inputs.
    
    Extends LlamaModel to process heterogeneous graphs with different node and edge types
    through specialized graph neural networks.
    
    Args:
        config (LlamaConfig): Model configuration
    """
    
    config_class = HeteroLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(HeteroLlamaModel, self).__init__(config)

        if hasattr(config, "graph_tower"):
            if config.graph_tower == 'MPNN':
                self.graph_tower = MPNN(
                    in_channels=config.graph_hidden_size,
                    hidden_channels=config.graph_hidden_size * 2,
                    out_channels=config.graph_hidden_size,
                    dropout=0.1,
                    num_layers=2,
                    if_param=False
                )
            elif config.graph_tower == "meta_hgt":
                self.graph_tower = load_metahgt_pretrained(MetaHGTConv, config.pretrain_graph_model_path)

        if hasattr(config, "use_graph_proj"):
            self.graph_projector = nn.Linear(config.graph_hidden_size, config.hidden_size)

    def get_graph_tower(self):
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def initialize_graph_modules(self, graph_tower, graph_select_layer,
                               pretrain_graph_mlp_adapter=None, fsdp=None):
        self.config.graph_tower = graph_tower

        if not hasattr(self, 'graph_tower'):
            if self.config.graph_tower == 'MPNN':
                graph_tower = MPNN(
                    in_channels=self.config.graph_hidden_size,
                    hidden_channels=self.config.graph_hidden_size * 2,
                    out_channels=self.config.graph_hidden_size,
                    dropout=0.1,
                    num_layers=2,
                    if_param=False
                )
            elif self.config.graph_tower == "meta_hgt":
                graph_tower = load_metahgt_pretrained(MetaHGTConv, self.config.pretrain_graph_model_path)
        else:
            graph_tower = self.graph_tower

        graph_tower.requires_grad_(False)

        if fsdp is not None and len(fsdp) > 0:
            self.graph_tower = [graph_tower]
        else:
            self.graph_tower = graph_tower

        self.config.use_graph_proj = True
        self.config.graph_select_layer = graph_select_layer

        if not hasattr(self, 'graph_projector'):
            self.graph_projector = nn.Linear(self.config.graph_hidden_size, self.config.hidden_size)

        if pretrain_graph_mlp_adapter is not None:
            graph_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
            self.graph_projector.load_state_dict({k.split('.')[-1]: v for k, v in graph_projector_weights.items()})

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
        hetero_key_order: Optional[List[List[str]]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass of the HeteroLLaMA model.
        
        Processes heterogeneous graph inputs along with text through the model.
        
        Args:
            input_ids (torch.LongTensor, optional): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            past_key_values (List[torch.FloatTensor], optional): Cached key/values
            inputs_embeds (torch.FloatTensor, optional): Pre-computed input embeddings
            use_cache (bool, optional): Whether to use past key/values
            output_attentions (bool, optional): Whether to output attention weights
            output_hidden_states (bool, optional): Whether to output all hidden states
            graph_data (Data, optional): Input heterogeneous graph data
            return_dict (bool, optional): Whether to return a dictionary output
            hetero_key_order (List[List[str]], optional): Node type ordering for heterogeneous graphs
            
        Returns:
            Union[Tuple, BaseModelOutputWithPast]: Model outputs including hidden states,
                attention weights and past key/values if requested
        """
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        graph_tower = self.get_graph_tower()
        if graph_tower is not None and (input_ids.shape[1] != 1 or self.training) and graph_data is not None:
            with torch.no_grad():
                if type(graph_data) is list:
                    graph_node_features = []
                    if type(graph_data[0]) is Data:
                        for g in graph_data:
                            node_forward_out = graph_tower(g.x_dict, g.edge_index_dict)
                            graph_node_features.append(node_forward_out)
                    elif type(graph_data[0]) is dict:
                        for g_dict in graph_data:
                            node_forward_out_1 = graph_tower(g_dict['graph_1'])
                            node_forward_out_2 = graph_tower(g_dict['graph_2'])
                            graph_node_features.append(node_forward_out_1)
                            graph_node_features.append(node_forward_out_2)
                else:
                    raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')

            if type(graph_data) is list:
                graph_node_features_list = []
                for idx, order in enumerate(hetero_key_order):
                    graph_node_features_list.extend([graph_node_features[idx][o] for o in order])
                graph_node_features = [self.graph_projector(node_feature) for node_feature in graph_node_features_list]
            else:
                raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')

            dummy_graph_features = torch.zeros(256, 128, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_graph_features = self.graph_projector(dummy_graph_features)

            new_input_embeds = []
            cur_graph_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == graph_tower.config.graph_patch_token).sum() == 0:
                    cur_input_embeds = cur_input_embeds + (0. * dummy_graph_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_graph_idx += 1
                    continue
                if graph_tower.config.use_graph_start_end:
                    cur_graph_features = graph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    if (cur_input_ids == graph_tower.config.graph_start_token).sum() != (cur_input_ids == graph_tower.config.graph_end_token).sum():
                        raise ValueError("The number of graph start tokens and graph end tokens should be the same.")
                    graph_start_tokens = torch.where(cur_input_ids == graph_tower.config.graph_start_token)[0]
                    for graph_start_token_pos in graph_start_tokens:
                        cur_graph_features = graph_node_features[cur_graph_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_graph_features.shape[0]
                        if cur_input_ids[graph_start_token_pos + num_patches + 1] != graph_tower.config.graph_end_token:
                            raise ValueError("The graph end token should follow the graph start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:graph_start_token_pos].detach(), cur_input_embeds[graph_start_token_pos:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:graph_start_token_pos + num_patches + 2], cur_input_embeds[graph_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:]), dim=0)
                        cur_graph_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_graph_features = graph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    if (cur_input_ids == graph_tower.config.graph_patch_token).sum() != num_patches:
                        raise ValueError("The number of graph patch tokens should be the same as the number of graph patches.")
                    masked_indices = torch.where(cur_input_ids == graph_tower.config.graph_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The graph patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_graph_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_graph_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_graph_idx += 1

            assert cur_graph_idx == len(graph_node_features)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(HeteroLlamaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class HeteroLlamaForCausalLM(LlamaForCausalLM):
    """
    HeteroLLaMA model for causal language modeling with heterogeneous graphs.
    
    Extends LlamaForCausalLM to support language modeling conditioned on
    heterogeneous graph structures.
    
    Args:
        config (HeteroLlamaConfig): Model configuration
    """
    config_class = HeteroLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = HeteroLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        """Get the underlying HeteroLlamaModel."""
        return self.model

    def get_graph_tower(self):
        """Get the heterogeneous graph processing component."""
        return self.get_model().get_graph_tower()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
        hetero_key_order: Optional[List[List[str]]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            graph_data=graph_data,
            hetero_key_order=hetero_key_order
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if kwargs.get("graph_data") is None:
            model_inputs.update(
                {
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                    "graph_data": None,
                    "hetero_key_order": [kwargs.get("hetero_key_order", None)]
                }
            )
        else:
            model_inputs.update(
                {
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                    "graph_data": [kwargs.get("graph_data", None)],
                    "hetero_key_order": [kwargs.get("hetero_key_order", None)]
                }
            )
        return model_inputs

    def initialize_graph_tokenizer(self, use_graph_start_end, tokenizer, device,
                                 tune_graph_mlp_adapter=False, pretrain_graph_mlp_adapter=None):
        vision_config = self.get_graph_tower().config
        vision_config.use_graph_start_end = use_graph_start_end
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if use_graph_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.graph_start_token, vision_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_graph_mlp_adapter:
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_graph_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]


AutoConfig.register("HeteroLlama", HeteroLlamaConfig)
AutoModelForCausalLM.register(HeteroLlamaConfig, HeteroLlamaForCausalLM)


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    Tokenize a list of strings.
    
    Args:
        strings (Sequence[str]): List of input strings to tokenize
        tokenizer (PreTrainedTokenizer): Tokenizer to use
        
    Returns:
        Dict: Dictionary containing:
            - input_ids: Token IDs
            - labels: Labels for language modeling
            - input_ids_lens: Lengths of input sequences
            - labels_lens: Lengths of label sequences
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    """
    Mask target tokens for dialogue modeling.
    
    Args:
        target: Target token IDs to mask
        tokenized_lens: List of token sequence lengths
        speakers: List of speaker identifiers
    """
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Resize tokenizer and embedding layers to accommodate new special tokens.
    
    Args:
        special_tokens_dict (Dict): Dictionary of special tokens to add
        tokenizer (PreTrainedTokenizer): Tokenizer to modify
        model (PreTrainedModel): Model whose embeddings need resizing
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwrap a model from potential containers.
    
    Args:
        model (nn.Module): Model to unwrap
        
    Returns:
        nn.Module: Unwrapped model
    """
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

def find_all_linear_names(model):
    """
    Find all linear layer names in the model.
    
    Args:
        model: Model to analyze
        
    Returns:
        list: List of linear layer names, excluding lm_head
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Collects the state dict and saves model to disk safely.
    
    Handles DeepSpeed and regular model saving with proper synchronization.
    
    Args:
        trainer (transformers.Trainer): HuggingFace trainer instance
        output_dir (str): Directory to save model
    """
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def get_peft_state_maybe_zero_3(named_params, bias):
    """
    Get PEFT state dict handling DeepSpeed ZeRO-3.
    
    Args:
        named_params: Named parameters from model
        bias (str): Bias handling mode ('none', 'all', or 'lora_only')
        
    Returns:
        dict: State dict with gathered parameters
        
    Raises:
        NotImplementedError: If bias mode is not recognized
    """
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """
    Get non-LoRA PEFT state dict handling DeepSpeed ZeRO-3.
    
    Args:
        named_params: Named parameters from model
        require_grad_only (bool, optional): Whether to only include parameters requiring gradients. Defaults to True
        
    Returns:
        dict: State dict with gathered parameters
    """
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def maybe_zero_3(param, ignore_status=False, name=None):
    """
    Handle DeepSpeed ZeRO-3 parameters.
    
    Args:
        param: Model parameter
        ignore_status (bool, optional): Whether to ignore parameter status. Defaults to False
        name (str, optional): Parameter name for logging. Defaults to None
        
    Returns:
        torch.Tensor: Gathered parameter data
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def download_cached_file(url: str, check_hash: bool = True, progress: bool = True) -> str:
    """
    Download a file from URL and cache it locally.
    
    Args:
        url (str): URL to download from
        check_hash (bool, optional): Whether to verify file hash. Defaults to True
        progress (bool, optional): Whether to show progress bar. Defaults to True
        
    Returns:
        str: Path to cached file
    """
    from torch.hub import download_url_to_file, get_dir
    from urllib.parse import urlparse
    import os

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(get_dir(), filename)

    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)

    return cached_file


class MetaHeteroLinear(nn.Module):
    """
    Meta-learning based linear layer for heterogeneous inputs.
    
    Implements a dynamic or static linear transformation that can handle
    different types of inputs through meta-learning.
    
    Args:
        text_width (int): Width of text embeddings for generating weights
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        dynamic (bool, optional): Whether to use dynamic weight generation. Defaults to True
    """
    def __init__(self, text_width: int, in_features: int, out_features: int, dynamic: bool = True):
        super().__init__()
        self.text_width = text_width
        self.in_features = in_features
        self.out_features = out_features
        self.dynamic = dynamic

        if dynamic:
            self.weight_proj = nn.Linear(text_width, in_features * out_features)
            self.bias_proj = nn.Linear(text_width, out_features)
        else:
            self.weight = nn.Parameter(torch.randn(in_features, out_features) / in_features ** 0.5)
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor, type_id: Tensor, type_embed_dict: Dict[int, Tensor]) -> Tensor:
        """
        Forward pass with type-specific transformations.
        
        Args:
            x (Tensor): Input features
            type_id (Tensor): Type IDs for each input
            type_embed_dict (Dict[int, Tensor]): Type embeddings dictionary
            
        Returns:
            Tensor: Transformed features
        """
        if self.dynamic:
            weight = []
            bias = []
            for i in range(len(type_embed_dict)):
                type_embed = type_embed_dict[i]
                cur_weight = self.weight_proj(type_embed).view(self.in_features, self.out_features)
                cur_bias = self.bias_proj(type_embed)
                weight.append(cur_weight)
                bias.append(cur_bias)
            weight = torch.stack(weight)
            bias = torch.stack(bias)
            weight = weight[type_id]
            bias = bias[type_id]
        else:
            weight = self.weight
            bias = self.bias

        return F.linear(x, weight, bias)

class MetaHeteroDictLinear(nn.Module):
    """
    Meta-learning based linear layer for dictionary of heterogeneous inputs.
    
    Similar to MetaHeteroLinear but handles dictionary inputs where each key
    represents a different node/edge type.
    
    Args:
        text_width (int): Width of text embeddings for generating weights
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        dynamic (bool, optional): Whether to use dynamic weight generation. Defaults to True
    """
    def __init__(self, text_width: int, in_features: int, out_features: int, dynamic: bool = True):
        super().__init__()
        self.text_width = text_width
        self.in_features = in_features
        self.out_features = out_features
        self.dynamic = dynamic

        if dynamic:
            self.weight_proj = nn.Linear(text_width, in_features * out_features)
            self.bias_proj = nn.Linear(text_width, out_features)
        else:
            self.weight = nn.Parameter(torch.randn(in_features, out_features) / in_features ** 0.5)
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x_dict: Dict[str, Tensor], type_embed_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass with type-specific transformations on dictionary inputs.
        
        Args:
            x_dict (Dict[str, Tensor]): Dictionary of input features by type
            type_embed_dict (Dict[str, Tensor]): Dictionary of type embeddings
            
        Returns:
            Dict[str, Tensor]: Dictionary of transformed features by type
        """
        out_dict = {}
        for node_type, x in x_dict.items():
            if self.dynamic:
                type_embed = type_embed_dict[node_type]
                weight = self.weight_proj(type_embed).view(self.in_features, self.out_features)
                bias = self.bias_proj(type_embed)
            else:
                weight = self.weight
                bias = self.bias
            out_dict[node_type] = F.linear(x, weight, bias)
        return out_dict


class HeteClipLoss(nn.Module):
    """
    Contrastive loss for heterogeneous CLIP model.
    
    Implements InfoNCE-style contrastive loss between graph and text embeddings
    with temperature scaling.
    """
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, graph_features, text_features, logit_scale):
        """
        Compute contrastive loss between graph and text features.
        
        Args:
            graph_features (Tensor): Graph embeddings
            text_features (Tensor): Text embeddings
            logit_scale (Tensor): Temperature scaling factor
            
        Returns:
            Tensor: Contrastive loss value
        """
        device = graph_features.device
        local_batch_size = graph_features.shape[0]

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * [1]
            self.labels = torch.LongTensor(self.labels).to(device)
            self.last_local_batch_size = local_batch_size


        graph_features = F.normalize(graph_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits_per_graph = logit_scale * graph_features @ text_features.t()
        logits_per_text = logits_per_graph.t()

        loss = (
            F.cross_entropy(logits_per_graph, self.labels) +
            F.cross_entropy(logits_per_text, self.labels)
        ) / 2

        return loss

openai_imagenet_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photograph of a {c}.",
    lambda c: f"an image of a {c}.",
    lambda c: f"a picture of a {c}.",
]