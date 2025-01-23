# Copyright 2023 The HiGPT Team
# Licensed under the Apache License, Version 2.0

from typing import List, Optional, Tuple, Union, Dict
import os
import os.path as osp
import glob
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_scatter import scatter_add
from collections import OrderedDict
from transformers.configuration_utils import PretrainedConfig

# Constants
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"

# MetaHGT Model Configuration
class MetaHGTConvCfg:
    def __init__(self, in_channels=768, out_channels=768, heads=8, dynamic=True, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dynamic = dynamic
        for k, v in kwargs.items():
            setattr(self, k, v)

# Text Configuration for CLIP
class CLIPTextCfg:
    def __init__(self, context_length=77, vocab_size=49408, width=512, heads=8, layers=12, **kwargs):
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.heads = heads
        self.layers = layers
        for k, v in kwargs.items():
            setattr(self, k, v)

# HiGPT Configuration
class HiGPTConfig(LlamaConfig):
    model_type = "HiGPT"
    
    def __init__(
        self,
        graph_hidden_size: int = 768,
        graph_intermediate_size: int = 3072,
        graph_num_hidden_layers: int = 12,
        graph_num_attention_heads: int = 12,
        graph_max_position_embeddings: int = 512,
        graph_type_vocab_size: int = 2,
        graph_vocab_size: int = 50000,
        graph_layer_norm_eps: float = 1e-12,
        graph_hidden_dropout_prob: float = 0.1,
        graph_attention_probs_dropout_prob: float = 0.1,
        graph_initializer_range: float = 0.02,
        graph_type_vocab_size_a: int = 2,
        graph_type_vocab_size_b: int = 2,
        is_decoder: bool = True,
        is_encoder_decoder: bool = False,
        **kwargs,
    ):
        super().__init__(is_decoder=is_decoder, is_encoder_decoder=is_encoder_decoder, **kwargs)
        self.graph_hidden_size = graph_hidden_size
        self.graph_intermediate_size = graph_intermediate_size
        self.graph_num_hidden_layers = graph_num_hidden_layers
        self.graph_num_attention_heads = graph_num_attention_heads
        self.graph_max_position_embeddings = graph_max_position_embeddings
        self.graph_type_vocab_size = graph_type_vocab_size
        self.graph_vocab_size = graph_vocab_size
        self.graph_layer_norm_eps = graph_layer_norm_eps
        self.graph_hidden_dropout_prob = graph_hidden_dropout_prob
        self.graph_attention_probs_dropout_prob = graph_attention_probs_dropout_prob
        self.graph_initializer_range = graph_initializer_range
        self.graph_type_vocab_size_a = graph_type_vocab_size_a
        self.graph_type_vocab_size_b = graph_type_vocab_size_b

# Graph Neural Network Layers

def gcn_conv(h, edge_index):
    """GCN卷积层的实现"""
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

    rows, cols = edge_index
    edge_msg = h[rows, :] * torch.unsqueeze(edge_weight, dim=-1)
    col_embeds = h[cols, :]
    tem = torch.zeros([N, node_feas]).to(edge_msg.device)
    rows = rows.to(edge_msg.device)
    h_prime = tem.index_add_(0, rows, edge_msg)
    return h_prime

class MPNN(nn.Module):
    """消息传递神经网络的实现"""
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super(MPNN, self).__init__()
        self.config = PretrainedConfig()
        self.dropout = kwargs.get('dropout', 0.1)
        self.num_layers = kwargs.get('num_layers', 2)
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.if_param = kwargs.get('if_param', True)

        if self.if_param:
            self.fcs = nn.ModuleList([])
            self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
            for _ in range(self.num_layers - 2):
                self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias))
            self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias))
            self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.fcs:
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, g, use_conv=True):
        x = g.graph_node
        edge_index = g.edge_index
        try:
            device = next(self.parameters()).device
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

class LayerNorm(nn.LayerNorm):
    """处理fp16的LayerNorm实现"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    """快速GELU激活函数"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

def PositionalEncoding(q_len, d_model, normalize=True):
    """位置编码实现"""
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

def pos_encoding(pe, learn_pe, nvar, d_model):
    """位置编码生成器"""
    if pe == None:
        W_pos = torch.empty((nvar, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((nvar, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((nvar, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((nvar, 1))
        nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((nvar, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(nvar, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

class ResidualAttentionBlock(nn.Module):
    """残差注意力块"""
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """Transformer编码器"""
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class GTLayer(nn.Module):
    """Graph Transformer层"""
    def __init__(self, args):
        super(GTLayer, self).__init__()
        self.qTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.att_d_model, args.att_d_model)))
        self.kTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.att_d_model, args.att_d_model)))
        self.vTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.att_d_model, args.att_d_model)))
        if args.att_norm:
            self.norm = nn.LayerNorm(args.att_d_model, eps=1e-6)
        self.args = args

    def forward(self, g, embeds):
        rows, cols = g.edge_index
        nvar, _ = embeds.shape
        rowEmbeds = embeds[rows, :]
        colEmbeds = embeds[cols, :]
        evar, _ = rowEmbeds.shape

        qEmbeds = (rowEmbeds @ self.qTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        
        att = torch.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = torch.clamp(att, -10.0, 10.0)
        expAtt = torch.exp(att)
        
        tem = torch.zeros([nvar, self.args.head]).to(expAtt.device, dtype=expAtt.dtype)
        rows = rows.to(expAtt.device)
        attNorm = (tem.index_add_(0, rows, expAtt))[rows, :]
        att = expAtt / (attNorm + 1e-8)
        
        resEmbeds = torch.einsum('eh, ehd -> ehd', att, vEmbeds).view([evar, self.args.att_d_model])
        tem = torch.zeros([nvar, self.args.att_d_model]).to(resEmbeds.device, dtype=resEmbeds.dtype)
        rows = rows.to(resEmbeds.device)
        tem = tem.to(resEmbeds.dtype)
        resEmbeds = tem.index_add_(0, rows, resEmbeds)
        resEmbeds = resEmbeds + embeds
        if self.args.att_norm:
            resEmbeds = self.norm(resEmbeds)
        return resEmbeds

class GraphTransformer(nn.Module):
    """Graph Transformer模型"""
    def __init__(self, args):
        super(GraphTransformer, self).__init__()
        self.config = PretrainedConfig()
        self.gtLayers = nn.Sequential(*[GTLayer(args) for i in range(args.gt_layers)])
        self.W_pos = pos_encoding('zeros', True, 1, args.att_d_model)
        self.W_P = nn.Linear(args.gnn_input, args.att_d_model)
        self.dropout = nn.Dropout(0.1)
        self.inverW_P = nn.Linear(args.att_d_model, args.gnn_output)
        self.args = args

    def forward(self, g):
        device = next(self.parameters()).device
        g = g.to(device)
        x = g.graph_node
        z = self.W_P(x)
        if self.args.if_pos:
            embeds = self.dropout(z + self.W_pos)
        else:
            embeds = self.dropout(z)
        for gt in self.gtLayers:
            embeds = gt(g, embeds)
        ret = self.inverW_P(embeds)
        return ret

# Graph Feature Extractor
class GraphFeatureExtractor(nn.Module):
    def __init__(self, config: HiGPTConfig):
        super().__init__()
        self.config = config
        
        # Graph embedding layers
        self.node_embeddings = nn.Linear(config.graph_hidden_size, config.hidden_size)
        self.edge_embeddings = nn.Linear(config.graph_hidden_size, config.hidden_size)
        self.graph_position_embeddings = nn.Embedding(config.graph_max_position_embeddings, config.hidden_size)
        
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.graph_layer_norm_eps)
        self.dropout = nn.Dropout(config.graph_hidden_dropout_prob)
        
        # Graph attention layers
        self.graph_encoder = MetaHGTConv(
            in_channels=config.graph_hidden_size,
            out_channels=config.hidden_size,
            heads=config.graph_num_attention_heads,
            dynamic=True
        )
        
    def forward(
        self,
        graph_data: Union[Data, Dict],
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        # Process heterogeneous graph data
        if isinstance(graph_data, dict):
            node_features = graph_data['x_dict']
            edge_indices = graph_data['edge_index_dict']
        else:
            node_features = {'graph': graph_data.x}
            edge_indices = {'graph': graph_data.edge_index}
        
        # Apply graph neural network
        graph_outputs = self.graph_encoder(node_features, edge_indices)
        
        # Process each node type
        processed_features = {}
        for node_type, features in graph_outputs.items():
            # Apply position embeddings
            position_ids = torch.arange(features.size(0), device=features.device)
            position_embeddings = self.graph_position_embeddings(position_ids)
            
            # Combine features
            features = self.node_embeddings(features) + position_embeddings
            features = self.LayerNorm(features)
            features = self.dropout(features)
            
            processed_features[node_type] = features
            
        return processed_features

# HiGPT Core Model
class HiGPTModel(LlamaModel):
    config_class = HiGPTConfig

    def __init__(self, config: HiGPTConfig):
        super(HiGPTModel, self).__init__(config)

        if hasattr(config, "graph_tower"):
            self.graph_tower = GraphFeatureExtractor(config)
            
        if hasattr(config, "use_graph_proj"):
            self.graph_projector = nn.Linear(config.graph_hidden_size, config.hidden_size)
            
        self.post_init()

    def get_graph_tower(self):
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def initialize_graph_modules(self, graph_tower, graph_select_layer,
                               pretrain_graph_mlp_adapter=None, fsdp=None):
        self.config.graph_tower = graph_tower

        if not hasattr(self, 'graph_tower'):
            self.graph_tower = GraphFeatureExtractor(self.config)
        else:
            self.graph_tower = self.graph_tower
            
        self.graph_tower.requires_grad_(False)

        if fsdp is not None and len(fsdp) > 0:
            self.graph_tower = [self.graph_tower]
        
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
        graph_data: Optional[Union[Data, Dict]] = None,
        hetero_key_order: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # Process graph data if provided
        if self.graph_tower is not None and graph_data is not None:
            graph_features = self.graph_tower(graph_data)
            
            if hetero_key_order is not None:
                # Combine features based on key order
                combined_features = []
                for key in hetero_key_order:
                    if key in graph_features:
                        combined_features.append(graph_features[key])
                graph_features = torch.cat(combined_features, dim=0)
            else:
                # Use all features
                graph_features = torch.cat(list(graph_features.values()), dim=0)
            
            # Project graph features
            if hasattr(self, 'graph_projector'):
                graph_features = self.graph_projector(graph_features)
            
            # Combine with text embeddings
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            
            # Insert graph features at appropriate positions
            if hasattr(self.config, 'use_graph_start_end') and self.config.use_graph_start_end:
                # Use start/end tokens to insert graph features
                graph_start_tokens = torch.where(input_ids == self.config.graph_start_token)[1]
                for idx, start_token in enumerate(graph_start_tokens):
                    inputs_embeds[idx, start_token+1:start_token+1+graph_features.size(0)] = graph_features
            else:
                # Insert at graph token positions
                graph_tokens = torch.where(input_ids == self.config.graph_patch_token)[1]
                for idx, token_pos in enumerate(graph_tokens):
                    inputs_embeds[idx, token_pos] = graph_features[idx]
        
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

# HiGPT Causal Language Model
class HiGPTForCausalLM(LlamaForCausalLM):
    config_class = HiGPTConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = HiGPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def get_graph_tower(self):
        return self.get_model().get_graph_tower()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph_data: Optional[Union[Data, Dict]] = None,
        hetero_key_order: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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
        self, 
        input_ids, 
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
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
                "graph_data": kwargs.get("graph_data", None),
                "hetero_key_order": kwargs.get("hetero_key_order", None)
            }
        )
        return model_inputs

    def initialize_graph_tokenizer(
        self, 
        use_graph_start_end=False,
        tokenizer=None,
        device='cuda',
        tune_graph_mlp_adapter=False,
        pretrain_graph_mlp_adapter=None
    ):
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if use_graph_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            self.config.graph_start_token, self.config.graph_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN]
            )

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
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                                  f"Current: {input_embeddings.shape}. Number of new tokens: {num_new_tokens}.")

        self.config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]

# Register models
AutoConfig.register("HiGPT", HiGPTConfig)
AutoModelForCausalLM.register(HiGPTConfig, HiGPTForCausalLM)

class CLIP(nn.Module):
    """CLIP模型实现"""
    def __init__(self, args):
        super().__init__()

        self.context_length = args.context_length
        self.args = args
        self.edge_coef = args.edge_coef

        if args.gnn_type == 'gcn':
            self.gnn = MPNN(args)
        elif args.gnn_type == 'gt':
            self.gnn = GraphTransformer(args)
            
        self.transformer = Transformer(
            width=args.transformer_width,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = args.vocab_size
        self.token_embedding = nn.Embedding(args.vocab_size, args.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, args.transformer_width))
        self.ln_final = LayerNorm(args.transformer_width)
        self.text_projection = nn.Parameter(torch.empty(args.transformer_width, args.embed_dim))

        if args.gnn_type == 'gcn':
            self.dtype = self.gnn.vars[0].dtype
        elif args.gnn_type == 'gt':
            self.dtype = self.gnn.W_pos.dtype

        self.optim = torch.optim.Adam([
            {'params': self.token_embedding.weight},
            {'params': self.positional_embedding},
            {'params': self.transformer.parameters()},
            {'params': self.text_projection},
            {'params': self.gnn.parameters()}
        ], lr=args.lr)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

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
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def encode_image(self, idx_train, g):
        embs = self.gnn(g)
        idx_train = idx_train.to(embs.device)
        train_embs = embs[idx_train]
        return train_embs

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = x @ self.text_projection
        return x

    def forward(self, g, s_n, t_n, s_n_text, t_n_text, training=True):
        s_image_features = self.encode_image(s_n, g)
        s_text_features = self.encode_text(s_n_text)
        t_text_features = self.encode_text(t_n_text)
        
        t_text_features = t_text_features.reshape(s_image_features.shape[0], self.args.neigh_num, self.args.gnn_output)
        t_text_features = torch.mean(t_text_features, dim=1, keepdim=False)
        
        s_image_features = s_image_features / s_image_features.norm(dim=-1, keepdim=True)
        s_text_features = s_text_features / s_text_features.norm(dim=-1, keepdim=True)
        t_text_features = t_text_features / t_text_features.norm(dim=-1, keepdim=True)

        labels = torch.arange(s_image_features.shape[0]).cuda()
        return s_image_features, s_text_features, t_text_features, labels 