import os
import sys
import random
import json
import torch
import dgl
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import glob
import os.path as osp
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data, HeteroData
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn import CrossEntropyLoss

# Import local models
from ggfm.models import HiGPTForCausalLM, MetaHGTConv

# Constants
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    graph_tower: Optional[str] = field(
        default="MetaHGT",
        metadata={"help": "The type of graph tower to use"}
    )
    graph_select_layer: Optional[int] = field(
        default=-1,
        metadata={"help": "Which layer to select from graph tower"}
    )
    pretrain_graph_mlp_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained graph MLP adapter"}
    )
    use_graph_start_end: bool = field(
        default=False,
        metadata={"help": "Whether to use graph start/end tokens"}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_path: str = field(
        default="./processed_datasets",
        metadata={"help": "Path to the training data."}
    )
    graph_root: str = field(
        default="ggfm/ggfm/datasets",
        metadata={"help": "Root directory containing graph data"}
    )
    output_dir: str = field(
        default="./processed_datasets",
        metadata={"help": "Output directory for processed data and checkpoints"}
    )
    num_shot: int = field(
        default=0,
        metadata={"help": "Number of shots for few-shot learning"}
    )

def assign_paper_labels(original_graph_path, processed_graph_path):
    """Assign labels to papers"""
    print("Loading original dataset...")
    with open(original_graph_path, 'rb') as f:
        original_g = pickle.load(f)
    
    print("Loading processed dataset...")
    processed_glist, _ = dgl.load_graphs(processed_graph_path)
    processed_g = processed_glist[0]
    
    # Get L2 level paper-field connections
    field_features = original_g.node_feature['field']
    paper_field_edges = original_g.edge_list['paper']['field']
    l2_edges = paper_field_edges.get('rev_PF_in_L2', {})
    
    # Filter L2 level fields
    l2_fields = field_features[field_features['attr'] == 'L2']
    
    # Assign labels to each paper
    num_papers = processed_g.num_nodes('paper')
    paper_labels = torch.full((num_papers,), -1, dtype=torch.long)
    
    # Track unlabeled papers
    unlabeled_papers = []
    
    # Assign a random L2 field label to each paper
    for paper_id in range(num_papers):
        if paper_id in l2_edges:
            fields = [f for f in l2_edges[paper_id] if f in l2_fields.index]
            if fields:
                chosen_field = random.choice(fields)
                field_label = l2_fields.loc[chosen_field, 'label']
                paper_labels[paper_id] = field_label
            else:
                unlabeled_papers.append(paper_id)
        else:
            unlabeled_papers.append(paper_id)
    
    print(f"Total number of papers: {num_papers}")
    print(f"Number of papers with labels: {num_papers - len(unlabeled_papers)}")
    print(f"Number of unlabeled papers: {len(unlabeled_papers)}")
    
    # Add labels to paper nodes in the graph
    processed_g.nodes['paper'].data['label'] = paper_labels
    
    # Save as new file
    output_path = osp.join(osp.dirname(processed_graph_path), "labeled_field_hg.bin")
    dgl.save_graphs(output_path, [processed_g])
    print(f"Saved labeled graph to: {output_path}")
    
    # Create and save label to field mapping
    label_to_field = {}
    unique_labels = sorted(l2_fields['label'].unique())
    for label in unique_labels:
        fields_with_label = l2_fields[l2_fields['label'] == label]
        field_infos = []
        for _, field_info in fields_with_label.iterrows():
            field_infos.append({
                'field_id': int(field_info.name),
                'name': field_info['name'],
                'type': field_info['type'],
                'attr': field_info['attr']
            })
        label_to_field[str(label)] = field_infos
    
    # Add special marker
    label_to_field['-1'] = [{
        'field_id': -1,
        'name': 'Unlabeled',
        'type': 'special',
        'attr': 'none'
    }]
    
    # Save mapping to json file
    mapping_path = osp.join(osp.dirname(processed_graph_path), "label_to_field.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(label_to_field, f, indent=2, ensure_ascii=False)
    print(f"Saved label to field mapping to: {mapping_path}")
    
    return output_path, mapping_path

def generate_type_embeddings(graph_path, output_dir, device_id=0):
    """Generate embeddings for node and edge types"""
    device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")
    
    # Load sentence-bert model
    print("Loading sentence-bert model...")
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    model = model.to(device)
    
    # Load graph data
    print("Loading graph data...")
    graph_list, _ = dgl.load_graphs(graph_path)
    g = graph_list[0]
    
    # Get all node types and edge types
    node_types = g.ntypes
    edge_types = g.canonical_etypes
    
    def generate_node_descriptions(node_type):
        """Generate descriptions for each node type"""
        base_desc = [
            f"This node represents a {node_type}",
            f"This is a {node_type} node",
            f"A {node_type} node in the graph",
            f"This node is a {node_type}",
            f"The node type is {node_type}"
        ]
        return base_desc
    
    def generate_edge_descriptions(edge_type):
        """Generate descriptions for each edge type"""
        src, rel, dst = edge_type
        base_desc = [
            f"This {src} {rel} that {dst}",
            f"A {src} {rel} another {dst}",
            f"{src} to {dst} {rel}",
            f"This is a {rel} relationship",
            f"A {rel} between {src} and {dst}"
        ]
        return base_desc
    
    def get_embedding(descriptions, model, device):
        """Get embedding vector for descriptions"""
        embeddings = model.encode(descriptions, convert_to_tensor=True)
        embeddings = embeddings.to(device)
        avg_embedding = embeddings.mean(dim=0)
        return avg_embedding
    
    # Generate node type embeddings
    print("Generating node type embeddings...")
    node_type_embeddings = {}
    for node_type in tqdm(node_types):
        descriptions = generate_node_descriptions(node_type)
        embedding = get_embedding(descriptions, model, device)
        node_type_embeddings[node_type] = embedding.cpu()
    
    # Generate edge type embeddings
    print("Generating edge type embeddings...")
    edge_type_embeddings = {}
    for edge_type in tqdm(edge_types):
        descriptions = generate_edge_descriptions(edge_type)
        embedding = get_embedding(descriptions, model, device)
        edge_type_embeddings[edge_type] = embedding.cpu()
    
    # Save embeddings
    print("Saving embeddings...")
    save_dir = osp.join(output_dir, 'meta_dict/oag/')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(node_type_embeddings, f'{save_dir}/node_type.pt')
    torch.save(edge_type_embeddings, f'{save_dir}/edge_type.pt')
    
    print("Done!")
    return save_dir

def prepare_training_data(graph_path, output_dir, num_shot=0):
    """Prepare training data"""
    # Create output directories
    base_dir = Path(output_dir) / "stage2_data/OAG-small"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = base_dir / "instruct_ds_oag"
    for split in ['train', 'test', 'val']:
        (data_dir / 'ann').mkdir(parents=True, exist_ok=True)
        (data_dir / 'graph_data' / split).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    glist, _ = dgl.load_graphs(graph_path)
    g = glist[0]
    
    # Load node name mapping
    try:
        with open(osp.join(osp.dirname(graph_path), "graph_node_name.pkl"), 'rb') as f:
            node_names = pickle.load(f)
    except:
        node_names = {}
    
    # Load predefined dataset split indices
    print("Loading dataset split indices...")
    splits = {}
    for split in ['train', 'test', 'val']:
        with open(osp.join(osp.dirname(graph_path), f"{split}_ids.pkl"), 'rb') as f:
            splits[split] = pickle.load(f)
    
    # Select 5% of data for each set
    random.seed(42)
    small_splits = {}
    for split_name, paper_ids in splits.items():
        num_small = int(len(paper_ids) * 0.05)
        small_splits[split_name] = random.sample(paper_ids, num_small)
    
    def sample_subgraph(g, seed_nid, num_hops=2, fanout=10):
        """Sample neighbors for given seed nodes"""
        nodes = {ntype: [] for ntype in g.ntypes}
        nodes['paper'].append(seed_nid)
        
        for hop in range(num_hops):
            current_nodes = {ntype: torch.tensor(nodes[ntype], dtype=torch.int64) 
                           for ntype in nodes if len(nodes[ntype]) > 0}
            
            for etype in g.canonical_etypes:
                src_type, rel_type, dst_type = etype
                if dst_type in current_nodes:
                    frontier = dgl.sampling.sample_neighbors(
                        g,
                        {dst_type: current_nodes[dst_type]},
                        fanout,
                        edge_dir="in",
                        replace=False
                    )
                    src_nodes = frontier.edges(etype=etype)[0]
                    if src_type not in nodes:
                        nodes[src_type] = []
                    nodes[src_type].extend(src_nodes.tolist())
        
        for ntype in nodes:
            nodes[ntype] = list(set(nodes[ntype]))
        
        subg = dgl.node_subgraph(g, nodes)
        
        # Convert to PyG format
        pyg_data = HeteroData()
        for ntype in subg.ntypes:
            num_nodes = subg.num_nodes(ntype)
            pyg_data[ntype].x = torch.zeros((num_nodes, 768))
            pyg_data[ntype].dgl_to_pyg = torch.arange(num_nodes)
        
        for etype in subg.canonical_etypes:
            src_type, rel_type, dst_type = etype
            src, dst = subg.edges(etype=etype)
            edge_index = torch.stack([src, dst], dim=0)
            pyg_data[src_type, rel_type, dst_type].edge_index = edge_index
        
        return pyg_data
    
    def create_conversation(g, paper_id, node_names, label):
        """Create conversation content"""
        paper_title = node_names['paper'][paper_id] if 'paper' in node_names else f"Paper_{paper_id}"
        
        authors = []
        if ('paper', 'is writen by', 'author') in g.edge_types:
            edge_index = g['paper', 'is writen by', 'author'].edge_index
            paper_idx = 0
            author_mask = edge_index[0] == paper_idx
            author_ids = edge_index[1][author_mask].tolist()
            authors.extend([node_names['author'][aid] if 'author' in node_names else f"Author_{aid}" 
                          for aid in author_ids])
        
        prompt = (
            "Given a heterogeneous academic network graph about computer science, there are four types of nodes, "
            "namely: paper, author, affiliation, venue. The relationships (meta paths) between different nodes include: "
            "[('affiliation', 'employs', 'author'), ('author', 'is affiliated with', 'affiliation'), "
            "('author', 'writes', 'paper'), ('paper', 'cited by', 'paper'), ('paper', 'cites', 'paper'), "
            "('paper', 'is writen by', 'author'), ('paper', 'published in', 'venue'), ('venue', 'publishes', 'paper')]. "
            "By performing random sampling of 2-hop 10 neighbors centered on the target paper node, a heterogeneous "
            "subgraph is obtained. In the subgraph, \"paper\" nodes: <graph>, where the 0-th node is the central node "
            f"that represents a paper with the following information: \nTitle: {paper_title} \n"
            f"Published author lists: {authors} \n"
            "\"author\" nodes: <graph>; \"affiliation\" nodes: <graph>; \"venue\" nodes: <graph>. \n"
            "Question: Which of the following areas does this paper belong to: one of the 11 categories "
            "represented by integer labels ranging from 0 to 10? \n"
            "Give likely categories directly. "
        )
        
        return [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": str(label)}
        ]
    
    # Process each dataset split
    train_node_ids = []
    
    for split_name, paper_ids in small_splits.items():
        print(f"\nProcessing {split_name} set...")
        json_data = []
        
        for idx, paper_id in enumerate(tqdm(paper_ids, desc=f"Processing {split_name}")):
            # Sample subgraph
            pyg_graph = sample_subgraph(g, paper_id)
            
            # Save PyG format graph
            graph_file = f"OAG_{split_name}_{paper_id}.pt"
            graph_path = data_dir / 'graph_data' / split_name / graph_file
            torch.save(pyg_graph, graph_path)
            
            # Create JSON entry
            label = g.nodes['paper'].data['label'][paper_id].item()
            
            entry = {
                "id": f"OAG_{split_name}_{paper_id}",
                "graph_id": f"OAG_{split_name}_{paper_id}",
                "graph": {
                    "node_idx": paper_id,
                    "graph_idx": idx,
                    "graph": f"instruct_ds_oag/graph_data/{split_name}/{graph_file}",
                    "keys_order": ["paper", "author", "affiliation", "venue"],
                    "label": label
                },
                "conversations": create_conversation(pyg_graph, paper_id, node_names, label)
            }
            json_data.append(entry)
            
            if split_name == 'train':
                train_node_ids.append(paper_id)
        
        # Save JSON file
        json_file = data_dir / 'ann' / f'OAG_{split_name}_std_0_{len(paper_ids)}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Completed processing {split_name} set, total {len(paper_ids)} samples")
    
    # Generate and save masks
    print("\nGenerating mask files...")
    num_total_papers = g.num_nodes('paper')
    
    def generate_masks(train_node_ids, num_total_papers, seed=42):
        """Generate masks for training, testing, validation and few-shot"""
        random.seed(seed)
        
        train_mask = np.zeros(num_total_papers, dtype=bool)
        test_mask = np.zeros(num_total_papers, dtype=bool)
        val_mask = np.zeros(num_total_papers, dtype=bool)
        
        train_mask[train_node_ids] = True
        
        few_shot_sizes = [1, 3, 5, 10, 20, 40, 60]
        few_shot_masks = {}
        
        train_indices = list(range(len(train_node_ids)))
        random.shuffle(train_indices)
        
        full_train_mask = np.zeros(num_total_papers, dtype=bool)
        full_train_mask[train_node_ids] = True
        few_shot_masks[len(train_node_ids)] = full_train_mask
        
        for size in few_shot_sizes:
            few_shot_mask = np.zeros(num_total_papers, dtype=bool)
            selected_indices = train_indices[:size]
            selected_nodes = [train_node_ids[i] for i in selected_indices]
            few_shot_mask[selected_nodes] = True
            few_shot_masks[size] = few_shot_mask
        
        return train_mask, test_mask, val_mask, few_shot_masks
    
    train_mask, test_mask, val_mask, few_shot_masks = generate_masks(train_node_ids, num_total_papers)
    
    # Save mask files
    mask_dir = base_dir / 'processed'
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(mask_dir / 'seed_42_shot_test_mask.npy', test_mask)
    np.save(mask_dir / 'seed_42_shot_val_mask.npy', val_mask)
    np.save(mask_dir / 'seed_42_400_shot_train_mask.npy', train_mask)
    
    for size, mask in few_shot_masks.items():
        np.save(mask_dir / f'seed_42_{size}_shot_train_mask.npy', mask)
    
    return str(data_dir)

class NodeClassificationDataset(Dataset):
    """Dataset for node classification"""
    def __init__(self, data_path, tokenizer, num_shot=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.num_shot = num_shot
        
        # Load data
        print("Loading data...")
        data_files = glob.glob(osp.join(data_path, "ann/*.json"))
        self.samples = []
        
        if num_shot > 0:
            # Load few-shot mask
            mask_path = osp.join(osp.dirname(data_path), "processed", f"seed_42_{num_shot}_shot_train_mask.npy")
            few_shot_mask = np.load(mask_path)
            few_shot_ids = few_shot_mask.nonzero()[0]
        
        for data_file in data_files:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if num_shot > 0:
                    # Only use few-shot samples
                    data = [item for item in data if item['graph']['node_idx'] in few_shot_ids]
                self.samples.extend(data)
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load graph data
        graph_path = osp.join(self.data_path, sample['graph']['graph'])
        graph = torch.load(graph_path)
        
        # Process conversations
        conversations = sample['conversations']
        input_ids = self.tokenizer(
            conversations[0]['value'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).input_ids[0]
        
        labels = self.tokenizer(
            conversations[1]['value'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).input_ids[0]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'graph_data': graph,
            'hetero_key_order': sample['graph']['keys_order']
        }

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    

    print("\n=== Step 1: Assigning paper labels ===")
    labeled_graph_path, label_mapping_path = assign_paper_labels(
        original_graph_path=osp.join(data_args.graph_root, "graph_CS.pk"),
        processed_graph_path=osp.join(data_args.graph_root, "reduced_hg.bin")
    )
    

    print("\n=== Step 2: Generating type embeddings ===")
    type_embeddings_dir = generate_type_embeddings(
        graph_path=labeled_graph_path,
        output_dir=data_args.output_dir,
        device_id=training_args.local_rank
    )
    

    print("\n=== Step 3: Preparing training data ===")
    processed_data_dir = prepare_training_data(
        graph_path=labeled_graph_path,
        output_dir=data_args.output_dir,
        num_shot=data_args.num_shot
    )
    

    print("\n=== Step 4: Loading model and tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        padding_side="right"
    )
    
    model = HiGPTForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32
    )
    

    model.get_model().initialize_graph_modules(
        graph_tower=model_args.graph_tower,
        graph_select_layer=model_args.graph_select_layer,
        pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter
    )

    model.initialize_graph_tokenizer(
        use_graph_start_end=model_args.use_graph_start_end,
        tokenizer=tokenizer,
        device=training_args.device,
        tune_graph_mlp_adapter=False
    )
    

    print("\n=== Step 5: Preparing datasets ===")
    train_dataset = NodeClassificationDataset(
        data_path=processed_data_dir,
        tokenizer=tokenizer,
        num_shot=data_args.num_shot
    )

    print("\n=== Step 6: Starting training ===")
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    
    trainer.train()
    

    trainer.save_model(training_args.output_dir)
    print(f"\nTraining completed. Model saved to {training_args.output_dir}")


    print("\n=== Step 7: Generating node embeddings ===")
    model.eval()
    with torch.no_grad():

        glist, _ = dgl.load_graphs(osp.join(data_args.graph_root, "reduced_hg.bin"))
        g = glist[0]

        paper_embeddings = model.get_model().get_graph_tower()(g)['paper']

        output_file = osp.join(data_args.output_dir, "nc_infer.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(paper_embeddings.cpu().numpy(), f)
        print(f"Node embeddings saved to {output_file}")

if __name__ == "__main__":
    main() 