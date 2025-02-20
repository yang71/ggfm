import os

# os.environ["CUDA_VISIBLE_DEVICES"]='0'

import torch
import argparse
import numpy as np
import torch.nn as nn
from ggfm.data import args_print
from collections import OrderedDict
from sklearn.metrics import f1_score
from ggfm.data import renamed_load, sample_subgraph, open_pkl_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from warnings import filterwarnings
filterwarnings("ignore")

from ggfm.models import LlagaLlamaForCausalLM
from ggfm.models import LLAGA
import transformers
from transformers import Trainer
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence
import random
from torch.utils.data import Sampler
from transformers.trainer import has_length
from dataclasses import dataclass, field
from transformers import TrainingArguments
import copy
from enum import auto, Enum
from typing import List, Tuple
import dataclasses
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import pickle

IGNORE_INDEX = -100
GRAPH_TOKEN_INDEX = -200
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_START_TOKEN = "<GH>"
DEFAULT_GRAPH_END_TOKEN = "</GH>"
DEFAULT_GRAPH_PAD_ID = -500

def _save_checkpoint(model, optimizer, cur_epoch, args, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """
    os.makedirs(f'{args.output_dir}', exist_ok=True)

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": args,
        "epoch": cur_epoch,
    }
    path = f'{args.output_dir}/test.pth'
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, path))
    torch.save(save_obj, path)


def _reload_best_model(model, args):
    """
    Load the best checkpoint for evaluation.
    """
    checkpoint_path = f'{args.output_dir}/test.pth'

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model

def collate_fn(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    # if 'graph' in batch:
    #     batch['graph'] = Batch.from_data_list(batch['graph'])
    return batch
class LLAGADataset(Dataset):
    def __init__(self,
                 data_args, graph, idx, classes, ylabel):
        super(LLAGADataset, self).__init__()
        self.data = graph
        self.use_hop = data_args.use_hop
        self.template = data_args.template
        self.datas = {}
        self.idx = idx
        # self.index={}

        neighbors = {}
        graph_emb = {}

        pretrained_emb = self.data.node_feature['paper'].emb

        for target_id in idx:
            neighbors[target_id] = [target_id]
            tmp_graph_emb = []
            if graph.edge_list['paper']['paper']['rev_PP_cite'].get(target_id):
                neighbors[target_id]+=list(graph.edge_list['paper']['paper']['rev_PP_cite'].get(target_id).keys())
            for id in neighbors[target_id]:
                tmp_graph_emb.append(pretrained_emb[id])
            graph_emb[target_id] = torch.tensor(tmp_graph_emb)

        self.graph_emb = graph_emb
        self.task = data_args.task

        task_list_data_dict = []
        if self.task == "nc":
            for i, id in enumerate(self.idx):
                l = {}
                l['id'] = id
                l['graph'] = neighbors.get(id)
                label = list(np.where(ylabel[i] > 0)[0])
                l["question"] = f"Given a node-centered graph: <qraph>, each node represents a paper, we need to classify the center node into {classes} classes. please tell me which class thecenter node belongs to?"
                l["label"] = f"{label}"
                l['graph_emb'] = self.graph_emb.get(id)
                task_list_data_dict.append(l)
        else:
            print(f"{self.task} not exist!!!")
            raise ValueError

        random.shuffle(task_list_data_dict)
        self.task_list_data_dict = task_list_data_dict

    def __len__(self):
        return len(self.task_list_data_dict)

    def __getitem__(self, index):
        pair = self.task_list_data_dict[index]

        graph_id = pair["id"]
        label = pair["label"]
        graph = pair["graph"]
        question = pair["question"]
        graph_emb = pair['graph_emb']

        return {
            'id': index,
            'graph_id': graph_id,
            'label': label,
            'graph': graph,
            'question': question,
            'graph_emb': graph_emb
        }

def generate_nc_labels(graph, idxs, pairs, output_num):

    cand_list = list(graph.edge_list['field']['paper']['PF_in_L2'].keys())  # field_id
    field_labels = list(graph.node_feature['field']['label'])

    ylabel = np.zeros([len(idxs), output_num])
    for x_id, target_id in enumerate(idxs):
        if target_id not in pairs:
            print('error 1' + str(target_id))
        for source_id in pairs[target_id][0]:
            if source_id not in cand_list:
                print('error 2' + str(target_id))
            ylabel[x_id][field_labels[source_id]] = 1

    ylabel /= ylabel.sum(axis=1).reshape(-1, 1)
    return ylabel

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fine-Tuning on OAG Paper-Field (L2) classification task')
    '''
            Dataset arguments
        '''
    parser.add_argument('--data_dir', type=str, default='/home/cwj/OAG_CS/',
                        help='The address of data.')
    parser.add_argument('--llm_model_path', type=str,
                        default='/home/cwj/RAG/Meta-Llama-3-8B',
                        # default='/home/cwj/RAG/Llama-2-7b-hf',
                        help='The address for pretrained model.')
    '''
        Optimization arguments
    '''
    parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer to use.')
    parser.add_argument('--scheduler', type=str, default='cycle', help='Name of learning rate scheduler.',
                        choices=['cycle', 'cosine'])

    parser.add_argument('--n_pool', type=int, default=8, help='Number of process to sample subgraph')
    parser.add_argument('--n_batch', type=int, default=16, help='Number of batch (sampled graphs) for each epoch')
    parser.add_argument('--target_type', type=str, default='paper', help='target type for training')

    parser.add_argument('--data_percentage', type=int, default=1,
                        help='Percentage of training and validation data to use')
    parser.add_argument('--clip', type=int, default=0.5, help='Gradient Norm Clipping')
    parser.add_argument('--mm_hidden_size', type=int, default=768, help='hiden layer size')
    parser.add_argument('--tune_mm_mlp_adapter', type=str, default=True, help='tune projector layer')
    parser.add_argument('--use_hop', type=str, default=False)
    parser.add_argument('--template', type=str, default='ND')
    parser.add_argument('--task', type=str, default='nc', help='task')
    parser.add_argument('--llm_frozen', type=str, default=True, help='Frozen llm')
    parser.add_argument("--max_txt_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--lr", type=int, default=2e-3)
    parser.add_argument("--wd", type=int, default=0.05)
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epoch to run')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of output nodes for training')
    parser.add_argument('--output_dir', type=str, default='/home/cwj/tmp',help='saving path')
    parser.add_argument('--result_save_dir', type=str, default='/home/cwj/tmp/', help='saving path')


    args = parser.parse_args()
    args_print(args)

    if args.cuda != -1:
        device = torch.device("cuda:" + str(args.cuda))
    else:
        device = torch.device("cpu")

    # pre data for fine-tuning
    graph = renamed_load(open(args.data_dir + "oagcs_graph.pk", 'rb'))

    train_ids = open_pkl_file(args.data_dir + "train_ids.pkl")
    valid_ids = open_pkl_file(args.data_dir + "valid_ids.pkl")
    test_ids = open_pkl_file(args.data_dir + "test_ids.pkl")

    # Only train and valid with a certain percentage of data, if necessary.
    np.random.seed(43)
    train_ids = np.random.choice(train_ids, int(len(train_ids) * args.data_percentage), replace=False)
    valid_ids = np.random.choice(valid_ids, int(len(valid_ids) * args.data_percentage), replace=False)
    test_ids = np.random.choice(test_ids, int(len(test_ids)), replace=False)

    # get pairs for labels
    train_pairs = {}
    valid_pairs = {}
    test_pairs = {}

    # Prepare all the souce nodes (L2 field) associated with each target node (paper) as dict
    for target_id in graph.edge_list['paper']['field']['rev_PF_in_L2']:  # paper_id
        for source_id in graph.edge_list['paper']['field']['rev_PF_in_L2'][target_id]:  # field_id
            _time = graph.edge_list['paper']['field']['rev_PF_in_L2'][target_id][source_id]  # time
            if target_id in train_ids:
                if target_id not in train_pairs:
                    train_pairs[target_id] = [[], _time]
                train_pairs[target_id][0] += [source_id]

            elif target_id in valid_ids:
                if target_id not in valid_pairs:
                    valid_pairs[target_id] = [[], _time]
                valid_pairs[target_id][0] += [source_id]
            else:
                if target_id not in test_pairs:
                    test_pairs[target_id] = [[], _time]
                test_pairs[target_id][0] += [source_id]

    field_labels = list(graph.node_feature['field']['label'])
    output_num = max(field_labels) + 1  # 11 paper node classification

    ylabel_train = generate_nc_labels(graph, train_ids, train_pairs, output_num)
    ylabel_valid = generate_nc_labels(graph, valid_ids, valid_pairs, output_num)
    ylabel_test = generate_nc_labels(graph, test_ids, test_pairs, output_num)

    graph_node_name = {}
    graph_node_type = graph.get_types()
    for i in range(len(graph_node_type)):
        attr = "name"
        if graph_node_type[i] == "paper": attr = "title"
        graph_node_name[graph_node_type[i]] = graph.node_feature[graph_node_type[i]][attr].tolist()
    all_name = graph_node_name[args.target_type]

    train_dataset = LLAGADataset(args,graph=graph,idx=train_ids,classes=output_num,ylabel=ylabel_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True,collate_fn=collate_fn)
    valid_dataset = LLAGADataset(args, graph=graph, idx=valid_ids, classes=output_num, ylabel=ylabel_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn)
    test_dataset = LLAGADataset(args, graph=graph, idx=test_ids, classes=output_num, ylabel=ylabel_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn)

    model = LLAGA(args)

    # Step 4 Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):

            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            progress_bar.update(1)

        print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")

        val_loss = 0.
        eval_output = []
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                loss = model(batch)
                val_loss += loss.item()
            val_loss = val_loss / len(valid_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

    model = _reload_best_model(model, args)
    model.eval()
    eval_output = []
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model.inference(batch)
            eval_output.append(output)

        progress_bar_test.update(1)

    predictions, ground_truth = [], []
    final_preds, final_labels = [], []

    for i in range(len(eval_output)):
        for pred in eval_output[i]['pred']:
            tmp_pred = []
            for j in range(output_num):
                if str(j) in pred:
                    tmp_pred.append(j)
            predictions.append(tmp_pred)
        for label in eval_output[i]['label']:
            tmp_label = []
            for j in range(output_num):
                if str(j) in label:
                    tmp_label.append(j)
            ground_truth.append(tmp_label)

    pred_save_path = args.result_save_dir + 'pred.pkl'
    label_save_path = args.result_save_dir + 'label.pkl'
    with open(pred_save_path, 'wb') as f:
        pickle.dump(predictions, f)
    with open(label_save_path, 'wb') as f:
        pickle.dump(predictions, f)

    for i in range(len(predictions)):
        flag = 0
        predictions[i] = sorted(predictions[i])
        ground_truth[i] = sorted(ground_truth[i])
        if len(predictions[i]) != len(ground_truth[i]):
            flag = 1
        else:
            for j in range(len(predictions[i])):
                if predictions[i][j] != ground_truth[i][j]:
                    flag = 1
                    break
        if flag == 1:
            final_preds.append(0)
        else:
            final_preds.append(1)
        final_labels.append(1)

    ma = f1_score(final_labels, final_preds, average='macro')
    mi = f1_score(final_labels, final_preds, average='micro')

    print(f"Best Test ma: {ma}")
    print(f"Best Test mi: {mi}")
