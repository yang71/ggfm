import os
import dgl
import torch
import random
import argparse

import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from ggfm.models import Classifier
from transformers import AutoTokenizer, AutoModel
from dgl.data.utils import save_graphs, load_graphs
from transformers.optimization import get_cosine_schedule_with_warmup
from ggfm.data import args_print, open_pkl_file, open_txt_file, save_pkl_file, save_txt_file
from ggfm.data import renamed_load, construct_graph, construct_graph_node_name, metapath_based_corpus_construction

import wandb
wandb.init(mode="disabled")


class LM_Model(nn.Module):
    def __init__(self, card):
        super().__init__()

        self.LM = AutoModel.from_pretrained(card)

    def forward(self, tokenized_tensors):

        out = self.LM(output_hidden_states=True, **tokenized_tensors)['hidden_states']
        embedding = out[-1].mean(dim=1)

        return embedding
    

class LM_dataset(Dataset):
    def __init__(self, user_text: list, labels: torch.Tensor):
        super().__init__()
        self.user_text = user_text
        self.labels = labels
        
    def __getitem__(self, index):
        
        text = self.user_text[index]
        label = self.labels[index]
        return text, label

    def __len__(self):
        return len(self.user_text)


def build_LM_dataloader(batch_size, idx, user_seq, labels, mode):

    if mode == 'train':
        user_text = []
        for i in idx:
            user_text.append(user_seq[i.item()])
        loader = DataLoader(dataset=LM_dataset(user_text, labels[idx]), batch_size=batch_size, shuffle=True)

    elif mode == 'eval':
        user_text = []
        for i in idx:
            user_text.append(user_seq[i.item()])
        loader = DataLoader(dataset=LM_dataset(user_text, labels[idx]), batch_size=batch_size*5)
    
    else:
        raise ValueError('mode should be in ["train", "eval"].')

    return loader


def batch_to_tensor(args, tokenizer, batch, device):
                    
    tokenized_tensors = tokenizer(text=batch[0], return_tensors='pt', max_length=args.max_length, truncation=True, padding='longest', add_special_tokens=False)
    for key in tokenized_tensors.keys():
        tokenized_tensors[key] = tokenized_tensors[key].to(device)
    labels = batch[1].to(device)
    
    return tokenized_tensors, labels


def evaluation(args, device, tokenizer, model, batch_size, idx, user_seq, hard_labels):
            
    eval_loader =  build_LM_dataloader(batch_size, idx, user_seq, hard_labels, mode='eval')
    model.eval()

    with torch.no_grad():
        final_preds, final_labels = [], []
        losses = []
        for batch in tqdm(eval_loader):
            predictions, ground_truth = [], []

            tokenized_tensors, labels = batch_to_tensor(args, tokenizer, batch, device)
            output = model(tokenized_tensors)
            loss = criterion(output, labels)

            losses += [loss.cpu().detach().tolist()]

            output = output.detach().cpu()
            labels = labels.detach().cpu()

            cates = torch.count_nonzero(labels, dim=1).tolist()

            for i in range(output.shape[0]):
                k = cates[i]
                _, pred_indxs = torch.topk(output[i], k)
                predictions.append(pred_indxs.tolist())
                _, label_indxs = torch.topk(labels[i], k)
                ground_truth.append(label_indxs.tolist())
            
            for i in range(len(predictions)):
                flag = 0
                predictions[i] = sorted(predictions[i])
                ground_truth[i] = sorted(ground_truth[i])
                for j in range(len(predictions[i])):
                    if predictions[i][j] != ground_truth[i][j]:
                        flag = 1
                        break
                if flag == 1: final_preds.append(0)
                else: final_preds.append(1)
                final_labels.append(1)
        
        valid_mi_f1 = f1_score(final_labels, final_preds, average='micro')
        valid_ma_f1 = f1_score(final_labels, final_preds, average='macro')

        return losses, valid_mi_f1, valid_ma_f1


def cal_cl_loss(gnn_outputs, lm_outputs, labels):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * gnn_outputs @ lm_outputs.t()
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    ret_loss = (loss_i + loss_t) / 2
    return ret_loss


def seed_setting(seed_number):
    random.seed(seed_number)
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_nc_labels(graph):

    field_nc_labels = list(graph.node_feature['field']['label'])
    nc_output_num = max(field_nc_labels) + 1

    # prepare paper data field -> paper, only 529808 nodes has labels
    data_pairs  = {}

    for target_id in graph.edge_list['paper']['field']['rev_PF_in_L2']:  # paper_id
        for source_id in graph.edge_list['paper']['field']['rev_PF_in_L2'][target_id]:  # field_id
            if target_id not in data_pairs:
                data_pairs[target_id] = []
            data_pairs[target_id].append(source_id)  # paper: [field, field, field, ...]

    idxs = list(data_pairs.keys())
    num_paper = len(graph.node_feature['paper']['title'].tolist())
    
    nc_labels = np.zeros([num_paper, nc_output_num])  # all_papers 546704
    for i in range(len(idxs)):
        for source_id in data_pairs[idxs[i]]:
            nc_labels[idxs[i]][field_nc_labels[source_id]] = 1
    
    row_sums = nc_labels.sum(axis=1).reshape(-1, 1)
    nc_labels = np.where(row_sums != 0, nc_labels / row_sums, 0)
    return nc_labels


def nc_load_raw_data(data_dir, graph):
    
    # # 基于元路径生成语料库
    # src_dst2edge_type = {('paper', 'venue'): 'published in', 
    #                      ('venue', 'paper'): 'publishes', 
    #                      ('field', 'paper'): 'includes', 
    #                      ('author', 'paper'): 'writes', 
    #                      ('paper', 'field'): 'belongs to', 
    #                      ('author', 'affiliation'): 'is affiliated with', 
    #                      ('affiliation', 'author'): 'employs', 
    #                      ('paper', 'author'): 'is writen by'
    #                     }
    # construct_graph(data_dir, graph, src_dst2edge_type)
    # construct_graph_node_name(data_dir, graph)

    # # Prevent data leakage
    # glist, label_dict = load_graphs(data_dir + 'graph.bin')
    # g = glist[0]
    # edge_types = [('paper', 'published in', 'venue'), ('venue', 'publishes', 'paper'), 
    #               ('paper', 'cites', 'paper'), ('paper', 'cited by', 'paper'),
    #               ('author', 'writes', 'paper'), ('author', 'is affiliated with', 'affiliation'), 
    #               ('affiliation', 'employs', 'author'), ('paper', 'is writen by', 'author')]
    # edge_types_dict = {}
    # for et in edge_types:
    #     edge_types_dict[et[1]] = et

    # # delete field, renew graph
    # reduced_hg = {}
    # origin_edge_types = g.etypes
    # for et in origin_edge_types:
    #     if et not in ['includes', 'in', 'contains', 'belongs to']:
    #         reduced_hg[edge_types_dict[et]] = g.edges(etype=et)
    
    # new_g = dgl.heterograph(reduced_hg)
    # save_graphs(data_dir + "graph.bin", new_g)

    # metapaths = {'paper': [['cites', 'cited by'], ['is writen by', 'writes'], ['published in', 'publishes'], ['is writen by', 'is affiliated with', 'employs', 'writes']],  # ppp, pap, pvp, p-a(author)-a(affiliation)-a(author)-p
    #             }
    # relation = {'paper': [['cites', 'cited by'], ['is writen by', 'writes'], ['published in', 'publishes'], ['is writen by', 'is affiliated with', 'employs', 'writes']],
    #            }
    # mid_types = {'paper': [['paper'], ['author'], ['venue'], ['author', 'affiliation', 'author']],
    #             }

    # train_idxs = open_pkl_file(args.data_dir+"train_ids.pkl")
    # valid_idxs = open_pkl_file(args.data_dir+"valid_ids.pkl")
    # test_idxs = open_pkl_file(args.data_dir+"test_ids.pkl")
    
    # train_corpus = metapath_based_corpus_construction(data_dir, 'paper', metapaths, relation, mid_types, train_idxs)
    # valid_corpus = metapath_based_corpus_construction(data_dir, 'paper', metapaths, relation, mid_types, valid_idxs)
    # test_corpus = metapath_based_corpus_construction(data_dir, 'paper', metapaths, relation, mid_types, test_idxs)

    # # random pick 40000 for train, valid, test
    # length = min(40000, len(train_corpus))

    # train_length = int(length * 0.8)
    # valid_length = int(length * 0.1)

    # # sampling index for training and validation
    # train_idx = random.sample([i for i in range(len(train_corpus))], train_length)
    # valid_idx = random.sample([i for i in range(len(valid_corpus))], valid_length)
    # test_idx = [i for i in range(len(test_corpus))]


    # new_data = []
    # train_corpus = np.array(train_corpus)
    # valid_corpus = np.array(valid_corpus)
    # test_corpus = np.array(test_corpus)
    # train_idx, valid_idx, test_idx = np.array(train_idx), np.array(valid_idx), np.array(test_idx)
    # new_data.extend(train_corpus[train_idx].tolist())
    # new_data.extend(valid_corpus[valid_idx].tolist())
    # new_data.extend(test_corpus[test_idx].tolist())

    # new_label_idxs = []
    # train_idxs = np.array(train_idxs)
    # valid_idxs = np.array(valid_idxs)
    # test_idxs = np.array(test_idxs)
    # new_label_idxs.extend(train_idxs[train_idx].tolist())
    # new_label_idxs.extend(valid_idxs[valid_idx].tolist())
    # new_label_idxs.extend(test_idxs[test_idx].tolist())

    # save_pkl_file(data_dir+"lmch_nc_train_idxs.pkl", torch.tensor([i for i in range(len(train_idx))]))
    # save_pkl_file(data_dir+"lmch_nc_valid_idxs.pkl", torch.tensor([i for i in range(len(train_idx),len(train_idx)+len(valid_idx))]))
    # save_pkl_file(data_dir+"lmch_nc_test_idxs.pkl", torch.tensor([i for i in range(len(train_idx)+len(valid_idx), len(train_idx)+len(valid_idx)+len(test_idx))]))

    # save_txt_file(data_dir+"lmch_nc_sampled_pt_corpus.txt", new_data)
    # save_pkl_file(data_dir+"lmch_nc_sampled_pt_labeled_idxs.pkl", new_label_idxs)

    # print("length: ", len(new_data))
    # print("length: ", len(new_label_idxs))
    

    # from here...
    print('Loading data...')
    train_idx = open_pkl_file(data_dir+'lmch_nc_train_idxs.pkl')
    valid_idx = open_pkl_file(data_dir+'lmch_nc_valid_idxs.pkl')
    test_idx = open_pkl_file(data_dir+'lmch_nc_test_idxs.pkl')

    user_text = open_txt_file(data_dir + "lmch_nc_sampled_pt_corpus.txt")
    nc_labels = get_nc_labels(graph)

    pt_labeled_idxs = open_pkl_file(data_dir + "lmch_nc_sampled_pt_labeled_idxs.pkl")
    
    nc_labels = nc_labels[pt_labeled_idxs, :]
    nc_labels = torch.from_numpy(nc_labels)   # labels tensor & one-hot
    print("nc_labels.shape: ", nc_labels.shape)

    return {'train_idx': train_idx, 
            'valid_idx': valid_idx, 
            'test_idx': test_idx, 
            'user_text': user_text, 
            'labels': nc_labels}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fine-Tuning on OAG Paper-Field (L2) classification task')

    parser.add_argument('--data_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/datasets/', help='The address of preprocessed graph.')
    parser.add_argument('--pretrain_model_dir', type=str, default='/home/yjy/heteroPrompt/distilroberta-base', help='The address for pretrained model.')
    parser.add_argument('--model_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/fine_tuned_model', help='The address for storing the models and optimization results.')
    parser.add_argument('--task_name', type=str, default='lmch_nc', help='The name of the stored models and optimization results.')
    parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--label_smoothing_factor', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.6)
    parser.add_argument('--pretrain_epochs', type=float, default=3)
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--n_epoch', type=int, default=5, help='Number of epoch to run')
    parser.add_argument('--clip', type=float, default=0.5, help='Gradient Norm Clipping')
    parser.add_argument('--eval_patience', type=int, default=20, help='Gradient Norm Clipping')


    args = parser.parse_args()
    args_print(args)

    if args.cuda != -1: device = torch.device("cuda:" + str(args.cuda))
    else: device = torch.device("cpu")

    seed_setting(args.seed)

    # data for pretrain
    graph = renamed_load(open(args.data_dir + "graph.pk", 'rb'))
    field_labels = list(graph.node_feature['field']['label'])
    output_num = max(field_labels) + 1  # 11 paper node classification

    data = nc_load_raw_data(args.data_dir, graph)

    train_idx, valid_idx, test_idx, hard_labels, user_seq = data['train_idx'], data['valid_idx'], data['test_idx'], data['labels'], data['user_text'],

    # build model
    card = args.pretrain_model_dir
    lm_tokenizer = AutoTokenizer.from_pretrained(card)
    lm_encoder = LM_Model(card)
    classifier = Classifier(768, output_num)
    model = nn.Sequential(lm_encoder, classifier).to(device)

    # train
    print('LM fine-tuning start!')
    optimizer_args = dict(lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    criterion = CrossEntropyLoss(label_smoothing=args.label_smoothing_factor)

    pretrain_steps_per_epoch = train_idx.shape[0] // args.batch_size + 1
    pretrain_steps = int(pretrain_steps_per_epoch * args.pretrain_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, pretrain_steps_per_epoch * args.warmup, pretrain_steps)

    step = 0
    valid_acc_best = 0
    valid_step_best = 0
    
    torch.save(model, os.path.join(args.model_dir, args.task_name))
    
    train_loader = build_LM_dataloader(args.batch_size, train_idx, user_seq, hard_labels, 'train')

    for epoch in range(int(args.pretrain_epochs)+1):
        model.train()
        print(f'------LM Training Epoch: {epoch}/{int(args.pretrain_epochs)}------')
        train_losses = []
        for batch in tqdm(train_loader):
            step += 1
            if step >= pretrain_steps: break
            tokenized_tensors, labels = batch_to_tensor(args, lm_tokenizer, batch, device)

            output = model(tokenized_tensors)

            loss = criterion(output, labels)
            loss /= args.grad_accumulation
            loss.backward()

            train_losses += [loss.cpu().detach().tolist()]
            
            if step % args.grad_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            if step % args.eval_patience == 0:
                valid_losses, valid_mi_f1, valid_ma_f1 = evaluation(args, device, lm_tokenizer, model, args.batch_size, valid_idx, user_seq, hard_labels)

                print(("Epoch: %d  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid ACC: %.4f") % \
                  (epoch, optimizer.param_groups[0]['lr'], np.average(train_losses), np.average(valid_losses), valid_mi_f1))

                if valid_mi_f1 > valid_acc_best:
                    valid_acc_best = valid_mi_f1
                    valid_step_best = step
                    
                    torch.save(model, os.path.join(args.model_dir, args.task_name))
    
    # test
    best_model = torch.load(os.path.join(args.model_dir, args.task_name))
    test_losses, test_mi_f1, test_ma_f1 = evaluation(args, device, lm_tokenizer, best_model, args.batch_size, test_idx, user_seq, hard_labels)
    print(f"Best Test ACC: {test_mi_f1}")