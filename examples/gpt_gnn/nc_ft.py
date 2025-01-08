import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from ggfm.data import args_print
from collections import OrderedDict
from sklearn.metrics import f1_score
from ggfm.models import HGT, Classifier
from ggfm.data import renamed_load, sample_subgraph, open_pkl_file

from warnings import filterwarnings
filterwarnings("ignore")


def to_torch(feature, time, edge_list, graph):
    '''
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    '''
    node_dict = {}
    node_feature = []
    node_type    = []
    node_time    = []
    edge_index   = []
    edge_type    = []
    edge_time    = []
    
    node_num = 0
    types = graph.get_types()
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num += len(feature[t])
        
    for t in types:
        node_feature += list(feature[t])
        node_time    += list(time[t])
        node_type    += [node_dict[t][1] for _ in range(len(feature[t]))]
        
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict['self'] = len(edge_dict)

    # transfer to homogeneous graph
    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                    tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                    edge_index += [[sid, tid]]
                    edge_type  += [edge_dict[relation_type]]

                    # time ranges from 1900 - 2020, largest span is 120.
                    edge_time  += [node_time[tid] - node_time[sid] + 120]

    node_feature = torch.FloatTensor(node_feature)
    node_type    = torch.LongTensor(node_type)
    edge_time    = torch.LongTensor(edge_time)
    edge_index   = torch.LongTensor(edge_index).t()
    edge_type    = torch.LongTensor(edge_type)
    return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict


def node_sample(seed, graph, args, pairs, time_range, output_num):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers), get their time.
    '''
    np.random.seed(seed)
    target_ids = np.random.choice(list(pairs.keys()), args.batch_size, replace = False)
    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [[target_id, _time]]
    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    feature, times, edge_list, _, _ = sample_subgraph(graph, time_range, inp = {'paper': np.array(target_info)}, sampled_depth = args.sample_depth, sampled_number = args.sample_width)

    '''
        (3) Mask out the edge between the output target nodes (paper) with output source nodes (L2 field)
    '''
    masked_edge_list = []
    for i in edge_list['paper']['field']['rev_PF_in_L2']:
        if i[0] >= args.batch_size:  # only save the non mask-idxs
            masked_edge_list += [i]
    edge_list['paper']['field']['rev_PF_in_L2'] = masked_edge_list

    masked_edge_list = []
    for i in edge_list['field']['paper']['PF_in_L2']:
        if i[1] >= args.batch_size:
            masked_edge_list += [i]
    edge_list['field']['paper']['PF_in_L2'] = masked_edge_list
    '''
        (4) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_torch(feature, times, edge_list, graph)
    '''
        (5) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    cand_list = list(graph.edge_list['field']['paper']['PF_in_L2'].keys())  # field_id

    ylabel = np.zeros([args.batch_size, output_num])
    for x_id, target_id in enumerate(target_ids):
        if target_id not in pairs:
            print('error 1' + str(target_id))
        for source_id in pairs[target_id][0]:
            if source_id not in cand_list:
                print('error 2' + str(target_id))
            ylabel[x_id][field_labels[source_id]] = 1

    ylabel /= ylabel.sum(axis=1).reshape(-1, 1)
    x_ids = np.arange(args.batch_size) + node_dict['paper'][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel


def load_gnn(_dict):
    out_dict = {}
    for key in _dict:
        if 'gnn' in key:
            out_dict[key[4:]] = _dict[key]
    return OrderedDict(out_dict)


def randint():
    return np.random.randint(2**32 - 1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fine-Tuning on OAG Paper-Field (L2) classification task')

    '''
        Dataset arguments
    '''
    parser.add_argument('--data_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/datasets/', help='The address of data.')
    parser.add_argument('--use_pretrain', type=str, default=True, help='Whether to use pre-trained model')
    parser.add_argument('--pretrain_model_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/pretrained_model/gta_all_cs3', help='The address for pretrained model.')
    parser.add_argument('--model_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/fine_tuned_model', help='The address for storing the models and optimization results.')
    parser.add_argument('--task_name', type=str, default='gptgnn_nc', help='The name of the stored models and optimization results.')
    parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
    parser.add_argument('--sample_depth', type=int, default=6, help='How many numbers to sample the graph')
    parser.add_argument('--sample_width', type=int, default=128, help='How many nodes to be sampled per layer per type')

    '''
    Model arguments 
    '''
    parser.add_argument('--conv_name', type=str, default='hgt', help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
    parser.add_argument('--n_hid', type=int, default=400, help='Number of hidden dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention head')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
    parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers', action='store_true')
    parser.add_argument('--dropout', type=int, default=0.2, help='Dropout ratio')


    '''
        Optimization arguments
    '''
    parser.add_argument('--scheduler', type=str, default='cycle', help='Name of learning rate scheduler.' , choices=['cycle', 'cosine'])
    parser.add_argument('--data_percentage', type=int, default=0.1, help='Percentage of training and validation data to use')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epoch to run')
    parser.add_argument('--n_pool', type=int, default=8, help='Number of process to sample subgraph')    
    parser.add_argument('--n_batch', type=int, default=16, help='Number of batch (sampled graphs) for each epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of output nodes for training')
    parser.add_argument('--clip', type=int, default=0.5, help='Gradient Norm Clipping')

    args = parser.parse_args()
    args_print(args)

    if args.cuda != -1: device = torch.device("cuda:" + str(args.cuda))
    else: device = torch.device("cpu")


    # pre data for fine-tuning
    graph = renamed_load(open(args.data_dir+"graph.pk", 'rb'))

    target_type = 'paper'
    types = graph.get_types()

    pre_range   = {t: True for t in graph.times if t != None and t < 2014}
    train_range = {t: True for t in graph.times if t != None and t >= 2014 and t <= 2016}
    valid_range = {t: True for t in graph.times if t != None and t > 2016 and t <= 2017}
    test_range  = {t: True for t in graph.times if t != None and t > 2017}


    train_ids = open_pkl_file(args.data_dir+"train_ids.pkl")
    valid_ids = open_pkl_file(args.data_dir+"valid_ids.pkl")
    test_ids = open_pkl_file(args.data_dir+"test_ids.pkl")

    # get pairs for labels
    train_pairs = {}
    valid_pairs = {}
    test_pairs  = {}
    
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
                    test_pairs[target_id]  = [[], _time]
                test_pairs[target_id][0]  += [source_id]

    
    # Only train and valid with a certain percentage of data, if necessary.
    np.random.seed(43)
    sel_train_pairs = {p : train_pairs[p] for p in np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage), replace = False)}
    sel_valid_pairs = {p : valid_pairs[p] for p in np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage), replace = False)}


    field_labels = list(graph.node_feature['field']['label'])
    output_num = max(field_labels) + 1  # 11 paper node classification


    # Initialize GNN (model is specified by conv_name) and Classifier
    gnn = HGT(in_dim = len(graph.node_feature[target_type]['emb'].values[0]) + 401, n_hid = args.n_hid, \
            n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout, num_types = len(types), \
            num_relations = len(graph.get_meta_graph()) + 1, prev_norm = args.prev_norm, last_norm = args.last_norm)
    if args.use_pretrain:
        gnn.load_state_dict(load_gnn(torch.load(args.pretrain_model_dir)), strict = False)
        print('Load Pre-trained Model from (%s)' % args.pretrain_model_dir)
    classifier = Classifier(args.n_hid, output_num)
    model = nn.Sequential(gnn, classifier).to(device)

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer_args = dict(lr=5e-4)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)

    res = []
    best_val = 0
    train_step = 0
    
    for epoch in np.arange(args.n_epoch) + 1:
        train_data = []
        for _ in np.arange(args.n_batch):
            train_data.append(node_sample(randint(), graph, args, sel_train_pairs, train_range, output_num))
        valid_data = node_sample(randint(), graph, args, sel_valid_pairs, valid_range, output_num)
        
        
        model.train()
        train_losses = []
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res  = classifier.forward(node_rep[x_ids])
            loss = criterion(res, torch.FloatTensor(ylabel).to(device))

            optimizer.zero_grad() 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step(train_step)
            del res, loss
        
        model.eval()
        with torch.no_grad():
            final_preds, final_labels = [], []

            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res = classifier.forward(node_rep[x_ids])
            loss = criterion(res, torch.FloatTensor(ylabel).to(device))

            predictions, ground_truth = [], []

            output = res.detach().cpu()
            labels = torch.from_numpy(ylabel)

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

            ma = f1_score(final_labels, final_preds, average='macro')
            mi = f1_score(final_labels, final_preds, average='micro')

            if mi > best_val:
                best_val = mi
                torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))

            print(("Epoch: %d  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid ACC: %.4f") % \
                  (epoch, optimizer.param_groups[0]['lr'], np.average(train_losses), loss.cpu().detach().tolist(), mi))
            del res, loss
        del train_data, valid_data

    
    best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
    best_model.eval()
    gnn, classifier = best_model
    with torch.no_grad():
        final_preds, final_labels = [], []

        test_res = []
        all_res = []
        
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = node_sample(randint(), graph, args, test_pairs, test_range, output_num)
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
        res = classifier.forward(paper_rep)
        
        predictions, ground_truth = [], []

        output = res.detach().cpu()
        labels = torch.from_numpy(ylabel)

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

        ma = f1_score(final_labels, final_preds, average='macro')
        mi = f1_score(final_labels, final_preds, average='micro')

        print(f"Best Test ACC: {mi}")