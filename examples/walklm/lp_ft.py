import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from ggfm.models import Classifier
from transformers import AutoTokenizer, AutoModel
from ggfm.data import args_print, open_pkl_file, renamed_load, generate_lm_embs, ndcg_at_k, mean_reciprocal_rank


def generate_lp_labels(graph, idxs, pairs, output_num):

    cand_list = list(graph.edge_list['field']['paper']['PF_in_L2'].keys())  # field_id

    ylabel = np.zeros([len(idxs), output_num])
    for x_id, target_id in enumerate(idxs):
        if target_id not in pairs:
            print('error 1' + str(target_id))
        for source_id in pairs[target_id][0]:
            if source_id not in cand_list:
                print('error 2' + str(target_id))
            ylabel[x_id][cand_list.index(source_id)] = 1

    ylabel /= ylabel.sum(axis=1).reshape(-1, 1)
    return ylabel


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Tuning on OAG Paper-Field (L2) classification task')

    '''
        Dataset arguments
    '''
    parser.add_argument('--data_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/datasets/', help='The address of preprocessed graph.')
    parser.add_argument('--use_pretrain', type=str, default=True, help='Whether to use pre-trained model')
    parser.add_argument('--pretrain_model_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/pretrained_model/walklm/xyz', help='The address for pretrained model.')
    parser.add_argument('--model_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/fine_tuned_model', help='The address for storing the models and optimization results.')
    parser.add_argument('--task_name', type=str, default='walklm_lp', help='The name of the stored models and optimization results.')
    parser.add_argument('--cuda', type=int, default=2, help='Avaiable GPU ID')
    parser.add_argument('--clip', type=int, default=0.5, help='Gradient Norm Clipping')
    parser.add_argument('--n_epoch', type=int, default=5, help='Number of epoch to run')
    parser.add_argument('--target_type', type=str, default='paper', help='target type for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of output nodes for training')


    args = parser.parse_args()
    args_print(args)

    if args.cuda != -1: device = torch.device("cuda:" + str(args.cuda))
    else: device = torch.device("cpu")

    train_ids = open_pkl_file(args.data_dir+"train_ids.pkl")
    valid_ids = open_pkl_file(args.data_dir+"valid_ids.pkl")
    test_ids = open_pkl_file(args.data_dir+"test_ids.pkl")

    # load graph
    graph = renamed_load(open(args.data_dir + "graph.pk", 'rb'))

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

    
    cand_list = list(graph.edge_list['field']['paper']['PF_in_L2'].keys())
    output_num = len(cand_list)  # link prediction as node classification

    graph_node_name = {}
    graph_node_type = graph.get_types()
    for i in range(len(graph_node_type)):
        attr = "name"
        if graph_node_type[i] == "paper": attr = "title"
        graph_node_name[graph_node_type[i]] = graph.node_feature[graph_node_type[i]][attr].tolist()
    all_name = graph_node_name[args.target_type]

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_dir)
    lm_encoder = AutoModel.from_pretrained(args.pretrain_model_dir).to(device)
    model = Classifier(768, output_num).to(device)

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer_args = dict(lr=5e-4)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)

    res = []
    best_val = 0
    train_step = 0
    
    train_length = len(train_ids)
    valid_length = len(valid_ids)
    test_length = len(test_ids)
    for epoch in np.arange(args.n_epoch) + 1:
        print(f"This is epoch {epoch}...")
        model.train()
        train_losses = []
        print(f"Training begin...")
        for i in range(0, train_length, args.batch_size):
            if i + args.batch_size > train_length: train_idx = train_ids[i:train_length]
            else:  train_idx = train_ids[i:i+args.batch_size]
            
            # generate embs for current batch
            node_rep = generate_lm_embs(all_name, tokenizer, lm_encoder, train_idx, device)
            node_rep = torch.tensor(node_rep).to(device).float()
            res  = model.forward(node_rep)
            ylabel = generate_lp_labels(graph, train_idx, train_pairs, output_num)
            loss = criterion(res, torch.FloatTensor(ylabel).to(device))

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step(train_step)
            del res, loss
        
        print(f"Evaluating begin...")
        model.eval()
        valid_losses = []
        with torch.no_grad():
            valid_res = []
            for i in range(0, valid_length, args.batch_size):
                if i + args.batch_size > valid_length: valid_idx = valid_ids[i:valid_length]
                else:  valid_idx = valid_ids[i:i+args.batch_size]
                node_rep = generate_lm_embs(all_name, tokenizer, lm_encoder, valid_idx, device)
                node_rep = torch.tensor(node_rep).to(device).float()
                res = model.forward(node_rep)
                ylabel = generate_lp_labels(graph, valid_idx, valid_pairs, output_num)
                loss = criterion(res, torch.FloatTensor(ylabel).to(device))
                valid_losses += [loss.cpu().detach().tolist()]

                for ai, bi in zip(ylabel, res.argsort(descending = True)):
                    valid_res += [ai[bi.cpu().numpy()]]

            # calculate ndcg        
            valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
            if valid_ndcg > best_val:
                best_val = valid_ndcg
                torch.save(model, os.path.join(args.model_dir, args.task_name))

            print(("Epoch: %d  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") % \
                  (epoch, optimizer.param_groups[0]['lr'], np.average(train_losses), np.average(valid_losses), valid_ndcg))
        
    # for test
    print(f"Testing begin...")
    best_model = torch.load(os.path.join(args.model_dir, args.task_name))
    best_model.eval()
    model = best_model
    with torch.no_grad():
        test_res = []
        for i in range(0, test_length, args.batch_size):
            if i + args.batch_size > test_length: test_idx = test_ids[i:test_length]
            else:  test_idx = test_ids[i:i+args.batch_size]
            node_rep = generate_lm_embs(all_name, tokenizer, lm_encoder, test_idx, device)
            node_rep = torch.tensor(node_rep).to(device).float()
            res = model.forward(node_rep)
            ylabel = generate_lp_labels(graph, test_idx, test_pairs, output_num)
            
            for ai, bi in zip(ylabel, res.argsort(descending = True)):
                test_res += [ai[bi.cpu().numpy()]]
        test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
        print('Best Test NDCG: %.4f' % np.average(test_ndcg))
        test_mrr = mean_reciprocal_rank(test_res)
        print('Best Test MRR:  %.4f' % np.average(test_mrr))