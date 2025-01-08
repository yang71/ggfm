import torch
import argparse
import numpy as np
from texttable import Texttable
from collections import defaultdict
from ggfm.models import HGT, RNNModel, Matcher, GPT_GNN, get_optimizer
from ggfm.data import renamed_load, sample_subgraph, ndcg_at_k, feature_extractor
from warnings import filterwarnings
filterwarnings("ignore")


def args_print(args):
    _dict = vars(args)
    t = Texttable() 
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())


def randint():
    return np.random.randint(2**32 - 1)


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


def node_sample(seed, graph, target_nodes, time_range, batch_size):
    np.random.seed(seed)
    samp_target_nodes = target_nodes[np.random.choice(len(target_nodes), batch_size)]
    threshold = 0.5
    
    feature, times, edge_list, _, attr = sample_subgraph(graph, time_range, inp = {target_type: samp_target_nodes}, \
                                                         feature_extractor = feature_extractor, sampled_depth = args.sample_depth, \
                                                         sampled_number = args.sample_width)
    

    rem_edge_list = defaultdict(  # source_type
                        lambda: defaultdict(  # relation_type
                            lambda: []  # [target_id, source_id] 
                                ))
    
    # source_type, relation_type, edges[]
    ori_list = {} 
    for source_type in edge_list[target_type]:
        ori_list[source_type] = {}
        for relation_type in edge_list[target_type][source_type]:
            ori_list[source_type][relation_type] = np.array(edge_list[target_type][source_type][relation_type])
            el = []
            for target_ser, source_ser in edge_list[target_type][source_type][relation_type]:
                if relation_type not in rel_stop_list and target_ser < batch_size and np.random.random() > threshold:
                    rem_edge_list[source_type][relation_type] += [[target_ser, source_ser]]
                    continue
                el += [[target_ser, source_ser]]
            el = np.array(el)
            edge_list[target_type][source_type][relation_type] = el
            
            if relation_type == 'self':
                continue
            else:
                if 'rev_' in relation_type:
                    rev_relation = relation_type[4:]
                else:
                    rev_relation = 'rev_' + relation_type
                edge_list[source_type]['paper'][rev_relation] = list(np.stack((el[:,1], el[:,0])).T)
                
    '''
        Adding feature nodes:
    '''
    n_target_nodes = len(feature[target_type])
    feature[target_type] = np.concatenate((feature[target_type], np.zeros([batch_size, feature[target_type].shape[1]])))
    times[target_type]   = np.concatenate((times[target_type], times[target_type][:batch_size]))

    for source_type in edge_list[target_type]:
        for relation_type in edge_list[target_type][source_type]:
            el = []
            for target_ser, source_ser in edge_list[target_type][source_type][relation_type]:
                if target_ser < batch_size:
                    if relation_type == 'self':
                        el += [[target_ser + n_target_nodes, target_ser + n_target_nodes]]
                    else:
                        el += [[target_ser + n_target_nodes, source_ser]]
            if len(el) > 0:
                edge_list[target_type][source_type][relation_type] = \
                    np.concatenate((edge_list[target_type][source_type][relation_type], el))


    rem_edge_lists = {}
    for source_type in rem_edge_list:
        rem_edge_lists[source_type] = {}
        for relation_type in rem_edge_list[source_type]:
            rem_edge_lists[source_type][relation_type] = np.array(rem_edge_list[source_type][relation_type])
    del rem_edge_list
    
    return to_torch(feature, times, edge_list, graph), rem_edge_lists, ori_list, \
            attr[:batch_size], (n_target_nodes, n_target_nodes + batch_size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pre-training GPT-GNN on a given graph (heterogeneous / homogeneous)')

    '''
    GPT-GNN arguments 
    '''
    parser.add_argument('--attr_ratio', type=float, default=0.5, help='Ratio of attr-loss against link-loss, range: [0-1]') 
    parser.add_argument('--attr_type', type=str, default='text', choices=['text', 'vec'], help='The type of attribute decoder')
    parser.add_argument('--neg_samp_num', type=int, default=255, help='Maximum number of negative sample for each target node.')
    parser.add_argument('--queue_size', type=int, default=256, help='Max size of adaptive embedding queue.')
    parser.add_argument('--w2v_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/pretrained_model/w2v_all', help='The address of w2v_model.')

    '''
        Dataset arguments
    '''
    parser.add_argument('--data_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/datasets/graph.pk', help='The address of data for pretrain.')
    parser.add_argument('--pretrained_model_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/pretrained_model/gta_all_cs3', help='The save address for the pretrained model.')
    parser.add_argument('--cuda', type=int, default=1, help='Avaiable GPU ID')      
    parser.add_argument('--sample_depth', type=int, default=6, help='How many layers within a mini-batch subgraph')
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
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate.')
    parser.add_argument('--scheduler', type=str, default='cycle', help='Name of learning rate scheduler.' , choices=['cycle', 'cosine'])
    parser.add_argument('--n_epoch', type=int, default=20, help='Number of epoch to run')
    parser.add_argument('--n_pool', type=int, default=8, help='Number of process to sample subgraph')    
    parser.add_argument('--n_batch', type=int, default=32, help='Number of batch (sampled graphs) for each epoch') 
    parser.add_argument('--batch_size', type=int, default=256, help='Number of output nodes for training')    
    parser.add_argument('--clip', type=float, default=0.5, help='Gradient Norm Clipping') 


    args = parser.parse_args()
    args_print(args)

    if args.cuda != -1: device = torch.device("cuda:" + str(args.cuda))
    else: device = torch.device("cpu")

    graph = renamed_load(open(args.data_dir, 'rb'))

    pre_range   = {t: True for t in graph.times if t != None and t < 2014}
    train_range = {t: True for t in graph.times if t != None and t >= 2014  and t <= 2016}

    # Pretraining at any cost
    # generative pretraining
    pre_target_nodes   = []
    train_target_nodes = []
    target_type = 'paper'
    rel_stop_list = ['self', 'rev_PF_in_L0', 'rev_PF_in_L5', 'rev_PV_Repository', 'rev_PV_Patent']

    for p_id, _time in graph.node_feature[target_type]['time'].items():
        if _time in pre_range:
            pre_target_nodes += [[p_id, _time]]
        elif _time in train_range:
            train_target_nodes += [[p_id, _time]]
    pre_target_nodes = np.array(pre_target_nodes)
    train_target_nodes = np.array(train_target_nodes)


    repeat_num = int(len(pre_target_nodes) / args.batch_size // args.n_batch)

    data, rem_edge_list, ori_edge_list, _, _ = node_sample(randint(), graph, pre_target_nodes, pre_range, args.batch_size)
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
    types = graph.get_types()

    gnn = HGT(conv_name = args.conv_name, in_dim = len(graph.node_feature[target_type]['emb'].values[0]) + 401, n_hid = args.n_hid, \
            n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout, num_types = len(types), \
            num_relations = len(graph.get_meta_graph()) + 1, prev_norm = args.prev_norm, last_norm = args.last_norm)


    if args.attr_type == 'text':  
        from gensim.models import Word2Vec
        w2v_model = Word2Vec.load(args.w2v_dir)
        n_tokens = len(w2v_model.wv.key_to_index)
        attr_decoder = RNNModel(n_word = n_tokens, ninp = gnn.n_hid, \
                nhid = w2v_model.vector_size, nlayers = 2)
        attr_decoder.from_w2v(torch.FloatTensor(w2v_model.wv.vectors))
    else:
        attr_decoder = Matcher(gnn.n_hid, gnn.in_dim)
        
    gpt_gnn = GPT_GNN(gnn = gnn, rem_edge_list = rem_edge_list, attr_decoder = attr_decoder, \
                    neg_queue_size = 0, types = types, neg_samp_num = args.neg_samp_num, device = device)
    gpt_gnn.init_emb.data = node_feature[node_type == node_dict[target_type][1]].mean(dim=0).detach()
    gpt_gnn = gpt_gnn.to(device)



    best_val   = 100000
    train_step = 0
    stats = []
    optimizer_args = dict(lr=args.max_lr, weight_decay=1e-2, eps=1e-06)
    optimizer = torch.optim.AdamW(gpt_gnn.parameters(), **optimizer_args)

    if args.scheduler == 'cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.02, anneal_strategy='linear', final_div_factor=100,\
                            max_lr = args.max_lr, total_steps = repeat_num * args.n_batch * args.n_epoch + 1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, repeat_num * args.n_batch, eta_min=1e-6)

    for epoch in np.arange(args.n_epoch) + 1:
        gpt_gnn.neg_queue_size = args.queue_size * epoch // args.n_epoch
        for batch in np.arange(repeat_num) + 1:
            train_data = []
            for _ in np.arange(args.n_batch):
                train_data.append(node_sample(randint(), graph, pre_target_nodes, pre_range, args.batch_size))
            valid_data = node_sample(randint(), graph, train_target_nodes, train_range, args.batch_size)

            train_link_losses = []
            train_attr_losses = []
            gpt_gnn.train()
            for data, rem_edge_list, ori_edge_list, attr, (start_idx, end_idx) in train_data:
                node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
                node_feature = node_feature.detach()
                node_feature[start_idx : end_idx] = gpt_gnn.init_emb
                node_emb = gpt_gnn.gnn(node_feature.to(device), node_type.to(device), edge_time.to(device), \
                                    edge_index.to(device), edge_type.to(device))

                loss_link, _ = gpt_gnn.link_loss(node_emb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue=True)
                if args.attr_type == 'text':
                    loss_attr = gpt_gnn.text_loss(node_emb[start_idx : end_idx], attr, w2v_model, device)
                else:
                    loss_attr = gpt_gnn.feat_loss(node_emb[start_idx : end_idx], torch.FloatTensor(attr).to(device))


                loss = loss_link + loss_attr * args.attr_ratio


                optimizer.zero_grad() 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gpt_gnn.parameters(), args.clip)
                optimizer.step()

                train_link_losses += [loss_link.item()]
                train_attr_losses += [loss_attr.item()]
                scheduler.step()

            # valid 
            gpt_gnn.eval()
            with torch.no_grad():
                data, rem_edge_list, ori_edge_list, attr, (start_idx, end_idx) = valid_data
                node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
                node_feature = node_feature.detach()
                node_feature[start_idx : end_idx] = gpt_gnn.init_emb
                node_emb = gpt_gnn.gnn(node_feature.to(device), node_type.to(device), edge_time.to(device), \
                                        edge_index.to(device), edge_type.to(device))
                loss_link, ress = gpt_gnn.link_loss(node_emb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = False, update_queue=True)
                loss_link = loss_link.item()
                if args.attr_type == 'text':   
                    loss_attr = gpt_gnn.text_loss(node_emb[start_idx : end_idx], attr, w2v_model, device)
                else:
                    loss_attr = gpt_gnn.feat_loss(node_emb[start_idx : end_idx], torch.FloatTensor(attr).to(device))

                ndcgs = []
                for i in ress:
                    ai = np.zeros(len(i[0]))
                    ai[0] = 1
                    ndcgs += [ndcg_at_k(ai[j.cpu().numpy()], len(j)) for j in i.argsort(descending = True)]     
                    
                valid_loss = loss_link + loss_attr * args.attr_ratio
                
                print(("Epoch: %d, (%d / %d)  LR: %.5f Train Loss: (%.3f, %.3f)  Valid Loss: (%.3f, %.3f)  NDCG: %.3f  Norm: %.3f  queue: %d") % \
                    (epoch, batch, repeat_num, optimizer.param_groups[0]['lr'], np.average(train_link_losses), np.average(train_attr_losses), \
                    loss_link, loss_attr, np.average(ndcgs), node_emb.norm(dim=1).mean(), gpt_gnn.neg_queue_size))  
                
            if valid_loss < best_val:
                best_val = valid_loss
                torch.save(gpt_gnn.state_dict(), args.pretrain_model_dir)
            stats += [[np.average(train_link_losses),  loss_link, loss_attr, valid_loss]]