import numpy as np
from collections import defaultdict


def feature_extractor(layer_data, graph):
    r"""`"GPT-GNN: Generative Pre-Training of Graph Neural Networks"
    <https://arxiv.org/abs/2006.15437>`_ paper.

    Extract relevent features.
    
    Parameters
    ----------
    layer_data: dict
        Sampled node indexes for each node type.
    graph: class:`ggfm.data.Graph`
        Target graph.
    """

    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))  #  origin_indxs
        tims  = np.array(list(layer_data[_type].values()))[:,1]  # times
        
        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=float)
        else:  # intialize as 0
            feature[_type] = np.zeros([len(idxs), 400])
        # 400, 768, 1
        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),\
            np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)
        
        times[_type]   = tims
        indxs[_type]   = idxs
        
        if _type == 'paper':
            attr = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=str)
    return feature, times, indxs, attr


def sample_subgraph(graph, time_range, sampled_depth = 2, sampled_number = 8, inp=None):
    
    r"""`"GPT-GNN: Generative Pre-Training of Graph Neural Networks"
    <https://arxiv.org/abs/2006.15437>`_ paper.

    Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
    Budgets are maintained for each node type, indexed by <node_id, time>.
    Currently sampled nodes are stored in layer_data.
    After nodes are sampled, the sampled adjacancy matrix are constructed.

    Parameters
    ----------
    graph: class:`ggfm.data.Graph`
        Target graph.
    time_range: list
        Time range of target nodes.
    sampled_depth: int, optional
        Sampled depth.
        (default: :obj:`2`)
    sampled_number: int, optional
        Sampled number.
        (default: :obj:`8`)
    inp: dict
        Input data for sampling. 
        `inp = {target_type: samp_target_nodes}`
    """
    layer_data  = defaultdict( #target_type
                        lambda: {} # {target_id: [ser, time]}
                    )
    budget      = defaultdict( #source_type
                                    lambda: defaultdict(  # source_id
                                        lambda: [0., 0] # [sampled_score, time]
                            ))
    
    '''
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    '''
    def add_budget(te, target_id, target_time, layer_data, budget):
        for source_type in te:  # source_type
            tes = te[source_type]  # relation
            for relation_type in tes:  # such as: rev_PV_Conference, rev_PV_Journal
                if relation_type == 'self' or target_id not in tes[relation_type]:
                    continue
                adl = tes[relation_type][target_id]  # {source_id: year, }
                if len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
                for source_id in sampled_ids:
                    source_time = adl[source_id]
                    if source_time == None:
                        source_time = target_time
                    if source_time > np.max(list(time_range.keys())) or source_id in layer_data[source_type]:
                        continue
                    budget[source_type][source_id][0] += 1. / len(sampled_ids)  # score
                    budget[source_type][source_id][1] = source_time  # time

    '''
        First adding the sampled nodes then updating budget.
    '''
    # inp = {target_type: samp_target_nodes}  # inp['paper'].shape = (batch_size, 2) [[id, year], [id, year], ]
    for _type in inp:  # paper
        for _id, _time in inp[_type]:  # id transfer
            layer_data[_type][_id] = [len(layer_data[_type]), _time]  # id -> cur_length // layer_data: {'paper': {id: [cur_length, year], }}
    for _type in inp:  # sampling nodes for each source type of each target type
        te = graph.edge_list[_type]  # such as: paper_venue, paper_paper, paper_field, paper_author
        for _id, _time in inp[_type]:
            add_budget(te, _id, _time, layer_data, budget)
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    '''
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = graph.edge_list[source_type]
            keys = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                sampled_ids = np.arange(len(keys))
            else:
                score = np.array(list(budget[source_type].values()))[:,0] ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), sampled_number, p = score, replace = False) 
            sampled_keys = keys[sampled_ids]
            '''
                First adding the sampled nodes then updating budget.
            '''
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][1]]  # layer_data[source_type] {id: [cur_length, time], }
            for k in sampled_keys:
                add_budget(te, k, budget[source_type][k][1], layer_data, budget)
                budget[source_type].pop(k)
    
    # Prepare feature, time and adjacency matrix for the sampled graph, indxs are the origin indexes, texts are the title information of papers
    feature, times, indxs, texts = feature_extractor(layer_data, graph)
            
    edge_list = defaultdict(  # target_type
                        lambda: defaultdict(  # source_type
                            lambda: defaultdict(  # relation_type
                                lambda: [] # [target_id, source_id] 
                                    )))
    for _type in layer_data:  # {type: {id: [cur_id, year], }, }
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]  # cur_id
            edge_list[_type][_type]['self'] += [[_ser, _ser]]  # add self-loop
    
    '''
        Reconstruct sampled adjacancy matrix by checking whether each
        link exist in the original graph
    '''
    for target_type in graph.edge_list:
        te = graph.edge_list[target_type]
        tld = layer_data[target_type]  # {type: {id: [cur_id, year], }, }
        for source_type in te:
            tes = te[source_type]
            sld  = layer_data[source_type]
            for relation_type in tes:  # relation
                tesr = tes[relation_type]  # target_id
                for target_key in tld:  # sampled target_ids
                    if target_key not in tesr:
                        continue
                    target_ser = tld[target_key][0]  # cur_id
                    for source_key in tesr[target_key]:
                        # Check whether each link (target_id, source_id) exist in original adjacancy matrix
                        if source_key in sld:
                            source_ser = sld[source_key][0]
                            edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]
    
    return feature, times, edge_list, indxs, texts