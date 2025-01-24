from collections import defaultdict

import dill


class Graph():
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" 
    <https://arxiv.org/abs/2003.01332>`_ paper.
    
    A Graph object describe a homogeneous/heterogeneous graph. 
    The node-related information is included in node_feature: a DataFrame containing all features
    The link-related information is included in edge_list: index the adjacancy matrix (time) by 
        <target_type, source_type, relation_type, target_id, source_id>
    """

    def __init__(self):
        super(Graph, self).__init__()
        '''
            node_forward and bacward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict(  # target_type
            lambda: defaultdict(  # source_type
                lambda: defaultdict(  # relation_type
                    lambda: defaultdict(  # target_id
                        lambda: defaultdict(  # source_id
                            lambda: int  # time
                        )))))
        self.times = {}

    def add_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]  # type: [node(dict), node(dict),]
            ser = len(nfl)
            nfl[node['id']] = ser  # id -> cur_length
            return ser
        return nfl[node['id']]

    def add_edge(self, source_node, target_node, time=None, relation_type=None, directed=True):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
        if directed:
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time
        self.times[time] = True

    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas

    def get_types(self):
        return list(self.node_feature.keys())


class HomogeneousGraph(Graph):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "data":
            renamed_module = "ggfm.data.graph"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    r"""
        Load Graph object.
    """
    return RenameUnpickler(file_obj).load()
