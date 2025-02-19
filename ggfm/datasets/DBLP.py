import os.path as osp
import shutil
import os

import numpy as np
import pandas as pd
from typing import Callable, Optional
from ggfm.data import graph as gg
from ggfm.data.utils import download_url,read_npz,extract_zip
from ggfm.utils.get_split import get_train_val_test_split

from itertools import product
import scipy.sparse as sp
import os
import os.path as osp
from typing import Callable, List, Optional
import numpy as np


class DBLP(gg):
    r"""A subset of the DBLP computer science bibliography website, as
    collected in the `"MAGNN: Metapath Aggregated Graph Neural Network for
    Heterogeneous Graph Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.

    DBLP is a heterogeneous graph containing four types of entities - authors
    (4,057 nodes), papers (14,328 nodes), terms (7,723 nodes), and conferences
    (20 nodes).
    The authors are divided into four research areas (database, data mining,
    artificial intelligence, information retrieval).
    Each author is described by a bag-of-words representation of their paper
    keywords.
    see in ggfm.nginx.show/download/dataset/DBLP


    Parameters
    ----------
    <接口信息>

    Statistics
    ----------
    .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: imdb_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Authors
          - 4,057
          - Number of authors in the dataset.
        * - Papers
          - 14,328
          - Number of papers in the dataset.
        * - Conferences
          - 20
          - Number of conferences in the dataset.
        * - Terms
          - 7,723
          - Number of terms in the dataset.

    Dataset Preprocessing
    ---------------------
    The DBLP dataset is processed by creating a heterogeneous graph where nodes represent entities like authors, papers, terms, and conferences. The edges represent various relationships between these entities, such as authors writing papers, papers being published in conferences, etc. The dataset is preprocessed in the following steps:

    1. **Load Node Data**: Load CSV files for authors, papers, conferences, and terms.
    2. **Add Nodes**: For each type of entity, nodes are added to the graph with specific attributes (e.g., 'type', 'id', 'name', 'label').
    3. **Load Edge Data**: Various CSV files are loaded to add relationships between nodes, such as authors writing papers, papers being published in conferences, etc.
    4. **Add Edges**: The edges are added dynamically based on the relationship type and source/target node types.
    5. **Add Node Features**: Each node type is assigned relevant features, which could include attributes like labels or other node-specific information.

    Graph Class Implementation
    --------------------------
    ```python
    import dill
    import pandas as pd
    from collections import defaultdict

    class Graph():
        # Graph class represents a heterogeneous graph supporting dynamic addition of nodes and edges.

        def __init__(self):
            # Initialize the Graph class, creating dictionaries for node tracking, edge list, and time.
            super(Graph, self).__init__()
            self.node_forward = defaultdict(lambda: {})
            self.node_bacward = defaultdict(lambda: [])
            self.node_feature = defaultdict(lambda: [])

            # Using defaultdict to initialize a nested dictionary structure for edge list
            self.edge_list = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(
                                lambda: int  # time, default type is int
                            )
                        )
                    )
                )
            )
            self.times = {}

        def add_node(self, node):

            # Add a node to the graph.
            # If the node does not exist, add it and return its index.

            nfl = self.node_forward[node['type']]
            if node['id'] not in nfl:
                self.node_bacward[node['type']].append(node)
                ser = len(nfl)
                nfl[node['id']] = ser
                return ser
            return nfl[node['id']]

        def add_edge(self, source_node, target_node, time=None, relation_type=None, directed=True):

            # Add an edge to the graph, supporting bi-directional edges and different relation types.

            edge = [self.add_node(source_node), self.add_node(target_node)]

            # Add bi-directional edges with different relation types

            self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
            if directed:
                self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
            else:
                self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time
            self.times[time] = True
    
        def get_meta_graph(self):

            # Get the meta-graph, which includes all the node types and relationships.

            metas = []
            for target_type in self.edge_list:
                for source_type in self.edge_list[target_type]:
                    for r_type in self.edge_list[target_type][source_type]:
                        metas.append((target_type, source_type, r_type))
            return metas
    
        def get_types(self):

            # Get all node types in the graph.

            return list(self.node_feature.keys())
    
    # Data loading function
    def load_data(file_path):

        # Load CSV files, using comma as a separator, and remove any rows with NaN values.
        # This function is primarily used for loading node data.

        df = pd.read_csv(file_path, header=0, sep=',')  # Using comma separator
        return df.dropna()  # Remove rows containing NaN values
    
    # Convert defaultdict to a normal dict for easier processing
    def convert_defaultdict_to_dict(graph):

        # Convert the edge list from defaultdict to a regular dictionary and ensure that all edges are correctly processed.

        edg = {}
    
        # Iterate through each level of the edge list in the graph
        for k1 in graph.edge_list:  # Target type (e.g., 'movie')
            if k1 not in edg:
                edg[k1] = {}
            for k2 in graph.edge_list[k1]:  # Source type (e.g., 'actor')
                if k2 not in edg[k1]:
                    edg[k1][k2] = {}
                for k3 in graph.edge_list[k1][k2]:  # Relation type (e.g., 'actor_to_movie')
                    if k3 not in edg[k1][k2]:
                        edg[k1][k2][k3] = {}
                    for e1 in graph.edge_list[k1][k2][k3]:  # Target node id (e.g., movie ID)
                        if len(graph.edge_list[k1][k2][k3][e1]) == 0:  # Skip empty edges
                            continue
                        edg[k1][k2][k3][e1] = {}
                        for e2 in graph.edge_list[k1][k2][k3][e1]:  # Source node id (e.g., actor ID)
                            edg[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]
    
        # After processing, update the graph's edge list with the normal dictionary
        graph.edge_list = edg
        return graph
    
    # Load edge data function
    def load_dataa(file_path):

        # Load edge data from CSV files.

        return pd.read_csv(file_path)
    
    # Build the graph and add nodes and edges
    def build_graph():

        # Build the graph by loading node and edge data, then adding them to the graph structure.

        graph = Graph()
        
        # Load node data
        author_data = load_data('/path/to/file.csv')
        conf_data = load_data('/path/to/file.csv')
        paper_data = load_data('/path/to/file.csv')
        term_data = load_data('/path/to/file.csv')
    
        # Add nodes to the graph
        for _, row in author_data.iterrows():
            graph.add_node({'type': 'author', 'id': int(row['id']), 'name': row['name'], 'label': row['label']})
        for _, row in conf_data.iterrows():
            graph.add_node({'type': 'conf', 'id': int(row['id']), 'name': row['name']})
        for _, row in paper_data.iterrows():
            graph.add_node({'type': 'paper', 'id': int(row['id']), 'name': row['name']})
        for _, row in term_data.iterrows():
            graph.add_node({'type': 'term', 'id': int(row['id']), 'name': row['name']})
    
        # Load and add edge data to the graph
        author_write_edges = load_dataa('/path/to/file.csv')
        conf_receive_edges = load_dataa('/path/to/file.csv')
        paper_was_published_in_term_edges = load_dataa('/path/to/file.csv')
        paper_was_received_by_conf_edges = load_dataa('/path/to/file.csv')
        paper_was_written_by_author_edges = load_dataa('/path/to/file.csv')
        term_publish_paper_edges = load_dataa('/path/to/file.csv')
    
        # Add edges to the graph
        for _, row in author_write_edges.iterrows():
            graph.add_edge(source_node={'type': 'author', 'id': int(row['src_id'])},
                        target_node={'type': 'paper', 'id': int(row['dst_id'])},
                        relation_type='write')
        # Add other edge types similarly...
    
        return graph
    
    # Add features to the graph's nodes
    def add_node_features(graph):

        # Add node features to the graph.

        graph.node_feature['author'] = pd.DataFrame(graph.node_bacward['author'])
        graph.node_feature['conf'] = pd.DataFrame(graph.node_bacward['conf'])
        graph.node_feature['paper'] = pd.DataFrame(graph.node_bacward['paper'])
        graph.node_feature['term'] = pd.DataFrame(graph.node_bacward['term'])
        return graph
    
    # Build and save the graph
    graph = build_graph()
    graph = add_node_features(graph)
    
    # Convert edge_list to a regular dictionary before saving
    graph = convert_defaultdict_to_dict(graph)
    
    # Save the processed graph
    save_path = '/path/to/save/graph.pk'
    with open(save_path, 'wb') as f:
        dill.dump(graph, f)
    print(f"Graph with node features saved as {save_path}")


    """





    url = 'https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=1'

    def __init__(self, root: str = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, force_reload: bool = False):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'adjM.npz', 'features_0.npz', 'features_1.npz', 'features_2.npy',
            'labels.npy', 'node_types.npy', 'train_val_test_idx.npz'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'DBLP_pre_data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        data = gg()

        # node_types = ['author', 'paper', 'term', 'conference']
        # for i, node_type in enumerate(node_types[:2]):
        #     x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
        #     data[node_type].x = tlx.convert_to_tensor(x.todense(), dtype=tlx.float32)
        #
        # x = np.load(osp.join(self.raw_dir, 'features_2.npy'))
        # data['term'].x = tlx.convert_to_tensor(x, dtype=tlx.int64)
        #
        # node_type_idx = np.load(osp.join(self.raw_dir, 'node_types.npy'))
        # node_type_idx = tlx.convert_to_tensor(node_type_idx, dtype=tlx.int64)
        # data['conference'].num_nodes = int(tlx.reduce_sum(tlx.cast(node_type_idx == 3, dtype=tlx.int64)))
        #
        # y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        # data['author'].y = tlx.convert_to_tensor(y, dtype=tlx.int64)
        #
        # split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        # for name in ['train', 'val', 'test']:
        #     idx = split[f'{name}_idx']
        #     idx = tlx.convert_to_tensor(idx, dtype=tlx.int64)
        #     mask = tlx.zeros((data['author'].num_nodes,), dtype=tlx.bool)
        #     mask = tlx.convert_to_numpy(mask)
        #     mask[idx] = True
        #     data['author'][f'{name}_mask'] = tlx.convert_to_tensor(mask, dtype=tlx.bool)
        #
        # s = {}
        # N_a = data['author'].num_nodes
        # N_p = data['paper'].num_nodes
        # N_t = data['term'].num_nodes
        # N_c = data['conference'].num_nodes
        # s['author'] = (0, N_a)
        # s['paper'] = (N_a, N_a + N_p)
        # s['term'] = (N_a + N_p, N_a + N_p + N_t)
        # s['conference'] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)
        #
        # A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))
        # for src, dst in product(node_types, node_types):
        #     A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
        #     if A_sub.nnz > 0:
        #         row = tlx.convert_to_tensor(A_sub.row, dtype=tlx.int64)
        #         col = tlx.convert_to_tensor(A_sub.col, dtype=tlx.int64)
        #         data[src, dst].edge_index = tlx.stack([row, col], axis=0)
        #
        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)
        #
        # self.save_data(self.collate([data]), self.processed_paths[0])
        pass
        # todo

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'