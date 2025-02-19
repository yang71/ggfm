import os.path as osp
import shutil
import os

import numpy as np
import pandas as pd
from typing import Callable, Optional
from ggfm.data import graph as gg
from ggfm.data.utils import download_url,read_npz,extract_zip
from ggfm.utils.get_split import get_train_val_test_split





class AMiner(gg):
    r"""The heterogeneous AMiner dataset from the `"metapath2vec: Scalable
    Representation Learning for Heterogeneous Networks"
    <https://dl.acm.org/doi/pdf/10.1145/3097983.3098036>`_ paper, consisting of nodes from
    type :obj:`"paper"`, :obj:`"author"` and :obj:`"venue"`.
    Aminer is a heterogeneous graph containing three types of entities - author
    (1,693,531 nodes), venue (3,883 nodes), and paper (3,194,405 nodes).
    Venue categories and author research interests are available as ground
    truth labels for a subset of nodes.

    Parameters
    ----------

    Stats:
    ----------
     .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: imdb_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Paper
          - 2,385,922
          - Number of papers in the dataset.
        * - Author
          - 1,712,433
          - Number of authors in the dataset.
        * - Concept
          - 4,055,687
          - Number of concepts in the dataset.
        * - Affiliation
          - 624,751
          - Number of affiliations in the dataset.
        * - Venue
          - 264,840
          - Number of venues in the dataset.


    Dataset Preprocessing
    ---------------------
    The Aminer dataset is processed by creating a heterogeneous graph where nodes represent entities like paper, author, concept, affiliation, and venue.
    The edges represent various relationships between these entities. The dataset is preprocessed in the following steps:

    1. **Load Node Data**: Load CSV files for paper, author, concept, affiliation, and venue.
    2. **Add Nodes**: For each type of entity, nodes are added to the graph with specific attributes (e.g., 'type', 'id', 'name', 'label').
    3. **Load Edge Data**: Various CSV files are loaded to add relationships between nodes, such as authors being linked to papers or venues.
    4. **Add Edges**: The edges are added dynamically based on the relationship type and source/target node types.
    5. **Add Node Features**: Each node type is assigned relevant features, which could include attributes like labels or other node-specific information.

    Graph Class Implementation
    --------------------------
    Graph class represents a heterogeneous graph supporting dynamic addition of nodes and edges.

    class Graph:

        def __init__(self):
            super(Graph, self).__init__()
            self.node_forward = defaultdict(lambda: {})
            self.node_bacward = defaultdict(lambda: [])
            self.node_feature = defaultdict(lambda: [])
            self.edge_list = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(
                                lambda: int  # time，默认为 int 类型
                            )
                        )
                    )
                )
            )
            self.times = {}

        def add_node(self, node):
            nfl = self.node_forward[node['type']]
            if node['id'] not in nfl:
                self.node_bacward[node['type']] += [node]
                ser = len(nfl)
                nfl[node['id']] = ser
                return ser
            return nfl[node['id']]

        def add_edge(self, source_node, target_node, time=None, relation_type=None, directed=True):
            edge = [self.add_node(source_node), self.add_node(target_node)]
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

    # A function to load the dataset
    def load_data(file_path):
        # Safely load data from a CSV file
        return pd.read_csv(file_path, header=0)

    # Adding node features to the graph
    def add_node_features(graph):
        graph.node_feature['paper'] = pd.DataFrame(graph.node_bacward['paper'])
        graph.node_feature['author'] = pd.DataFrame(graph.node_bacward['author'])
        graph.node_feature['concept'] = pd.DataFrame(graph.node_bacward['concept'])
        graph.node_feature['affiliation'] = pd.DataFrame(graph.node_bacward['affiliation'])
        graph.node_feature['venue'] = pd.DataFrame(graph.node_bacward['venue'])
        return graph

    # Convert defaultdict to a regular dictionary
    def convert_defaultdict_to_dict(graph):
        edg = {}

        # 遍历 graph.edge_list 中的每一层
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

        # After processing, we now have a fully initialized edge list in `edg`
        graph.edge_list = edg  # Update the graph's edge list with the manually created dictionary
        return graph


    def build_graph():

        graph = Graph()

        # 1. Load node data
        paper_data = load_data('/path/to/file.csv')
        author_data = load_data('/path/to/file.csv')
        concept_data = load_data('/path/to/file.csv')
        affiliation_data = load_data('/path/to/file.csv')
        venue_data = load_data('/path/to/file.csv')

        # Add nodes to the graph
        for _, row in paper_data.iterrows():
            graph.add_node({'type': 'paper', 'id': row['paperID'], 'name': row['title'], 'year': row['year']})

        for _, row in author_data.iterrows():
            graph.add_node({'type': 'author', 'id': row['authorID'], 'name': row['name']})

        for _, row in concept_data.iterrows():
            graph.add_node({'type': 'concept', 'id': row['conceptID'], 'name': row['name']})

        for _, row in affiliation_data.iterrows():
            graph.add_node({'type': 'affiliation', 'id': row['affiliationID'], 'name': row['name']})

        for _, row in venue_data.iterrows():
            graph.add_node({'type': 'venue', 'id': row['venueID'], 'name': row['name']})

        # 2. Load edge data
        def add_edges(graph, edge_file, src_type, tgt_type, relation):
            edge_data = load_data(edge_file)
            for _, row in edge_data.iterrows():
                graph.add_edge(
                    {'type': src_type, 'id': row['sourceID']},
                    {'type': tgt_type, 'id': row['targetID']},
                    relation_type=relation
                )

        # Add edges between different node types
        add_edges(graph, '/path/to/file.csv', 'author', 'affiliation', 'author_to_affiliation')
        add_edges(graph, '/path/to/file.csv', 'author', 'concept', 'author_to_concept')
        add_edges(graph, '/path/to/file.csv', 'author', 'paper', 'author_to_paper')
        add_edges(graph, '/path/to/file.csv', 'paper', 'venue', 'paper_to_venue')
        add_edges(graph, '/path/to/file.csv', 'paper', 'paper', 'citation')

        return graph

    # Save the graph
    graph = build_graph()
    graph = add_node_features(graph)
    graph = convert_defaultdict_to_dict(graph)

    # Save the processed graph
    save_path = '/path/to/save/graph.pk'
    with open(save_path, 'wb') as f:
        dill.dump(graph, f)
    print(f"Graph saved as {save_path}")

    """

    url = 'https://www.dropbox.com/s/1bnz8r7mofx0osf/net_aminer.zip?dl=1'
    y_url = 'https://www.dropbox.com/s/nkocx16rpl4ydde/label.zip?dl=1'

    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload = force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'id_author.txt', 'id_conf.txt', 'paper.txt', 'paper_author.txt',
            'paper_conf.txt', 'label'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'Aminer_pre_data.pt'

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'net_aminer'), self.raw_dir)
        os.unlink(path)
        path = download_url(self.y_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        # data = gg()
        #
        # # Get author labels.
        # path = osp.join(self.raw_dir, 'id_author.txt')
        # author = pd.read_csv(path, sep='\t', names=['idx', 'name'],
        #                      index_col=1)
        #
        # path = osp.join(self.raw_dir, 'label',
        #                 'googlescholar.8area.author.label.txt')
        # df = pd.read_csv(path, sep=' ', names=['name', 'y'])
        # df = df.join(author, on='name')
        #
        # data['author'].y = tlx.convert_to_tensor(df['y'].values) - 1
        # data['author'].y_index = tlx.convert_to_tensor(df['idx'].values)
        #
        # # Get venue labels.
        # path = osp.join(self.raw_dir, 'id_conf.txt')
        # venue = pd.read_csv(path, sep='\t', names=['idx', 'name'], index_col=1)
        #
        # path = osp.join(self.raw_dir, 'label',
        #                 'googlescholar.8area.venue.label.txt')
        # df = pd.read_csv(path, sep=' ', names=['name', 'y'])
        # df = df.join(venue, on='name')
        #
        # data['venue'].y = tlx.convert_to_tensor(df['y'].values) - 1
        # data['venue'].y_index = tlx.convert_to_tensor(df['idx'].values)
        #
        # # Get paper<->author connectivity.
        # path = osp.join(self.raw_dir, 'paper_author.txt')
        # paper_author = pd.read_csv(path, sep='\t', header=None)
        # paper_author = tlx.convert_to_tensor(paper_author.values)
        # if tlx.BACKEND == 'mindspore':
        #     paper_author = paper_author.T
        # else:
        #     paper_author = tlx.ops.transpose(paper_author)
        # M, N = int(max(paper_author[0]) + 1), int(max(paper_author[1]) + 1)
        # paper_author = coalesce(paper_author, num_nodes=max(M, N))
        # data['paper'].num_nodes = M
        # data['author'].num_nodes = N
        # data['paper', 'written_by', 'author'].edge_index = paper_author
        # paper_author = tlx.convert_to_numpy(paper_author)
        # data['author', 'writes', 'paper'].edge_index = tlx.convert_to_tensor(np.flip(paper_author, axis=0).copy())
        #
        # # Get paper<->venue connectivity.
        # path = osp.join(self.raw_dir, 'paper_conf.txt')
        # paper_venue = pd.read_csv(path, sep='\t', header=None)
        # paper_venue = tlx.convert_to_tensor(paper_venue.values)
        # if tlx.BACKEND == 'mindspore':
        #     paper_venue = paper_venue.T
        # else:
        #     paper_venue = tlx.ops.transpose(paper_venue)
        # M, N = int(max(paper_venue[0]) + 1), int(max(paper_venue[1]) + 1)
        # paper_venue = coalesce(paper_venue, num_nodes=max(M, N))
        # data['venue'].num_nodes = N
        # data['paper', 'published_in', 'venue'].edge_index = paper_venue
        # paper_venue = tlx.convert_to_numpy(paper_venue)
        # data['venue', 'publishes', 'paper'].edge_index = tlx.convert_to_tensor(np.flip(paper_venue, axis=0).copy())
        #
        #
        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)
        #
        # self.save_data(self.collate([data]), self.processed_paths[0])
        pass
        # todo