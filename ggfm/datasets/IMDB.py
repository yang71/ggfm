import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data.utils import download_url, extract_zip


class IMDB(gg):
    r"""
    The IMDb dataset is a comprehensive collection of movie-related information,
    including details about movies, TV series, podcasts, home videos, video games,
    and streaming content. It encompasses data such as cast and crew, plot summaries,
    trivia, ratings, and user and critic reviews. This dataset is widely used for applications
    in recommendation systems, sentiment analysis, and media analytics.
    A subset of the IMDb movie database is used in the "MAGNN: Metapath Aggregated Graph
    Neural Network for Heterogeneous Graph Embedding" <https://arxiv.org/abs/2002.01680>_ paper.
    This subset focuses on heterogeneous graph structures involving various entities such as movies,
    actors, directors, genres, and more, facilitating research in graph neural networks and their
    applications to multimedia data.

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
        * - Movie
          - 4,278
          - Number of movies in the dataset.
        * - Actor
          - 5,257
          - Number of actors in the dataset.
        * - Director
          - 2,081
          - Number of directors in the dataset.

    For more detailed information, please refer to the official IMDb dataset page:
    https://developer.imdb.com/non-commercial-datasets/
    see in ggfm.nginx.show/download/dataset/IMDB


    Dataset Preprocessing
    ---------------------
    The IMDB dataset is processed by creating a heterogeneous graph where nodes represent entities like movie, actor, and director. The edges represent various relationships between these entities. The dataset is preprocessed in the following steps:

    1. **Load Node Data**: Load CSV files for movie, actor, and director.
    2. **Add Nodes**: For each type of entity, nodes are added to the graph with specific attributes (e.g., 'type', 'id', 'name').
    3. **Load Edge Data**: Various CSV files are loaded to add relationships between nodes, such as actors acting in movies and directors directing movies.
    4. **Add Edges**: The edges are added dynamically based on the relationship type and source/target node types.
    5. **Add Node Features**: Each node type is assigned relevant features, which could include attributes like names, labels, or other node-specific information.

    Graph Class Implementation
    --------------------------
    # Graph class represents a heterogeneous graph supporting dynamic addition of nodes and edges.
    class Graph():
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
            self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
            if directed:
                self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
            else:
                self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time
            self.times[time] = True

        def update_node(self, node):
            # Update the node by adding new attributes if necessary.
            nbl = self.node_bacward[node['type']]
            ser = self.add_node(node)
            for k in node:
                if k not in nbl[ser]:
                    nbl[ser][k] = node[k]

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

    def add_node_features(graph):
        # Add features to the graph's nodes.
        graph.node_feature['movie'] = pd.DataFrame(graph.node_bacward['movie'])
        graph.node_feature['actor'] = pd.DataFrame(graph.node_bacward['actor'])
        graph.node_feature['director'] = pd.DataFrame(graph.node_bacward['director'])
        return graph

    def convert_defaultdict_to_dict(graph):
        # Convert the edge list from defaultdict to a regular dictionary and ensure that all edges are correctly processed.
        edg = {}

        # Iterate through each level of the edge list in the graph
        for k1 in graph.edge_list:
            if k1 not in edg:
                edg[k1] = {}
            for k2 in graph.edge_list[k1]:
                if k2 not in edg[k1]:
                    edg[k1][k2] = {}
                for k3 in graph.edge_list[k1][k2]:
                    if k3 not in edg[k1][k2]:
                        edg[k1][k2][k3] = {}
                    for e1 in graph.edge_list[k1][k2][k3]:
                        if len(graph.edge_list[k1][k2][k3][e1]) == 0:
                            continue
                        edg[k1][k2][k3][e1] = {}
                        for e2 in graph.edge_list[k1][k2][k3][e1]:
                            edg[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]

        # After processing, we now have a fully initialized edge list in `edg`
        graph.edge_list = edg  # Update the graph's edge list with the manually created dictionary
        return graph

    def load_data(file_path, sep='\t', usecols=None):
        # Load data from a CSV file, returning a DataFrame.
        return pd.read_csv(file_path, header=0, sep=sep, usecols=usecols)

    def build_graph():
        # Build the graph by loading node and edge data, then adding them to the graph structure.
        graph = Graph()

        # Load node data
        movie_data = load_data('/path/to/file.txt')
        actor_data = load_data('/path/to/file.tsv')
        director_data = load_data('/path/to/file.tsv')

        # Add nodes to the graph
        for _, row in movie_data.iterrows():
            graph.add_node({'type': 'movie', 'id': int(row['movie_id']), 'name': row['movie_name'], 'label': row['label']})
        for _, row in actor_data.iterrows():
            graph.add_node({'type': 'actor', 'id': int(row['id']), 'name': row['name']})
        for _, row in director_data.iterrows():
            graph.add_node({'type': 'director', 'id': int(row['id']), 'name': row['name']})

        # Load edge data
        actor_movie_edges = load_data('/path/to/file.txt')
        director_movie_edges = load_data('/path/to/file.txt')

        # Add edges to the graph
        for _, row in actor_movie_edges.iterrows():
            graph.add_edge(source_node={'type': 'actor', 'id': int(row['actor_id'])},
                           target_node={'type': 'movie', 'id': int(row['movie_id'])},
                           relation_type='actor_to_movie')
        for _, row in director_movie_edges.iterrows():
            graph.add_edge(source_node={'type': 'director', 'id': int(row['director_id'])},
                           target_node={'type': 'movie', 'id': int(row['movie_id'])},
                           relation_type='director_to_movie')

        return graph

    # Save the graph
    graph = build_graph()
    graph = add_node_features(graph)
    graph = convert_defaultdict_to_dict(graph)

    # Save the processed graph
    save_path = '/path/to/graph.pk'
    with open(save_path, 'wb') as f:
        dill.dump(graph, f)
    print(f"Graph with node features saved as {save_path}")


    """

    url = 'todo'

    def __init__(self, root: str = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, force_reload: bool = False):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> str:
        return 'PubMed_pre_data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        data = gg()

        pass
        # todo

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
