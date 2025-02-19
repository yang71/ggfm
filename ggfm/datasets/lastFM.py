import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data.utils import download_url, extract_zip


class LastFM(gg):
    r"""
    The Last.fm dataset is a comprehensive collection of user listening histories from the Last.fm music platform. It
    includes detailed records of user interactions with artists and tracks, making it valuable for research in areas
    such as music recommendation systems, user behavior analysis, and social network analysis within the music domain.

    Parameters
    ----------
    <接口信息>

    Statistics
    ----------
    .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: lastfm_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Artist
          - 18,022
          - Number of artists in the dataset.
        * - Tag
          - 11,946
          - Number of tags in the dataset.
        * - User
          - 1,892
          - Number of users in the dataset.

    For more detailed information, please refer to the official Last.fm dataset page:
    https://www.last.fm/api
    see in ggfm.nginx.show/download/dataset/lastFM


    Dataset Preprocessing
    ---------------------
    The LastFM dataset is processed by creating a heterogeneous graph where nodes represent entities like artist, tag, and user. The edges represent various relationships between these entities. The dataset is preprocessed in the following steps:

    1. **Load Node Data**: Load CSV files for artists, tags, and users.
    2. **Add Nodes**: For each type of entity, nodes are added to the graph with specific attributes (e.g., 'type', 'id', 'name').
    3. **Load Edge Data**: Various CSV files are loaded to add relationships between nodes, such as users listening to artists, users tagging artists, and friendships between users.
    4. **Add Edges**: The edges are added dynamically based on the relationship type and source/target node types.
    5. **Add Node Features**: Each node type is assigned relevant features, which could include attributes like names, tags, or other node-specific information.

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
            self.edge_list = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(lambda: int)
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
                self.node_bacward[node['type']] += [node]
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
            types = self.get_types()
            metas = []
            for target_type in self.edge_list:
                for source_type in self.edge_list[target_type]:
                    for r_type in self.edge_list[target_type][source_type]:
                        metas += [(target_type, source_type, r_type)]
            return metas

        def get_types(self):
            # Get all node types in the graph.
            return list(self.node_feature.keys())

    def add_node_features(graph):
        # Add features to the graph's nodes.
        graph.node_feature['artist'] = pd.DataFrame(graph.node_bacward['artist'])
        graph.node_feature['tag'] = pd.DataFrame(graph.node_bacward['tag'])
        graph.node_feature['user'] = pd.DataFrame(graph.node_bacward['user'])
        return graph

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

        # After processing, we now have a fully initialized edge list in `edg`
        graph.edge_list = edg  # Update the graph's edge list with the manually created dictionary
        return graph

    def safe_read_csv(file_path, sep='\t', encoding='utf-8'):
        # Safely read CSV files with fallback encoding.
        try:
            return pd.read_csv(file_path, sep=sep, encoding=encoding)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, sep=sep, encoding='latin1')

    def build_graph():
        # Build the graph by loading node and edge data, then adding them to the graph structure.
        base_path = '/path/to/LastFM'
        graph = Graph()

        # Load node data
        artists_data = safe_read_csv(f'{base_path}/artists.dat')
        tags_data = safe_read_csv(f'{base_path}/tags.dat')

        # Add artist nodes
        for _, row in artists_data.iterrows():
            graph.add_node({'type': 'artist', 'id': row['id'], 'name': row['name']})

        # Add tag nodes (adjusting to actual column names)
        for _, row in tags_data.iterrows():
            graph.add_node({'type': 'tag', 'id': row['tagID'], 'value': row['tagValue']})

        # Load edge data
        user_artists_data = safe_read_csv(f'{base_path}/user_artists.dat')
        user_friends_data = safe_read_csv(f'{base_path}/user_friends.dat')
        user_taggedartists_data = safe_read_csv(f'{base_path}/user_taggedartists.dat')

        # Extract unique user IDs from all edge files
        user_ids = set(user_artists_data['userID']).union(
            user_friends_data['userID'],
            user_friends_data['friendID'],
            user_taggedartists_data['userID']
        )

        # Add user nodes
        for user_id in user_ids:
            graph.add_node({'type': 'user', 'id': user_id})

        # Add user-artist edges
        for _, row in user_artists_data.iterrows():
            graph.add_edge(
                {'type': 'user', 'id': row['userID']},
                {'type': 'artist', 'id': row['artistID']},
                relation_type='listens_to'
            )

        # Add user-tagged-artist edges
        for _, row in user_taggedartists_data.iterrows():
            graph.add_edge(
                {'type': 'user', 'id': row['userID']},
                {'type': 'artist', 'id': row['artistID']},
                relation_type='tags'
            )

        # Add user-friend edges
        for _, row in user_friends_data.iterrows():
            graph.add_edge(
                {'type': 'user', 'id': row['userID']},
                {'type': 'user', 'id': row['friendID']},
                relation_type='friends_with'
            )

        return graph

    # Save the graph
    graph = build_graph()
    graph = add_node_features(graph)
    graph = convert_defaultdict_to_dict(graph)
    save_path = '/path/to/LastFM/graph.pk'
    with open(save_path, 'wb') as f:
        dill.dump(graph, f)

    print(f"Graph saved as {save_path}")

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
        return 'lastFM_pre_data.pt'

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
