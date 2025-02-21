import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data.utils import download_url, extract_zip


class YELP(gg):
    r"""
    The Yelp Open Dataset is a comprehensive collection of data from the Yelp platform,
    including information about businesses, user reviews, and user interactions.
    This dataset is widely used for research in areas such as recommendation systems,
    sentiment analysis, and social network analysis.

    Parameters
    ----------
    businesses : list of dict
        A list of dictionaries, each containing information about a business, such as
        name, location, categories, and attributes like hours, parking availability,
        and ambiance.
    reviews : list of dict
        A list of dictionaries, each representing a review, including details like
        the review text, rating, and the user who wrote it.
    users : list of dict
        A list of dictionaries, each containing information about a user, such as
        user ID, name, and the number of reviews written.
    photos : list of dict
        A list of dictionaries, each representing a photo associated with a business,
        including the photo's URL and the business it is associated with.

    Statistics
    ----------
    .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: yelp_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Business
          - 7474
          - Number of business entities in the dataset.
        * - Location
          - 39
          - Number of locations in the dataset.
        * - Phrase
          - 74943
          - Number of phrases in the dataset.
        * - Stars
          - 9
          - Number of star ratings in the dataset.
    For more detailed information, please refer to the official Yelp Open Dataset page:
    https://business.yelp.com/data/resources/open-dataset/
    see in ggfm.nginx.show/download/dataset/YELP



    Dataset Preprocessing
    ---------------------
    The Yelp dataset is processed by creating a heterogeneous graph where nodes represent entities like business, location, phrase, and stars. The edges represent various relationships between these entities. The dataset is preprocessed in the following steps:

    1. **Load Node Data**: Load CSV files for business, location, phrase, and stars.
    2. **Add Nodes**: For each type of entity, nodes are added to the graph with specific attributes (e.g., 'type', 'id', 'name', 'label').
    3. **Load Edge Data**: Various CSV files are loaded to add relationships between nodes, such as businesses being associated with locations or phrases.
    4. **Add Edges**: The edges are added dynamically based on the relationship type and source/target node types.
    5. **Add Node Features**: Each node type is assigned relevant features, which could include attributes like labels or other node-specific information.

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
            '''
                Add bi-directional edges with different relation types
            '''
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

    def extract_edges(file_path):
        # Extract edges from the given file.
        source_nodes = []
        destination_nodes = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            match = re.match(r'Edge ID: \d+ \| Source Node: (\d+) \| Destination Node: (\d+)', line)
            if match:
                source_nodes.append(int(match.group(1)))
                destination_nodes.append(int(match.group(2)))
        return list(zip(source_nodes, destination_nodes))

    def load_data(file_path):
        # Load CSV files, using tab as a separator, and remove any rows with NaN values.
        df = pd.read_csv(file_path, header=0, sep='\t')  # Using tab separator
        return df.dropna()  # Remove rows containing NaN values

    def load_dataa(file_path):
        # Read the data with space separator.
        return pd.read_csv(file_path, header=0, sep=' ')  # Using header=0 to ensure the first row is used as column names

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
                        if len(graph.edge_list[k1][k2][k3][e1]) == 0:  # Skip empty edges
                            continue
                        edg[k1][k2][k3][e1] = {}
                        for e2 in graph.edge_list[k1][k2][k3][e1]:
                            edg[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]

        # After processing, update the graph's edge list with the normal dictionary
        graph.edge_list = edg
        return graph

    def build_graph():
        graph = Graph()

        # Load node data
        business_data = load_dataa('/path/to/file.csv')
        location_data = load_data('/path/to/file.csv')
        phrase_data = load_data('/path/to/file.csv')
        stars_data = load_data('/path/to/file.csv')

        # Add nodes to the graph
        for _, row in business_data.iterrows():
            node = {'type': 'business', 'id': row['id'], 'name': row['name'], 'label': row['label']}
            graph.add_node(node)

        for _, row in location_data.iterrows():
            graph.add_node({'type': 'location', 'id': row['id'], 'name': row['name']})
        for _, row in phrase_data.iterrows():
            graph.add_node({'type': 'phrase', 'id': row['id'], 'name': row['name']})
        for _, row in stars_data.iterrows():
            graph.add_node({'type': 'stars', 'id': row['id'], 'name': row['name']})

        # Load and add edge data
        business_location_edges = extract_edges('/path/to/file.csv')
        for source, target in business_location_edges:
            graph.add_edge({'type': 'business', 'id': source}, {'type': 'location', 'id': target}, relation_type='location')

        business_phrase_edges = extract_edges('/path/to/business-phrase.txt')
        for source, target in business_phrase_edges:
            graph.add_edge({'type': 'business', 'id': source}, {'type': 'phrase', 'id': target}, relation_type='phrase')

        phrase_phrase_edges = extract_edges('/path/to/phrase-phrase.txt')
        for source, target in phrase_phrase_edges:
            graph.add_edge({'type': 'phrase', 'id': source}, {'type': 'phrase', 'id': target}, relation_type='phrase')

        business_stars_edges = extract_edges('/path/to/business-stars.txt')
        for source, target in business_stars_edges:
            graph.add_edge({'type': 'business', 'id': source}, {'type': 'stars', 'id': target}, relation_type='stars')

        return graph

    def add_node_features(graph):
        graph.node_feature['business'] = pd.DataFrame(graph.node_bacward['business'])
        graph.node_feature['location'] = pd.DataFrame(graph.node_bacward['location'])
        graph.node_feature['phrase'] = pd.DataFrame(graph.node_bacward['phrase'])
        graph.node_feature['stars'] = pd.DataFrame(graph.node_bacward['stars'])
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
        return 'YELP_pre_data.pt'

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
