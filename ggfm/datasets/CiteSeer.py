import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data.utils import download_url, extract_zip


class CiteSeer(gg):
    r"""
    The CiteSeer dataset is a widely used benchmark in graph neural network research, comprising 3,312 scientific
    publications in the field of computer science. These publications are categorized into six classes: Agents, AI,
    DB, IR, ML, and HCI. The dataset includes a citation network with 4,732 links, where each publication is
    represented by a 3,703-dimensional binary feature vector indicating the presence or absence of specific words in
    the document.

    Parameters
    ----------
    <接口信息>

    Statistics
    ----------
    .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: citeseer_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Papers
          - 3,312
          - Number of papers in the dataset.
        * - Classes
          - 6
          - Number of classes in the dataset.
        * - Citations
          - 4,732
          - Number of citation links between papers.
        * - Features
          - 3,703
          - Number of unique words used as features.

    For more detailed information, please refer to the official CiteSeer dataset page:
    https://relational.fel.cvut.cz/dataset/CiteSeer
    see in ggfm.nginx.show/download/dataset/CiteSeer


    Dataset Preprocessing
    ---------------------
    The Cora dataset is processed by creating a heterogeneous graph where nodes represent entities like paper. The edges represent various relationships between these entities. The dataset is preprocessed in the following steps:

    1. **Load Node Data**: Load CSV files for papers and their feature vectors.
    2. **Add Nodes**: For each paper, nodes are added to the graph with specific attributes such as 'type', 'id', 'features', and 'label'.
    3. **Load Edge Data**: Various CSV files are loaded to add relationships between papers, such as citation links.
    4. **Add Edges**: The edges are added dynamically based on the relationship type and source/target node types (e.g., citation links between papers).
    5. **Add Node Features**: Each node type is assigned relevant features, which could include attributes like feature vectors or labels.

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

    # A function to load the dataset
    def load_data(file_path):
        # Load data from a CSV file, returning a DataFrame.
        df = pd.read_csv(file_path, header=0, sep='\t')
        return df.dropna()  # Remove NaN values to prevent errors

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

    def build_graph():
        # Build the graph by loading node and edge data, then adding them to the graph structure.
        graph = Graph()

        # 1. Load Node Data
        content_file = '/path/to/cora_content_with_labels.txt'
        node_data = pd.read_csv(content_file, sep=' ', header=None)

        # Create a mapping from string ID to integer index
        id_map = {id_str: idx for idx, id_str in enumerate(node_data.iloc[:, 0])}

        # Add nodes to the graph
        for _, row in node_data.iterrows():
            graph.add_node({
                'type': 'paper',
                'id': id_map[row.iloc[0]],  # Map string ID to integer index
                'features': row.iloc[1:-1].tolist(),  # Extract features
                'label': row.iloc[-1]  # Extract label
            })

        # 2. Load Citation Data
        cites_file = '/path/to/cora_cites.txt'
        edge_data = pd.read_csv(cites_file, sep='\t', header=None)

        # Add edges to the graph
        for _, row in edge_data.iterrows():
            source_id = id_map.get(row.iloc[0], -1)
            target_id = id_map.get(row.iloc[1], -1)

            # If a node is missing from the mapping, skip the edge
            if source_id == -1 or target_id == -1:
                continue

            graph.add_edge(
                {'type': 'paper', 'id': source_id},  # Source node
                {'type': 'paper', 'id': target_id},  # Target node
                relation_type='cites'
            )

        return graph


    # Save the graph
    graph = build_graph()
    graph = add_node_features(graph)
    graph = convert_defaultdict_to_dict(graph)
    save_path = '/path/to/save/graph.pk'
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
        return 'CiteSeer_pre_data.pt'

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
