import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data.utils import download_url, extract_zip


class MAG(gg):
    r"""
    The Microsoft Academic Graph (MAG) is a comprehensive dataset that encompasses a vast collection of academic
    publications, authors, conferences, journals, and citation relationships. It serves as a valuable resource for
    research in bibliometrics, citation analysis, and academic network analysis.

    The dataset is sourced from https://ogb.stanford.edu/docs/lsc/mag240m/, which provides the MAG240M version,
    containing over 240 million citation links and other related information from academic papers across various domains.

    Parameters
    ----------
    <接口信息>

    Statistics
    ----------
    .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: mag_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Paper
          - 121,751,666
          - Number of paper nodes in the dataset.
        * - Author
          - 122,383,112
          - Number of author nodes in the dataset.
        * - Institution
          - 25,721
          - Number of institution nodes in the dataset.

    For more detailed information, please refer to the official MAG dataset page:
    https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/
    see in ggfm.nginx.show/download/dataset/MAG


    Dataset Preprocessing
    ---------------------
    The MAG dataset is processed by creating a heterogeneous graph where nodes represent entities like papers, authors, and institutions. The edges represent various relationships between these entities. The dataset is preprocessed in the following steps:

    1. **Load Node Data**: Load CSV files for papers, authors, and institutions.
    2. **Add Nodes**: For each type of entity, nodes are added to the graph with specific attributes (e.g., 'type', 'id', 'name').
    3. **Load Edge Data**: Various `.npy` files are loaded to add relationships between nodes, such as authors writing papers and institutions being affiliated with authors.
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

        def get_meta_graph(self):
            # Get the meta-graph, which includes all the node types and relationships.
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
        graph.node_feature['paper'] = pd.DataFrame(graph.node_bacward['paper'])
        graph.node_feature['author'] = pd.DataFrame(graph.node_bacward['author'])
        graph.node_feature['institution'] = pd.DataFrame(graph.node_bacward['institution'])
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
                        if len(graph.edge_list[k1][k2][k3][e1]) == 0:  # Skip empty edges
                            continue
                        edg[k1][k2][k3][e1] = {}
                        for e2 in graph.edge_list[k1][k2][k3][e1]:
                            edg[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]

        # After processing, we now have a fully initialized edge list in `edg`
        graph.edge_list = edg  # Update the graph's edge list with the manually created dictionary
        return graph

    def build_graph():
        # Build the graph by loading node and edge data, then adding them to the graph structure.
        graph = Graph()

        # Load node data
        paper_data = load_data('/path/to/file.csv', sep=',', usecols=['idx', 'title'])
        author_data = load_data('/path/to/file.csv', sep=',', usecols=['idx'])
        institution_data = load_data('/path/to/file.csv', sep=',', usecols=['idx'])

        # Add nodes to the graph
        for _, row in paper_data.iterrows():
            graph.add_node({'type': 'paper', 'id': row['idx'], 'title': row['title']})
        for _, row in author_data.iterrows():
            graph.add_node({'type': 'author', 'id': row['idx']})
        for _, row in institution_data.iterrows():
            graph.add_node({'type': 'institution', 'id': row['idx']})

        # Load and add edge data (example with numpy files)
        author_write_paper_edges = load_npy('/path/to/file.npy')
        for i in range(author_write_paper_edges.shape[1]):
            graph.add_edge(
                {'type': 'author', 'id': author_write_paper_edges[0, i]},
                {'type': 'paper', 'id': author_write_paper_edges[1, i]},
                relation_type='author_write_paper'
            )

        return graph

    # Build and save the graph
    graph = build_graph()
    graph = add_node_features(graph)
    graph = convert_defaultdict_to_dict(graph)

    # Save the processed graph
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
        return 'mag_pre_data.pt'

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
