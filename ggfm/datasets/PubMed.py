import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data.utils import download_url, extract_zip


class PubMed(gg):
    r"""
    The PubMed dataset is a comprehensive collection of biomedical literature,
    including research articles, clinical studies, and reviews. It serves as a
    valuable resource for various applications, such as literature mining,
    biomedical text mining, and information retrieval.

    Parameters
    ----------
    <接口信息>

    Statistics
    ----------
    .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: pubmed_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Disease
          - 20,163
          - Number of disease nodes in the dataset.
        * - Gene
          - 13,561
          - Number of gene nodes in the dataset.
        * - Chemical
          - 26,522
          - Number of chemical nodes in the dataset.
        * - Species
          - 2,863
          - Number of species nodes in the dataset.

    see in ggfm.nginx.show/download/dataset/PubMed

    Dataset Preprocessing
    ---------------------
    The PubMed dataset is processed by creating a heterogeneous graph where nodes represent entities like disease, gene, chemical, and species. The edges represent various relationships between these entities. The dataset is preprocessed in the following steps:

    1. **Load Node Data**: Load CSV files for disease, gene, chemical, and species.
    2. **Add Nodes**: For each type of entity, nodes are added to the graph with specific attributes (e.g., 'type', 'id', 'name', 'label').
    3. **Load Edge Data**: Various CSV files are loaded to add relationships between nodes, such as diseases being associated with genes or chemicals.
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

    # Manually converting defaultdict to a normal dictionary
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
        disease_data = load_dataa('/path/to/file.csv')
        gene_data = load_data('/path/to/file.csv')
        chemical_data = load_data('/path/to/file.csv')
        species_data = load_data('/path/to/file.csv')

        # Add nodes to the graph
        for _, row in disease_data.iterrows():
            graph.add_node({'type': 'disease', 'id': row['id'], 'name': row['name'], 'label': row['label']})

        for _, row in gene_data.iterrows():
            graph.add_node({'type': 'gene', 'id': row['id'], 'name': row['name']})

        for _, row in chemical_data.iterrows():
            graph.add_node({'type': 'chemical', 'id': row['id'], 'name': row['name']})

        for _, row in species_data.iterrows():
            graph.add_node({'type': 'species', 'id': row['id'], 'name': row['name']})

        # Load and add edges to the graph
        edge_files = {
            ('chemical', 'chemical', 'chemical'): '/path/to/file.csv',
            ('chemical', 'disease', 'chemical_in_disease'): '/path/to/file.csv',
            ('chemical', 'gene', 'chemical_in_gene'): '/path/to/file.csv',
            ('chemical', 'species', 'chemical_in_species'): '/path/to/file.csv',
            ('disease', 'disease', 'disease'): '/path/to/file.csv',
            ('gene', 'gene', 'gene'): '/path/to/file.csv',
            ('gene', 'disease', 'gene_causing_disease'): '/path/to/file.csv',
            ('species', 'species', 'species'): '/path/to/file.csv',
            ('species', 'disease', 'species_with_disease'): '/path/to/file.csv',
            ('species', 'gene', 'species_with_gene'): '/path/to/file.csv',
        }

        for (src, tgt, rel), path in edge_files.items():
            edges = extract_edges(path)
            for source, target in edges:
                graph.add_edge({'type': src, 'id': source}, {'type': tgt, 'id': target}, relation_type=rel)

        return graph

    def add_node_features(graph):
        # Add features to the graph's nodes.
        graph.node_feature['gene'] = pd.DataFrame(graph.node_bacward['gene'])
        graph.node_feature['disease'] = pd.DataFrame(graph.node_bacward['disease'])
        graph.node_feature['chemical'] = pd.DataFrame(graph.node_bacward['chemical'])
        graph.node_feature['species'] = pd.DataFrame(graph.node_bacward['species'])
        return graph


    # Build and save the graph
    graph = build_graph()
    graph = add_node_features(graph)

    # Convert edge_list to a normal dictionary before saving
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
