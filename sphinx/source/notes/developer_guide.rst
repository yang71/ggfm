Developer Guide
=================
.. toctree::
   :maxdepth: 2
   :titlesonly:

Evaluate a new dataset
=================
**Overview and Source of the DBLP Dataset**

The **DBLP dataset** is a subset extracted from the DBLP computer science bibliography website. The data collection process follows the method outlined in the paper **"MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding"**. This dataset includes multiple entities (such as authors, papers, conferences, and terms) along with various relationships between these entities (e.g., authors writing papers, papers published in conferences, etc.). The goal is to transform the DBLP dataset into a heterogeneous graph structure suitable for graph-based machine learning tasks like Graph Neural Networks (GNNs).

**Applicability of Custom Datasets**

Although this example uses the DBLP dataset, the preprocessing steps shown here are general and can be applied to other heterogeneous graph datasets. When working with other datasets, you can modify the steps to accommodate different node types and relationships. For example, you can add new node types (such as companies, countries, or products) and define relationships between them (e.g., customer purchasing products, user watching videos). You can also assign custom node features based on the specific requirements of your dataset.

**Preprocessing Steps for Building a Heterogeneous Graph with DBLP Dataset**

Next, we will outline how to preprocess the DBLP dataset and build a heterogeneous graph. The process involves the following steps:

*1. Load Node Data*

The first step is to load the different entity data from the DBLP dataset. These data are typically stored in CSV files. The main entities in the DBLP dataset are:

- **Authors**
- **Papers**
- **Conferences**
- **Terms**

Each entity's CSV file contains attributes such as ID, name, and labels. For example, the author's file may contain columns like `author_id`, `author_name`, and `author_label`. We load and extract the data from these files using the `load_data` function.

*2. Add Nodes to the Graph*

After loading the node data, the next step is to add these entities as nodes to the heterogeneous graph. Each node should have the following attributes:

- **Type (type)**: The type of the node (e.g., author, paper, conference, term).
- **ID (id)**: A unique identifier for the node.
- **Name (name)**: The name of the node (e.g., author's name, paper title, etc.).
- **Label (label)**: If available, a label or classification for the node.

Using the `add_node` function, each entity is added to the graph, ensuring that the node's type and attributes are correctly assigned.

*3. Load Edge Data*

Next, we need to load the edge data that represent the relationships between the nodes. In the DBLP dataset, common relationships include:

- **Author writes Paper**: Connects author nodes to the papers they have written.
- **Paper published in Conference**: Connects paper nodes to the conferences in which they were published.
- **Paper contains Term**: Connects paper nodes to related term nodes.

These relationships are stored in separate CSV files, which can be loaded using the `load_data` function. Each edge file contains pairs of nodes (e.g., author-paper, paper-conference) and defines the relationship between them.

*4. Add Edges to the Graph*

The `add_edge` function is used to add edges based on the relationships defined in the edge data. This function establishes connections between source and target nodes based on the relationship type (e.g., author-paper, paper-conference). The function supports both directed and undirected edges, and it can handle multiple types of relationships.

*5. Assign Node Features*

After adding the nodes and edges, the next step is to assign features to each node. Node features are additional information that can help improve performance in graph-based machine learning tasks. For instance:

- **Author nodes**: May have features such as the number of publications, research topics, or collaboration networks.
- **Paper nodes**: May include features like keywords, citation count, or publication year.

The `add_node_features` function is used to assign features to each node type (e.g., authors, papers, conferences), storing them in a dictionary of node features. These features can be represented as vectors, scalars, or other data types depending on the task.

**Definition and Role of the Graph Class**

In the above steps, the **Graph** class plays a central role. This class is responsible for managing the entire process of building the heterogeneous graph, including adding nodes, adding edges, and assigning node features. Specifically, the **Graph** class defines how nodes and edges are created and maintained, and it provides the following methods:

- **add_node**: Adds a node with its attributes.
- **add_edge**: Adds an edge between nodes based on a specific relationship.
- **add_node_features**: Assigns features to nodes.

By using the **Graph** class, users can easily build and manipulate heterogeneous graphs without worrying about the underlying implementation details.

**Graph Class Implementation**

Below is the implementation of the **Graph** class, which supports the dynamic addition of nodes and edges, along with the handling of node features and relationships:

.. code-block:: python

    import dill
    import pandas as pd
    from collections import defaultdict

    class Graph():
        # Graph class represents a heterogeneous graph supporting dynamic addition of nodes and edges.
        def __init__(self):
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
            nfl = self.node_forward[node['type']]
            if node['id'] not in nfl:
                self.node_bacward[node['type']].append(node)
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

        def get_meta_graph(self):
            metas = []
            for target_type in self.edge_list:
                for source_type in self.edge_list[target_type]:
                    for r_type in self.edge_list[target_type][source_type]:
                        metas.append((target_type, source_type, r_type))
            return metas

        def get_types(self):
            return list(self.node_feature.keys())

    def load_data(file_path):
        df = pd.read_csv(file_path, header=0, sep=',')
        return df.dropna()

    def convert_defaultdict_to_dict(graph):
        edg = {}
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

        graph.edge_list = edg
        return graph

    def load_dataa(file_path):
        return pd.read_csv(file_path)

    def build_graph():
        graph = Graph()

        author_data = load_data('/path/to/file.csv')
        conf_data = load_data('/path/to/file.csv')
        paper_data = load_data('/path/to/file.csv')
        term_data = load_data('/path/to/file.csv')

        for _, row in author_data.iterrows():
            graph.add_node({'type': 'author', 'id': int(row['id']), 'name': row['name'], 'label': row['label']})
        for _, row in conf_data.iterrows():
            graph.add_node({'type': 'conf', 'id': int(row['id']), 'name': row['name']})
        for _, row in paper_data.iterrows():
            graph.add_node({'type': 'paper', 'id': int(row['id']), 'name': row['name']})
        for _, row in term_data.iterrows():
            graph.add_node({'type': 'term', 'id': int(row['id']), 'name': row['name']})

        author_write_edges = load_dataa('/path/to/file.csv')
        conf_receive_edges = load_dataa('/path/to/file.csv')
        paper_was_published_in_term_edges = load_dataa('/path/to/file.csv')
        paper_was_received_by_conf_edges = load_dataa('/path/to/file.csv')
        paper_was_written_by_author_edges = load_dataa('/path/to/file.csv')
        term_publish_paper_edges = load_dataa('/path/to/file.csv')

        for _, row in author_write_edges.iterrows():
            graph.add_edge(source_node={'type': 'author', 'id': int(row['src_id'])},
                           target_node={'type': 'paper', 'id': int(row['dst_id'])},
                           relation_type='write')

        return graph

    def add_node_features(graph):
        graph.node_feature['author'] = pd.DataFrame(graph.node_bacward['author'])
        graph.node_feature['conf'] = pd.DataFrame(graph.node_bacward['conf'])
        graph.node_feature['paper'] = pd.DataFrame(graph.node_bacward['paper'])
        graph.node_feature['term'] = pd.DataFrame(graph.node_bacward['term'])
        return graph

    graph = build_graph()
    graph = add_node_features(graph)
    graph = convert_defaultdict_to_dict(graph)

    save_path = '/path/to/save/graph.pk'
    with open(save_path, 'wb') as f:
        dill.dump(graph, f)

    print(f"Graph with node features saved as {save_path}")

Apply a new example
-------------------
In this section, we will guide users on how to add a new example.

**Step 1: Add Pretrain and Fine-tuning Scripts**

Most existing graph-based models follow the "pretrain, fine-tuning" paradigm. Therefore, the implementation of an example typically consists of two types of scripts: the pretrain and fine-tuning scripts. If the model does not support multi-task fine-tuning, there can be multiple fine-tuning scripts.

For example, in WalkLM, the `example` folder contains `pretrain.py`, `nc_ft.py`, and `lp_ft.py`.

Therefore, when users add a new example, they only need to provide the complete versions of these two types of scripts.

.. note::
    Please note that existing graph foundation models have various pretraining and fine-tuning methods, and there are no strict limitations on the specific implementation process.
    However, to ensure fairness in baseline comparisons in benchmarks, we restrict the inputs and evaluation metrics for fine-tuning in each example.

**Step 2: Add Graph Preprocessing, Conv, and Model**

During the implementation process, it is highly likely that Graph Preprocessing (e.g., designing instructions in instruction fine-tuning), as well as adding convolution layers and models, will be involved.

We encourage users to abstract the Graph Preprocessing process into a separate class or method and add it to `ggfm.data`.

Following the guidelines of `PyG <https://www.pyg.org/>`_ and `DGL <https://github.com/dmlc/dgl>`_, for adding convolution layers and models, we adhere to the same conventions.