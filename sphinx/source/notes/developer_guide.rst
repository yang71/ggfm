Developer Guide
=================
.. toctree::
   :maxdepth: 2
   :titlesonly:

Evaluate a new dataset
----------------------
TODO @jiayi
.. You can specify your dataset if necessary. In this section we use HGBn-ACM as an example for the node classification dataset.

.. How to build a new dataset
.. >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. **First step: Process dataset**

.. We give a `demo <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/debug/HGBn-ACM2dgl.py>`_ to process the HGBn-ACM, which is
.. a node classification dataset.

.. First, download the HGBn-ACM from the `Link <https://www.biendata.xyz/hgb/#/datasets>`_.
.. After that, we process it as a `dgl.heterograph <https://docs.dgl.ai/en/latest/guide/graph-heterogeneous.html#guide-graph-heterogeneous>`_.

.. The following code snippet is an example of creating a heterogeneous graph in DGL.

.. .. code:: python

..     >>> import dgl
..     >>> import torch as th

..     >>> # Create a heterograph with 3 node types and 3 edges types.
..     >>> graph_data = {
..     ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
..     ...    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
..     ...    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
..     ... }
..     >>> g = dgl.heterograph(graph_data)
..     >>> g.ntypes
..     ['disease', 'drug', 'gene']
..     >>> g.etypes
..     ['interacts', 'interacts', 'treats']
..     >>> g.canonical_etypes
..     [('drug', 'interacts', 'drug'),
..      ('drug', 'interacts', 'gene'),
..      ('drug', 'treats', 'disease')]

.. We recommend to set the feature name as ``'h'``.

.. .. code:: python

..     >>> g.nodes['drug'].data['h'] = th.ones(3, 1)

.. DGL provides :func:`dgl.save_graphs` and :func:`dgl.load_graphs` respectively for saving and loading
.. heterogeneous graphs in binary format.
.. So we can use `dgl.save_graphs <https://docs.dgl.ai/en/latest/generated/dgl.save_graphs.html#>`_ to store graphs into the disk.

.. .. code:: python

..     >>> dgl.save_graphs("demo_graph.bin", g)

.. **Second step: Add extra information**

.. We can get a binary file named *demo_graph.bin* after the first step, and we should move it into the directory *openhgnn/dataset/*.
.. The next step is to specify information in the `NodeClassificationDataset.py <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/dataset/NodeClassificationDataset.py#L145>`_

.. For example, we should set the *category*, *num_classes* and *multi_label* (if necessary) with ``"paper"``, ``3``, ``True``, representing the node type to predict classes for,
.. the number of classes, and whether the task is multi-label classification respectively.
.. Please refer to :ref:`Base Node Classification Dataset <api-base-node-dataset>` for more details.

.. .. code:: python

..     if name_dataset == 'demo_graph':
..         data_path = './openhgnn/dataset/demo_graph.bin'
..         g, _ = load_graphs(data_path)
..         g = g[0].long()
..         self.category = 'author'
..         self.num_classes = 4
..         self.multi_label = False

.. **Third step: optional**

.. We can use ``demo_graph`` as our dataset name to evaluate an existing model.

.. .. code:: bash

..     python main.py -m GTN -d demo_graph -t node_classification -g 0 --use_best_config

.. If you have another dataset name, you should also modify the `build_dataset <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/dataset/__init__.py>`_.

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
