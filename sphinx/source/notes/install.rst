Install
============

System requrements
------------------
GGFM works with the following operating systems:

* Linux


Python environment requirments
------------------------------

- `Python <https://www.python.org/>`_ >= 3.8
- `PyTorch <https://pytorch.org/get-started/locally/>`_ >= 2.1.0
- `DGL <https://github.com/dmlc/dgl>`_ >= 2.0.0
- `PyG <https://www.pyg.org/>`_ >= 2.4.0

**1. Python environment (Optional):** We recommend using Conda package manager

.. code:: bash

    conda create -n ggfm python=3.8
    source activate ggfm

**2. Pytorch:** Follow their `tutorial <https://pytorch.org/get-started/>`_ to run the proper command according to
your OS and CUDA version. For example:

.. code:: bash

    pip install torch torchvision torchaudio

**3. DGL:** Follow their `tutorial <https://www.dgl.ai/pages/start.html>`_ to run the proper command according to
your OS and CUDA version. For example:

.. code:: bash

    pip install dgl -f https://data.dgl.ai/wheels/repo.html

**4. PyG:** Follow their `tutorial <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_ to run the proper command according to
your OS and CUDA version. For example:

.. code:: bash

    pip install torch_geometric

**4. Install openhgnn:**

* install from pypi

.. code:: bash

    pip install ggfm

* install from source

.. code:: bash

    git clone https://github.com/BUPT-GAMMA/GGFM
    cd GGFM
    pip install .