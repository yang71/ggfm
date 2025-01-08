Quick Start
==========================

Run experiments
---------------
Our models are located in the `examples` folder. Most GFM models follow the "pretrain, fine-tuning" paradigm, so each example typically includes two types of files: `pretrain.py` and `{task}_ft.py` (where `task` can be either node classification or link prediction, for example: in WalkLM, there are `pretrain.py`, `nc_ft.py`, and `lp_ft.py`).

Taking WalkLM as an example, we guide the user on how to implement model pretraining and fine-tuning.

During the pretraining and fine-tuning processes, users can either customize parameters or run with the default parameters we have set.

First, the user can run `pretrain.py` to generate a pretrained model, for example: 

.. code:: bash

   # cd ./examples/walklm
   # set parameters if necessary
   python pretrain.py

Alternatively, the pretrained model can be directly downloaded from `here <https://github.com/yang71/ggfm/tree/master>`_ to be used for downstream task fine-tuning.

Once the pretrained model parameters are obtained, the user can proceed with fine-tuning, for example: 

.. code:: bash

   # cd ./examples/walklm
   # set parameters if necessary
   python nc_ft.py

.. note::
   If users wish to switch datasets during training, they can simply modify the relevant parameters. 
   If users want to customize a dataset, they can refer to the `Evaluate a new dataset` section in the `Developer Guide`.


Finally, the user can obtain the training results for node classification and link prediction tasks. 
We use `Accuracy` as the evaluation metric for node classification, and `NDCG` and `MRR` as the evaluation metrics for link prediction tasks.