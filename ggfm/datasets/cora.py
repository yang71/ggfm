import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data.utils import download_url, extract_zip


class cora(gg):
    r"""
    The Cora dataset is a widely used benchmark in graph neural network research, comprising 2,708 scientific
    publications in the field of machine learning. These publications are categorized into seven classes: Case-Based,
    Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, and Theory.
    The dataset includes a citation network with 5,429 links, where each publication is represented by a 1,
    433-dimensional binary feature vector indicating the presence or absence of specific words in the document.

    Parameters
    ----------
    <接口信息>

    Statistics
    ----------
    .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: cora_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Authors
          - 2,708
          - Number of unique authors in the dataset.
        * - Papers
          - 2,708
          - Number of papers in the dataset.
        * - Classes
          - 7
          - Number of classes in the dataset.
        * - Citations
          - 5,429
          - Number of citation links between papers.
        * - Features
          - 1,433
          - Number of unique words used as features.
        * - Training Set
          - 140
          - Number of training samples.
        * - Validation Set
          - 500
          - Number of validation samples.
        * - Test Set
          - 1,000
          - Number of test samples.

    For more detailed information, please refer to the official Cora dataset page:
    https://linqs.soe.ucsc.edu/data
    or see in ggfm.nginx.show/download/dataset/cora
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
        return 'cora_pre_data.pt'

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
