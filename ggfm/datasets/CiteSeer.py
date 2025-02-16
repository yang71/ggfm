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
        * - Authors
          - 3,312
          - Number of unique authors in the dataset.
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
        * - Training Set
          - 120
          - Number of training samples.
        * - Validation Set
          - 500
          - Number of validation samples.
        * - Test Set
          - 1,000
          - Number of test samples.

    For more detailed information, please refer to the official CiteSeer dataset page:
    https://relational.fel.cvut.cz/dataset/CiteSeer
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
