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
        * - Authors
          - 113,171,945
          - Number of authors in the dataset.
        * - Papers
          - 172,209,563
          - Number of papers in the dataset.
        * - Conferences
          - 69,397
          - Number of conferences in the dataset.
        * - Journals
          - 52,678
          - Number of journals in the dataset.
        * - Citations
          - 1,300,000,000
          - Number of citation links between papers.
        * - Institutions
          - 26,000
          - Number of institutions involved in academic research.
        * - Fields of Study
          - 153
          - Number of research fields identified in the dataset.

    For more detailed information, please refer to the official MAG dataset page:
    https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/
    see in ggfm.nginx.show/download/dataset/MAG
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
