import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data.utils import download_url, extract_zip


class OAG_CS(gg):
    r"""
    The OAG-CS dataset is a subgraph of the Open Academic Graph (OAG) focusing on the Computer Science (CS) domain.
    It is designed to facilitate research in areas such as citation analysis, author collaboration networks, and academic trend analysis within the CS field.

    Parameters
    ----------
    <接口信息>

    Statistics
    ----------
    .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: oag_cs_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Paper
          - 546,704
          - Number of paper nodes in the dataset.
        * - Author
          - 511,122
          - Number of author nodes in the dataset.
        * - Venue
          - 6,946
          - Number of venue nodes in the dataset.
        * - Field
          - 45,775
          - Number of field nodes in the dataset.
        * - Affiliation
          - 9,090
          - Number of affiliation nodes in the dataset.

    For more detailed information, please refer to the official OAG-CS dataset page:
    https://www.aminer.cn/oagcs
    see in ggfm.nginx.show/download/dataset/OAG_CS
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
        return 'OAG_CS_pre_data.pt'

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
