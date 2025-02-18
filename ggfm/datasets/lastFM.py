import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data.utils import download_url, extract_zip


class LastFM(gg):
    r"""
    The Last.fm dataset is a comprehensive collection of user listening histories from the Last.fm music platform. It
    includes detailed records of user interactions with artists and tracks, making it valuable for research in areas
    such as music recommendation systems, user behavior analysis, and social network analysis within the music domain.

    Parameters
    ----------
    <接口信息>

    Statistics
    ----------
    .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: lastfm_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Users
          - 992
          - Number of unique users in the dataset.
        * - Artists
          - 107,528
          - Number of unique artists in the dataset.
        * - Tracks
          - 69,420
          - Number of unique tracks in the dataset.
        * - Listening Events
          - 19,150,868
          - Total number of listening events recorded.
        * - Tags
          - 11,946
          - Number of unique tags assigned to tracks.
        * - Tag Assignments
          - 186,479
          - Total number of tag assignments in the dataset.

    For more detailed information, please refer to the official Last.fm dataset page:
    https://www.last.fm/api
    see in ggfm.nginx.show/download/dataset/lastFM
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
        return 'lastFM_pre_data.pt'

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
