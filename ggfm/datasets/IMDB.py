import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data.utils import download_url, extract_zip


class IMDB(gg):
    r"""
    The IMDB dataset is a collection of movie-related information, including
    details about movies, TV series, podcasts, home videos, video games, and
    streaming content. It encompasses data such as cast and crew, plot summaries,
    trivia, ratings, and user and critic reviews. This dataset is widely used for
    applications in recommendation systems, sentiment analysis, and media analytics.

    Parameters
    ----------
    <接口信息>

    Statistics
    ----------
    .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: imdb_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Movies
          - 690,227
          - Number of movies listed in the dataset.
        * - TV Series
          - 270,752
          - Number of TV series listed in the dataset.
        * - Short Films
          - 1,028,843
          - Number of short films listed in the dataset.
        * - TV Episodes
          - 8,425,740
          - Number of TV episodes listed in the dataset.
        * - TV Mini Series
          - 57,884
          - Number of TV mini series listed in the dataset.
        * - TV Movies
          - 148,960
          - Number of TV movies listed in the dataset.
        * - TV Specials
          - 50,441
          - Number of TV specials listed in the dataset.
        * - TV Shorts
          - 10,471
          - Number of TV shorts listed in the dataset.
        * - Video Games
          - 40,357
          - Number of video games listed in the dataset.
        * - Videos
          - 194,020
          - Number of videos listed in the dataset.
        * - Music Videos
          - 180,621
          - Number of music videos listed in the dataset.
        * - Podcast Series
          - 66,266
          - Number of podcast series listed in the dataset.
        * - Podcast Episodes
          - 10,332,211
          - Number of podcast episodes listed in the dataset.

    For more detailed information, please refer to the official IMDb dataset page:
    https://developer.imdb.com/non-commercial-datasets/
    see in ggfm.nginx.show/download/dataset/IMDB

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
