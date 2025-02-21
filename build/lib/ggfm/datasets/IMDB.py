import os
import os.path
from ggfm.data import graph as gg
from typing import Callable, List, Optional
from ggfm.data import download_url, extract_zip


class IMDB:
    r"""
    The IMDb dataset is a comprehensive collection of movie-related information,
    including details about movies, TV series, podcasts, home videos, video games,
    and streaming content. It encompasses data such as cast and crew, plot summaries,
    trivia, ratings, and user and critic reviews. This dataset is widely used for applications
    in recommendation systems, sentiment analysis, and media analytics.
    A subset of the IMDb movie database is used in the "MAGNN: Metapath Aggregated Graph
    Neural Network for Heterogeneous Graph Embedding" <https://arxiv.org/abs/2002.01680>_ paper.
    This subset focuses on heterogeneous graph structures involving various entities such as movies,
    actors, directors, genres, and more, facilitating research in graph neural networks and their
    applications to multimedia data.
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
