import os
import os.path
from ggfm.data import graph as gg
from typing import Callable, List, Optional
from ggfm.data import download_url, extract_zip


class YELP:
    r"""
    The Yelp Open Dataset is a comprehensive collection of data from the Yelp platform,
    including information about businesses, user reviews, and user interactions.
    This dataset is widely used for research in areas such as recommendation systems,
    sentiment analysis, and social network analysis.


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
        return 'YELP_pre_data.pt'

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
