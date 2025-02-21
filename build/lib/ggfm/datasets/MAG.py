import os
import os.path
from ggfm.data import graph as gg
from typing import Callable, List, Optional
from ggfm.data import download_url, extract_zip


class MAG:
    r"""
    The Microsoft Academic Graph (MAG) is a comprehensive dataset that encompasses a vast collection of academic
    publications, authors, conferences, journals, and citation relationships. It serves as a valuable resource for
    research in bibliometrics, citation analysis, and academic network analysis.

    The dataset is sourced from https://ogb.stanford.edu/docs/lsc/mag240m/, which provides the MAG240M version,
    containing over 240 million citation links and other related information from academic papers across various domains.

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
