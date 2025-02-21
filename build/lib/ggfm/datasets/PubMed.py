import os
import os.path
from ggfm.data import graph as gg
from typing import Callable, List, Optional
from ggfm.data import download_url, extract_zip


class PubMed:
    r"""
    The PubMed dataset is a comprehensive collection of biomedical literature,
    including research articles, clinical studies, and reviews. It serves as a
    valuable resource for various applications, such as literature mining,
    biomedical text mining, and information retrieval.
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
