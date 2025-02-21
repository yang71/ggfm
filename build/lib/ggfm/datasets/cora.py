import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data import download_url, extract_zip


class cora:
    r"""
    The Cora dataset is a widely used benchmark in graph neural network research, comprising 2,708 scientific
    publications in the field of machine learning. These publications are categorized into seven classes: Case-Based,
    Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, and Theory.
    The dataset includes a citation network with 5,429 links, where each publication is represented by a 1,
    433-dimensional binary feature vector indicating the presence or absence of specific words in the document.
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
