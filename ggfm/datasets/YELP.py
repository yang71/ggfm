import os
import os.path
from typing import Callable, List, Optional

from ggfm.data import graph as gg
from ggfm.data.utils import download_url, extract_zip


class YELP(gg):
    r"""
    The Yelp Open Dataset is a comprehensive collection of data from the Yelp platform,
    including information about businesses, user reviews, and user interactions.
    This dataset is widely used for research in areas such as recommendation systems,
    sentiment analysis, and social network analysis.

    Parameters
    ----------
    businesses : list of dict
        A list of dictionaries, each containing information about a business, such as
        name, location, categories, and attributes like hours, parking availability,
        and ambiance.
    reviews : list of dict
        A list of dictionaries, each representing a review, including details like
        the review text, rating, and the user who wrote it.
    users : list of dict
        A list of dictionaries, each containing information about a user, such as
        user ID, name, and the number of reviews written.
    photos : list of dict
        A list of dictionaries, each representing a photo associated with a business,
        including the photo's URL and the business it is associated with.

    Statistics
    ----------
    .. list-table::
        :header-rows: 1
        :widths: 25 25 25 25
        :align: left
        :name: yelp_dataset_statistics

        * - Data Type
          - Number of Entries
          - Description
        * - Businesses
          - 150,346
          - Number of businesses listed in the dataset.
        * - Reviews
          - 6,990,280
          - Total number of reviews written by users.
        * - Users
          - 1,987,897
          - Number of distinct users who have written reviews.
        * - Photos
          - 200,100
          - Number of photos associated with businesses.
        * - Tips
          - 908,915
          - Number of tips provided by users.
        * - Check-ins
          - 131,930
          - Number of check-ins recorded for businesses.

    For more detailed information, please refer to the official Yelp Open Dataset page:
    https://business.yelp.com/data/resources/open-dataset/
    see in ggfm.nginx.show/download/dataset/YELP
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
