from .Amazon import Amazon
from .CiteSeer import CiteSeer
from .cora import cora
from .DBLP import DBLP
from .IMDB import IMDB
from .lastFM import LastFM
from .MAG import MAG
from .OAG_CS import OAG_CS
from .PubMed import PubMed
from .YELP import YELP
from .Aminer import AMiner

__all__ = [
    'Amazon',
    'AMiner',
    'CiteSeer',
    'cora',
    'DBLP',
    'IMDB',
    'LastFM',
    'MAG',
    'OAG_CS',
    'PubMed',
    'YELP',

]

classes = __all__
