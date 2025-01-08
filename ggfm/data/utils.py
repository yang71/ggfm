import pickle
import numpy as np
from texttable import Texttable

def open_pkl_file(file_path):
    r"""
    Open pickle file.

    Parameters
    ----------
    file_path: str
        File path for loading pickle files.
    """
    with open(file_path, 'rb') as file:
        file_content = pickle.load(file)
        return file_content


def save_pkl_file(file_path, contents):
    r"""
    Save pickle file.

    Parameters
    ----------
    file_path: str
        File path for saving pickle files.
    contents: list
        Contents for saving.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(contents, file)
    print("having saved pkl...")


def open_txt_file(file_path):
    r"""
    Open txt file.

    Parameters
    ----------
    file_path: str
        File path for loading txt files.
    """
    with open(file_path, 'r') as file:
        contents = [line.rstrip("\n") for line in file.readlines()]
        return contents


def save_txt_file(file_path, contents):
    r"""
    Save txt file.

    Parameters
    ----------
    file_path: str
        File path for saving txt files.
    contents: list
        Contents for saving.
    """
    with open(file_path, 'w') as file:
        for paragraph in contents:
            file.write(paragraph + "\n")
    print("having saved txt...")


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


def ndcg_at_k(r, k):
    r"""

    Compute the Normalized Discounted Cumulative Gain (NDCG) at rank k.

    Parameters
    ----------
    r: list
        A list of relevance scores representing the ranking of items.
    k: int
        The rank at which to compute NDCG.
    
    Returns
    -------
    float
      The Normalized Discounted Cumulative Gain (NDCG) value.
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    r"""
    Compute the Mean Reciprocal Rank (MRR) for a list of relevance scores.

    Parameters
    ----------
    rs: list of arrays
        A list of relevance score arrays where each array represents the indices of relevant items.

    Returns
    -------
    list
        A list of MRR values for each query.
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def args_print(args):
    r"""
    Print argments.

    Parameters
    ----------
    args: object
        args
    """
    _dict = vars(args)
    t = Texttable() 
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())