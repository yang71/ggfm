import pickle
import zipfile

import numpy as np
from texttable import Texttable
import os
import os.path as osp
import ssl
import sys
import urllib
from tqdm import tqdm
from typing import Optional
import scipy.sparse as sp

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




def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise

def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder.

        Parameters
        ----------
        url: str
            The url.
        folder: str
            The folder.
        log: bool, optional
            If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
        filename: str, optional
            The name of the file.

    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]
    if os.environ.get('GGL_GITHUB_PROXY') == 'TRUE' and ('raw.githubusercontent.com' in url or 'github.com' in url):
        url = 'https://ghproxy.com/' + url
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(url, context=context)

    file_size = response.getheader('Content-Length', '0')

    # print(f"downloading {filename} ...")
    file_size = int(file_size)
    if file_size == 0:
        print(f"Remote file size not found.")

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        # add download progress bar
        with tqdm(total=file_size, unit='B', unit_divisor=1024, unit_scale=True, desc=f'{filename}') as pbar:
            chunk_size = 10 * 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(chunk_size)

    return path


def download_google_url(id: str, folder: str,
                        filename: str, log: bool = True):
    r"""Downloads the content of a Google Drive ID to a specific folder."""
    url = f'https://drive.usercontent.google.com/download?id={id}&confirm=t'
    return download_url(url, folder, log, filename)


def read_npz(path):
    with np.load(path) as f:
        return parse_npz(f)

# parse_npz: see in gammagl


def maybe_log(path, log=True):
    if log:
        print(f'Extracting {path}', file=sys.stderr)
def extract_zip(path: str, folder: str, log: bool = True):
    r"""Extracts a zip archive to a specific folder.

    Parameters
    ----------
    path: str
        The path to the tar archive.
    folder: str
        The folder.
    log: bool, optional
        If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)