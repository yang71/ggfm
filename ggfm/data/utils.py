import pickle
import numpy as np
from texttable import Texttable

def open_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        file_content = pickle.load(file)
        return file_content


def save_pkl_file(file_path, contents):
    with open(file_path, 'wb') as file:
        pickle.dump(contents, file)
    print("having saved pkl...")


def open_txt_file(file_path):
    with open(file_path, 'r') as file:
        contents = [line.rstrip("\n") for line in file.readlines()]
        return contents


def save_txt_file(file_path, contents):
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
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def args_print(args):
    _dict = vars(args)
    t = Texttable() 
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())


# def data_trans(res, ylabel):
#     res = res.tolist()

#     m, n  = len(res), len(res[0])
#     cur_labels = [[0] * n for _ in range(m)]
#     cur_preds = [[0] * n for _ in range(m)]
#     non_zero_indices = [np.nonzero(row)[0].tolist() for row in ylabel]

#     for i in range(len(non_zero_indices)):
#         for j in range(len(non_zero_indices[i])):
#             cur_labels[i][non_zero_indices[i][j]] = 1
    
#     label_num_for_each_row = [len(np.nonzero(row)[0]) for row in ylabel]
#     for i in range(len(label_num_for_each_row)):
#         cur_num = label_num_for_each_row[i]
#         arr = np.array(res[i])
#         top_k_indices = arr.argsort()[-cur_num:][::-1]
#         for idx in top_k_indices:
#             cur_preds[i][idx] = 1
    
#     return cur_preds, cur_labels