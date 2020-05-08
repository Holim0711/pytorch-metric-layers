import json
from statistics import mean
from itertools import chain
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score
from .utils import read_logfile


__all__ = [
    'multilabel_recall_score',
    'multilabel_AP_score',
    'lrAP_score',
]


def make_topk_multihot(pred, k):
    topk_indices = np.argpartition(pred, -k)[-k:].tolist()
    pred = np.zeros(len(pred), dtype=int)
    pred[topk_indices] = 1
    return pred.tolist()


def multilabel_recall_score(filename, t=None, method='macro'):
    data = read_logfile(filename)

    true, pred = [*zip(*data)]

    if t == 0 or isinstance(t, float):
        pred = [[y > t for y in x] for x in pred]
    elif isinstance(t, int):
        pred = [make_topk_multihot(x, t) for x in pred]
    else:
        raise ValueError(f"Unknown type of t: {type(t)}")

    if method == 'macro':
        true = [*zip(*true)]
        pred = [*zip(*pred)]
        return mean(recall_score(y, ŷ) for y, ŷ in zip(true, pred))
    elif method == 'micro':
        true = chain.from_iterable(true)
        pred = chain.from_iterable(pred)
        return recall_score(true, pred)
    else:
        raise ValueError(f"Unknown method: {method}")


def multilabel_AP_score(filename, method='macro'):
    data = read_logfile(filename)

    true, pred = [*zip(*data)]

    if method == 'macro':  # case of mAP
        true = [*zip(*true)]
        pred = [*zip(*pred)]
        return mean(average_precision_score(y, ŷ) for y, ŷ in zip(true, pred))
    elif method == 'micro':
        return average_precision_score(true, pred, average='micro')
    else:
        raise ValueError(f"Unknown method: {method}")


def lrAP_score(filename):
    data = read_logfile(filename)
    return label_ranking_average_precision_score(*zip(*data))
