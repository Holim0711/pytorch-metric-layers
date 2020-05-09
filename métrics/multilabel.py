import json
from statistics import mean
from itertools import chain
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score
from .utils import read_logfile


__all__ = [
    'multilabel_precision_score',
    'multilabel_recall_score',
    'multilabel_f1_score',
    'multilabel_AP_score',
    'lrAP_score',
]


def make_topk_multihot(pred, k):
    topk_indices = np.argpartition(pred, -k)[-k:].tolist()
    pred = np.zeros(len(pred), dtype=int)
    pred[topk_indices] = 1
    return pred.tolist()


def quantize(pred, t=None):
    if t == 0 or isinstance(t, float):
        return [[y > t for y in x] for x in pred]
    elif isinstance(t, int):
        return [make_topk_multihot(x, t) for x in pred]
    else:
        raise ValueError(f"Unknown type of t: {type(t)}")


def calculate(func, true, pred, method):
    if method == 'macro':
        true = [*zip(*true)]
        pred = [*zip(*pred)]
        return mean(func(y, ŷ) for y, ŷ in zip(true, pred))
    elif method == 'micro':
        true = list(chain.from_iterable(true))
        pred = list(chain.from_iterable(pred))
        return func(true, pred)
    else:
        raise ValueError(f"Unknown method: {method}")


def multilabel_precision_score(filename, t=None, method='macro'):
    data = read_logfile(filename)

    true, pred = [*zip(*data)]

    pred = quantize(pred, t)

    return calculate(precision_score, true, pred, method)


def multilabel_recall_score(filename, t=None, method='macro'):
    data = read_logfile(filename)

    true, pred = [*zip(*data)]

    pred = quantize(pred, t)

    return calculate(recall_score, true, pred, method)


def multilabel_f1_score(filename, t=None, method='macro'):
    data = read_logfile(filename)

    true, pred = [*zip(*data)]

    pred = quantize(pred, t)

    return calculate(f1_score, true, pred, method)


def multilabel_AP_score(filename, method='macro'):
    data = read_logfile(filename)
    return calculate(average_precision_score, *zip(*data), method)


def lrAP_score(filename):
    data = read_logfile(filename)
    return label_ranking_average_precision_score(*zip(*data))
