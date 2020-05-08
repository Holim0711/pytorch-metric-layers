import json
from statistics import mean
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import label_ranking_average_precision_score


__all__ = [
    'multilabel_recall_score',
    'lrap_score',
]


def make_topk_pred(a, k):
    topk_idx = np.argpartition(a, -k)[-k:].tolist()
    ret = np.zeros(len(a), dtype=int)
    ret[topk_idx] = 1
    return ret.tolist()


def multilabel_recall_score(filename, t=None, method='perclass'):
    with open(filename) as file:
        data = [json.loads(line) for line in file]

    true, pred = [*zip(*data)]

    if t == 0 or isinstance(t, float):
        pred = [[y > t for y in x] for x in pred]
    elif isinstance(t, int):
        pred = [make_topk_pred(x, t) for x in pred]
    else:
        raise ValueError(f"Unknown type of t: {type(t)}")

    if method == 'perclass':
        true, pred = [*zip(*true)], [*zip(*pred)]
        return mean([recall_score(y, ŷ) for y, ŷ in zip(true, pred)])
    elif method == 'overall':
        ΣNc = sum(x == (1, 1) for y, ŷ in zip(true, pred) for x in zip(y, ŷ))
        ΣNt = sum(sum(x) for x in true)
        return ΣNc / ΣNt
    else:
        raise ValueError(f"Unknown method: {method}")


def lrap_score(filename):
    with open(filename) as file:
        data = [json.loads(line) for line in file]
    return label_ranking_average_precision_score(*zip(*data))
