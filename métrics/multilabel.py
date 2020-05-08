import os
import json
from sklearn.metrics import label_ranking_average_precision_score


__all__ = [
    'recall_score',
    'lrap_score',
]


def perclass_recall_score(filename, top=None, threshold=None):
    with open(filename) as file:
        data = [json.loads(line) for line in file]
    raise NotImplementedError()


def lrap_score(filename):
    with open(filename) as file:
        data = [json.loads(line) for line in file]
    return label_ranking_average_precision_score(*zip(*data))
