import numpy as np
from numbers import Number
from statistics import mean
import matplotlib.pyplot as plt
from itertools import product

__all__ = [
    'ConfusionMatrixAccumulator',
    'ImageNetAccuracyAccumulator',
    'plot_confusion_matrix',
]


class ConfusionMatrixAccumulator():
    def __init__(self, num_classes):
        """ Confusion Matrix Accumulator

        Args:
            - num_classes: int or (int, int)
                - int: #.classes
                - (int, int): (#.true classes, #.pred classes)
        """
        self.num_classes = num_classes

        if isinstance(num_classes, Number):
            self.matrix = np.zeros((num_classes, num_classes), dtype=np.uint)
        elif isinstance(num_classes, tuple) and len(num_classes) == 2:
            self.matrix = np.zeros(num_classes, dtype=np.uint)
        else:
            raise Exception(f"num_classes type error: {num_classes}")

    def update(self, true, pred):
        """
        Args:
            - true: list of true class indices
            - pred: list of predicted class indices
        """
        if len(true) != len(pred):
            raise Exception(f"length unmatched: {len(ture)} != {len(pred)}")
        for x in zip(true, pred):
            self.matrix[x] += 1

    def compute(self, mode=None, β=1):
        with np.errstate(divide='ignore', invalid='ignore'):
            if mode is None:
                return self.matrix
            elif mode == 'precision':
                result = self.matrix / self.matrix.sum(axis=0, keepdims=True)
            elif mode == 'recall':
                result = self.matrix / self.matrix.sum(axis=1, keepdims=True)
            elif mode == 'f1':
                inv_p = self.matrix.sum(axis=0, keepdims=True) / self.matrix
                inv_r = self.matrix.sum(axis=1, keepdims=True) / self.matrix
                result = 2 / (inv_p + inv_r)
            elif mode == 'fβ':
                inv_p = self.matrix.sum(axis=0, keepdims=True) / self.matrix
                inv_r = self.matrix.sum(axis=1, keepdims=True) / self.matrix
                result = (1 + β * β) / (inv_p + β * β * inv_r)
            else:
                raise Exception(f"unknown mode: {mode}")
        result[np.isnan(result)] = 0
        return result


class ImageNetAccuracyAccumulator():
    def __init__(self, num_classes):
        """ ImageNet Accuracy Accumulator

        Args:
            - num_classes: int
        """
        self.num_classes = num_classes
        self.rank_lists = []

    def update(self, true, pred):
        """
        Args:
            - true: list of list of true class indices
            - pred: list of predicted scores for each class
        """
        if len(true) != len(pred):
            raise Exception(f"length unmatched: {len(ture)} != {len(pred)}")
        for true_indices, pred_scores in zip(true, pred):
            self.rank_lists.append([
                sum(x > pred_scores[i] for x in pred_scores)
                for i in true_indices
            ])

    def compute(self, topk=5):
        corrects = [sum(y < topk for y in x) / len(x) for x in self.rank_lists]
        return mean(corrects)


def plot_confusion_matrix(cm, labs=None, ylabs=None, xlabs=None, cmap='Blues'):
    """
    Reference: scikit-learn/sklearn/metrics/_plot/confusion_matrix.py

    Args:
        - cm: confusion matrix (N x M)
        - lab: label texts (only when N == M)
        - ylab: true label texts
        - xlab: pred label texts
        - cmap: color theme (see 'colormaps' in matplotlib)
    """
    fig, ax = plt.subplots()

    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    text_ = None
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    # display numbers in each boxes
    text_ = np.empty_like(cm, dtype=object)

    thresh = (cm.max() + cm.min()) / 2.0  # text color threshold

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = cmap_max if cm[i, j] < thresh else cmap_min

        text_cm = format(cm[i, j], '.2g')
        if cm.dtype.kind != 'f':
            text_d = format(cm[i, j], 'd')
            if len(text_d) < len(text_cm):
                text_cm = text_d

        text_[i, j] = ax.text(j, i, text_cm,
                              ha="center", va="center", color=color)

    if ylabs is None:
        ylabs = np.arange(cm.shape[0]) if labs is None else labs

    if xlabs is None:
        xlabs = np.arange(cm.shape[1]) if labs is None else labs

    fig.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=xlabs,
           yticklabels=ylabs,
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_ylim((cm.shape[0] - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation='vertical')

    return fig
