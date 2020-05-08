import os
import json
import sklearn.metrics as skm
from .utils import read_logfile


__all__ = [
    'accuracy_score',
    'balanced_accuracy_score',
    'confusion_matrix',
    'save_confusion_matrix',
]


def argmax(list):
    return max(range(len(list)), key=lambda i: list[i])


def accuracy_score(filename, top=1):
    data = read_logfile(filename)

    Nc = sum(sum(x > out[trg] for x in out) < top for trg, out in data)

    Nt = len(data)

    return Nc / Nt


def balanced_accuracy_score(filename):
    data = read_logfile(filename)

    true, pred = [*zip(*data)]

    pred = [argmax(x) for x in pred]

    return skm.balanced_accuracy_score(true, pred)


def confusion_matrix(filename, labels=None, normalize=None):
    data = read_logfile(filename)

    true, pred = [*zip(*data)]

    pred = [argmax(x) for x in pred]

    return skm.confusion_matrix(true, pred, labels=labels, normalize=normalize)


def save_confusion_matrix(filename, output_path,
                          labels=None, normalize=None,
                          cmap='Blues', xticks_rotation='vertical'):
    cm = confusion_matrix(filename, normalize=normalize)
    display = skm.ConfusionMatrixDisplay(cm, labels)
    display = display.plot(cmap=cmap, xticks_rotation=xticks_rotation)
    try:
        os.makedirs(os.path.dirname(output_path))
    except OSError:
        pass
    display.figure_.savefig(output_path)
