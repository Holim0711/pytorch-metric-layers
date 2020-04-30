import os
import json
import sklearn.metrics as skm


__all__ = [
    'accuracy_score',
    'confusion_matrix',
    'save_confusion_matrix',
]


def argmax(list):
    return max(range(len(list)), key=lambda i: list[i])


def accuracy_score(filename, top=1):
    with open(filename) as file:
        data = [json.loads(line) for line in file]

    n_right = sum(sum(x > out[trg] for x in out) < top for trg, out in data)
    n_total = len(data)
    return  n_right / n_total


def confusion_matrix(filename, labels=None, normalize=None):
    with open(filename) as file:
        data = [json.loads(line) for line in file]

    trg = [x for x, y in data]
    out = [argmax(y) for x, y in data]
    return skm.confusion_matrix(trg, out, labels=labels, normalize=normalize)


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
