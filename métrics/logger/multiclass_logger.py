from .logger import SimpleLogger
from numpy import ndarray
from torch import tensor
from numbers import Number


class MulticlassLogger(SimpleLogger):

    def __init__(self, num_classes, *path):
        self.num_classes = num_classes
        super().__init__(*path)

    def write(self, output, target):
        assert len(output) == len(target)

        if isinstance(output, (ndarray, tensor)):
            assert output.shape == (len(target), self.num_classes)
            output = output.tolist()
        else:
            assert all(len(x) == self.num_classes for x in output)
            assert all(isinstance(y, Number) for x in output for y in x)

        if isinstance(target, (ndarray, tensor)):
            assert len(target.shape) == 1
            target = target.tolist()
        else:
            assert all(isinstance(x, int) for x in target)

        assert all(0 <= x < self.num_classes for x in target)

        for obj in zip(target, output):
            super().write(obj)
