from .logger import SimpleLogger
from numpy import ndarray
from torch import Tensor
from numbers import Number


class MulticlassLogger(SimpleLogger):

    def write(self, output, target):
        assert len(output) == len(target)

        if isinstance(output, (ndarray, Tensor)):
            assert len(output.shape) == 2
            output = output.tolist()
        else:
            assert all(len(x) == len(output[0]) for x in output)
            assert all(isinstance(y, Number) for x in output for y in x)

        if isinstance(target, (ndarray, Tensor)):
            assert len(target.shape) == 1
            target = target.tolist()
        else:
            assert all(isinstance(x, int) for x in target)

        for obj in zip(target, output):
            super().write(obj)
