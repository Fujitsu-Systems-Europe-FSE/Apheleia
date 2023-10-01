from abc import ABC, abstractmethod

import numpy as np


class Meter(ABC):
    def __init__(self, name, fmt=':f', expected_behavior='increasing'):
        assert expected_behavior in ['increasing', 'decreasing'], 'Invalid metric behavior.'
        self.expected_behavior = expected_behavior
        self.name = name
        self.fmt = fmt
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self):
        pass


class SumMeter(Meter):
    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def get(self):
        return self.sum

    def __str__(self):
        fmtstr = '{name} {sum' + self.fmt + '} {count' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class AverageMeter(Meter):
    """Computes and stores the average and current value"""
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get(self):
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class EMAMeter(Meter):
    def __init__(self, name, beta=0.99, fmt=':f'):
        self.beta = beta
        self.name = name
        self.fmt = fmt
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0

    def update(self, x):
        beta = self.beta * (1 - np.exp(-self.count))
        self.count += 1
        delta = self.val - x
        self.val = x + beta * delta

    def get(self):
        return self.val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)
