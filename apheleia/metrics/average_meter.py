from apheleia.metrics import Meter

import numpy as np


class AverageMeter(Meter):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

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
