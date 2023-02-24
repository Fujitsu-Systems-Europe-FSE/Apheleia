from abc import ABCMeta, abstractmethod
from torch import optim

import os
import torch


def valid_models():
    """
    Return the list of strings referencing valid models from the defined pipelines

    Returns:
        List[str]: list of valid models
    """
    models = []
    for k1, v1 in TRAINING_PIPELINES.items():
        for k2 in v1:
            models.append(f'{k1}_{k2}')
    return models


OPTIMIZERS = {k.lower(): v for k, v in optim.__dict__.items() if isinstance(v, type) and issubclass(v, optim.Optimizer)}


class Factory(metaclass=ABCMeta):
    """
    Generic factory
    """
    def __init__(self, opts, ctx) -> None:
        self._opts = opts
        self._ctx = ctx
        self._family, self._model = opts.arch.split('_', 1)
        self._save = torch.load(os.path.expanduser(self._opts.models), map_location=torch.device('cpu')) if self._opts.models else None

    @abstractmethod
    def build(self, *args):
        pass
