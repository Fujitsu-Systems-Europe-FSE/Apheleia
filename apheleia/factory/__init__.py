from abc import ABC, abstractmethod

import os
import torch


class Factory(ABC):
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
