from pathlib import Path
from abc import ABC, abstractmethod

import torch


class Factory(ABC):
    """
    Generic factory
    """
    def __init__(self, opts, ctx) -> None:
        self._opts = opts
        self._ctx = ctx

        self._namespace = ''
        self._model = opts.arch
        if '_' in opts.arch:
            self._namespace, self._model = opts.arch.split('_', 1)
        self._save = torch.load(Path(self._opts.models).expanduser(), map_location=torch.device('cpu')) if self._opts.models else None

    @abstractmethod
    def build(self, *args):
        pass
