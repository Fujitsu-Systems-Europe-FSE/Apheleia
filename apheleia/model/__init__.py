from torch import nn
from abc import ABC, abstractmethod
from apheleia.utils.logger import ProjectLogger
from apheleia.utils.gradients import calc_net_gradient_norm

import torch


class NeuralNet(ABC, nn.Module):

    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self._opts = opts
        self._build_structure()
        self._weight_init()

    def check_structure(self):
        try:
            _ = torch.jit.script(self)
        except Exception as e:
            class_name = self.__class__.__name__
            ProjectLogger().warning(f'Please correct model ({class_name}) structure, it cannot be exported in torchscript format {e}')

    def get_grads_stats(self):
        norms = calc_net_gradient_norm(self)
        return norms

    @staticmethod
    @abstractmethod
    def model_name():
        pass

    @abstractmethod
    def _build_structure(self):
        pass

    @abstractmethod
    def _weight_init(self):
        pass
