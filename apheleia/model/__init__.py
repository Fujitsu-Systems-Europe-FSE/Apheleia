from torch import nn
from abc import ABC, abstractmethod
from apheleia.utils.logger import ProjectLogger

import torch
import numpy as np


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
            ProjectLogger().warning(f'Please correct model structure, it cannot be exported in torchscript format {e}')

    def get_grads(self, *args):
        inputs, = args
        _, grad_means, grad_stds, grad_mean_norms, normalizers = grads_analysis(self._stages, inputs, detach=True)
        return [grad_means], [grad_stds], [np.arange(len(grad_stds))], [normalizers], ['stages']

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
