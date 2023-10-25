from abc import ABCMeta, abstractmethod

import torch


class Loss(metaclass=ABCMeta):

    def __init__(self, opts):
        self._opts = opts

    @staticmethod
    def check_grad_required(input):
        """
        Simple check to avoid annoying issues. Prediction results MUST be in derivative graph
        :param input: torch.Tensor or iterable of torch.Tensor
        :return:
        """
        if type(input) == torch.Tensor:
            assert input.requires_grad is True
        else:
            _ = [Loss.check_grad_required(e) for e in input]

    @abstractmethod
    def decompose(self):
        pass

    @abstractmethod
    def compute(self, prediction, target, *args):
        Loss.check_grad_required(prediction)
        assert target.requires_grad is False
