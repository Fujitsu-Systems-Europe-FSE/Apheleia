from torch import nn
from apheleia.model.losses import Loss


class CrossEntropy(Loss):
    def __init__(self, opts):
        super().__init__(opts)
        self._ce = nn.CrossEntropyLoss()
        self.components = ['loss/cross_entropy']

    def decompose(self):
        return {'loss/cross_entropy': [self._ce_loss_value]}

    def compute(self, prediction, target, *args):
        super(CrossEntropy, self).compute(prediction, target)
        if type(prediction) == tuple:
            prediction, _ = prediction
        self._ce_loss_value = self._ce(prediction, target)
        return self._ce_loss_value
