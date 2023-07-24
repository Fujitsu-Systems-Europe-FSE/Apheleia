from torch import optim
from abc import abstractmethod
from apheleia.utils.patterns import Singleton
from torch.optim.lr_scheduler import ExponentialLR, StepLR, PolynomialLR


class Catalog(dict, metaclass=Singleton):

    @abstractmethod
    def choices(self):
        pass


class PipelinesCatalog(Catalog):
    def choices(self):
        valid_models = []
        for k1, v1 in self.items():
            prefix = f'{k1}_' if len(k1) > 0 else k1
            for k2 in v1.keys():
                valid_models.append(f'{prefix}{k2}')
        return valid_models


class DatasetsCatalog(Catalog):
    def choices(self):
        valid_datasets = []
        for k1, v1 in self.items():
            prefix = f'{k1}_' if len(k1) > 0 else k1
            for k2 in v1.keys():
                valid_datasets.append(f'{prefix}{k2}')
        return valid_datasets


class LossesCatalog(Catalog):
    def choices(self):
        unique_keys = set()
        for k, v in self.items():
            unique_keys.update(list(v.keys()))
        return list(unique_keys)


class OptimizersCatalog(Catalog):
    def __init__(self):
        super().__init__()
        optims = {k.lower(): v for k, v in optim.__dict__.items() if isinstance(v, type) and issubclass(v, optim.Optimizer)}
        self.update(optims)

    def choices(self):
        return self.keys()


class SchedulesCatalog(Catalog):
    def __init__(self):
        super().__init__()
        self['exp'] = ExponentialLR
        self['step'] = StepLR
        self['poly'] = PolynomialLR

    def choices(self):
        return self.keys()
