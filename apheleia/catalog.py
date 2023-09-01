from torch import optim
from apheleia.model.losses import Loss
from apheleia.utils.patterns import Singleton
from apheleia.dataset.abstract_dataset import AbstractDataset
from torch.optim.lr_scheduler import ExponentialLR, StepLR, PolynomialLR, LRScheduler


class Catalog(dict, metaclass=Singleton):

    def choices(self):
        valid_models = []
        for k1, v1 in self.items():
            for k2 in v1.keys():
                naming = f'{k1}_{k2}' if len(k1) > 0 else k2
                valid_models.append(naming)
        return valid_models


class PipelinesCatalog(Catalog):
    pass


class DatasetsCatalog(Catalog):
    @staticmethod
    def load(element, namespace=''):
        return DatasetsCatalog()[namespace][element]

    def __setitem__(self, namespace, value_dict):
        assert issubclass(list(value_dict.values())[0], AbstractDataset)
        super().__setitem__(namespace, value_dict)


class LossesCatalog(Catalog):
    def __setitem__(self, namespace, value_dict):
        assert issubclass(list(value_dict.values())[0], Loss)
        super().__setitem__(namespace, value_dict)

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

    def __setitem__(self, namespace, value_dict):
        assert issubclass(list(value_dict.values())[0], optim.Optimizer)
        super().__setitem__(namespace, value_dict)

    def choices(self):
        return self.keys()


class SchedulesCatalog(Catalog):
    def __init__(self):
        super().__init__()
        self['exp'] = ExponentialLR
        self['step'] = StepLR
        self['poly'] = PolynomialLR

    def __setitem__(self, namespace, value):
        assert issubclass(value, LRScheduler)
        super().__setitem__(namespace, value)

    def choices(self):
        return self.keys()


# decorator wrapper to provide additional parameters
def register(namespace='', alias=None):
    # real decorator function
    def _register(orig_class):
        ns_key = namespace
        classname = orig_class.__name__ if alias is None else alias
        data_dict = {classname: orig_class}
        if issubclass(orig_class, Loss):
            add_in_catalog(LossesCatalog, ns_key, data_dict)
        elif issubclass(orig_class, AbstractDataset):
            add_in_catalog(DatasetsCatalog, ns_key, data_dict)
        else:
            raise Exception('Only classes inheriting from AbstractDataset or Loss can be registered in a Catalog')

        return orig_class
    return _register


def add_in_catalog(catalog, namespace, data_dict):
    if namespace not in catalog():
        catalog()[namespace] = data_dict
    else:
        catalog()[namespace].update(data_dict)
