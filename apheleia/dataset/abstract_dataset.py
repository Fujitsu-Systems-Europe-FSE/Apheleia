from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod


class AbstractDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_preprocessor(self):
        raise NotImplementedError()

    @abstractmethod
    def num_features(self):
        pass


class ImageDataset(AbstractDataset, metaclass=ABCMeta):
    def num_classes(self):
        raise NotImplementedError()


class VectorDataset(AbstractDataset, metaclass=ABCMeta):
    def num_classes(self):
        raise NotImplementedError()

    def features_name(self):
        raise NotImplementedError()


class Memory(AbstractDataset, metaclass=ABCMeta):
    pass
