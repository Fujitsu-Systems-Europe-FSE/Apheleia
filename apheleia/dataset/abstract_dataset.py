from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod


class AbstractDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def num_features(self):
        pass


class ImageDataset(AbstractDataset):
    pass


class VectorDataset(AbstractDataset):
    pass


class Memory(AbstractDataset):
    pass
