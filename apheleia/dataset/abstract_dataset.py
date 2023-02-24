from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod


class AbstractDataset(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def num_features(self):
        pass
