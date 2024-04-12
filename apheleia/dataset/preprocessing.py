from abc import ABC, abstractmethod

import numpy as np


class Preprocessor(ABC):

    def lazy_fit(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    @abstractmethod
    def inverse_transform(self, data):
        pass


class Normalizer(Preprocessor):
    def __init__(self):
        self._max = None
        self._min = None

    def lazy_fit(self, min_values, max_values):
        self._min = min_values
        self._max = max_values

    def fit(self, data):
        if not isinstance(data, np.ndarray):
            data = data.numpy()

        self._max = data.max(axis=0)
        self._min = data.min(axis=0)

    def transform(self, data):
        return (data - self._min) / (self._max - self._min)

    def inverse_transform(self, data):
        return data * (self._max - self._min) + self._min


class Standardizer(Preprocessor):
    def __init__(self):
        self._mean = None
        self._std = None

    def lazy_fit(self, mean_values, std_values):
        self._mean = mean_values
        self._std = std_values

    def fit(self, data):
        if not isinstance(data, np.ndarray):
            data = data.numpy()

        self._mean = data.mean(axis=0)
        self._std = data.std(axis=0)

    def transform(self, data):
        return (data - self._mean) / self._std

    def inverse_transform(self, data):
        return data * self._std + self._mean
