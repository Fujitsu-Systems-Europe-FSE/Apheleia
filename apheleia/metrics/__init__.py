from abc import ABC, abstractmethod


class Meter(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self):
        pass
