from abc import ABC, abstractmethod
from apheleia.trainer.trainer import Trainer
from apheleia.metrics.metric_store import MetricStore


class RLTrainer(Trainer, ABC):

    def __init__(self, opts, net, optims, scheds, loss, validator, metrics: MetricStore, ctx, model_name, *args, **kwargs):
        super().__init__(opts, net, optims, scheds, loss, validator, metrics, ctx, model_name, *args, **kwargs)
        self._environment = opts.env

    def _report_graph(self):
        pass

    @abstractmethod
    def _train_loop(self, *args, **kwargs):
        pass

    @abstractmethod
    def _optimize(self, *args, **kwargs):
        pass

    @abstractmethod
    def _select_action(self, state):
        pass
