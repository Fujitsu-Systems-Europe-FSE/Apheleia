from abc import ABC, abstractmethod
from apheleia.trainer.trainer import Trainer
from apheleia.utils.logger import ProjectLogger
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

    def _post_loop_hook(self, val_data, test_data, *args):
        if self.global_iter % self._opts.log_interval == 0:
            self._log_epoch()
        self._try_checkpoint()

    def _log_epoch(self):
        ProjectLogger().info('[Step {}] exec time: {:.2f}'.format(self.global_iter, self._e_duration()))
        self._metrics_store.flush(self.global_iter)

    @abstractmethod
    def _optimize(self, *args, **kwargs):
        pass

    @abstractmethod
    def _select_action(self, state):
        pass
