from time import time
from apheleia import ProjectLogger
from abc import abstractmethod, ABC
from apheleia.model.model_store import ModelStore
from apheleia.metrics.metric_store import MetricStore


class Validator(ABC):
    def __init__(self, opts, net: ModelStore, metrics: MetricStore, ctx):
        super(Validator, self).__init__()
        self._ctx = ctx
        self._net = net
        self._opts = opts
        self._metrics = metrics

    def evaluate(self, data, name):
        start_time = time()
        ProjectLogger().info(f'{name} in progress...')
        self._net.eval()
        self._process_evaluate(data, name)
        self._net.train()

        ProjectLogger().info(f'{name} complete. exec time: {time() - start_time:.2f}')

    @abstractmethod
    def _process_evaluate(self, data, name):
        pass

    def _report_metrics(self, eval_name, metric_dict):
        eval_name = eval_name.lower()
        prefixed_metric_dict = {f'{eval_name}/{k}': v for k, v in metric_dict.items()}
        getattr(self._metrics, f'update_{eval_name}')(prefixed_metric_dict)
