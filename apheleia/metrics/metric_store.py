from abc import ABCMeta
from typing import List, Dict
from apheleia.utils.logger import ProjectLogger
from torch.utils.tensorboard import SummaryWriter
from apheleia.metrics.average_meter import AverageMeter

import math


class MetricStore(metaclass=ABCMeta):
    def __init__(self, opts, loss, *args, **kwargs):
        self._opts = opts

        self.train = dict()
        self.validation = dict()

        for c in loss.components:
            self.train[c] = [AverageMeter('')]

        self.best_tgt_metric = 0
        self.sink: SummaryWriter = ...

    def flush(self, epoch):
        for d in [self.train, self.validation]:
            categories = {}
            for k, v in d.items():
                category, metric = k.split('/')
                if category not in categories:
                    categories[category] = {}
                categories[category][metric] = v

            for k, v in categories.items():
                self._flush_category(epoch, k, v)

    def _meters_to_dict(self, meters_list):
        tag_scalar_dict = {}
        for avg_meter in meters_list:
            value = avg_meter.get()
            if math.isnan(value) or avg_meter.count == 0:
                continue
            short_name = avg_meter.name if hasattr(avg_meter, 'name') else ''
            tag_scalar_dict[short_name] = value
        return tag_scalar_dict

    def _flush_category(self, epoch, category_name, metrics: Dict[str, List[AverageMeter]]):
        stdout = []
        for name, meters in metrics.items():
            # Multi plots not working with WANDB
            for i, meter in enumerate(meters):
                k = name
                if len(meters) > 1:
                    suffix = ['Generation', 'Reconstruction'][i]
                    k = f'{k}_{suffix}'

                main_tag = f'{category_name}/{k}'
                val = meter.get()
                if meter.count == 0 or math.isnan(val):
                    continue

                stdout.append(f'\t\t{k} = {val:.6f}')
                if type(self.sink) == SummaryWriter:
                    self.sink.add_scalar(main_tag, val, epoch)
                    self.sink.flush()

        self.sink.flush()
        if len(stdout) > 0:
            ProjectLogger().info(f'\t{category_name}:')
            ProjectLogger().info(''.join(stdout))

    @staticmethod
    def update_metrics(metrics, values_dict):
        for k, v in values_dict.items():
            assert k in metrics, f'No  {k} metric found.'
            if type(v) != list:
                v = [v]

            for i, val in enumerate(v):
                if val is None:
                    continue
                # If value is a tuple metric is expecting several inputs
                if type(val) == tuple:
                    metrics[k][i].update(*val)
                else:
                    metrics[k][i].update(val)

    def update_train(self, values):
        MetricStore.update_metrics(self.train, values)

    def update_validation(self, values):
        MetricStore.update_metrics(self.validation, values)

    def reset(self):
        [[m.reset() for m in v] for v in self.train.values()]
        [[m.reset() for m in v] for v in self.validation.values()]
