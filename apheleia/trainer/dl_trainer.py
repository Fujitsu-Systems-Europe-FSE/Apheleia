from time import time
from abc import ABC, abstractmethod
from apheleia.trainer.trainer import Trainer
from apheleia.utils.logger import ProjectLogger

import torch


class DLTrainer(Trainer, ABC):

    @abstractmethod
    def _iteration(self, batch, batch_idx, *args):
        pass

    def _b_tick(self):
        self._batch_tick = time.time()

    def _b_duration(self):
        return time.time() - self._batch_tick

    def _log_iteration(self, batch_idx):
        if self.global_iter % self._log_interval == 0:
            b_time = self._b_duration()
            speed = self._opts.batch_size / b_time
            iter_stats = 'exec time: {:.2f} second(s) speed: {:.2f} samples/s'.format(b_time, speed)
            ProjectLogger().info('[Epoch {}] --[{}/{}]-- {}'.format(self.current_epoch, batch_idx, self.num_iter, iter_stats))

    def _report_graph(self):
        if self.current_epoch == 1 and self.writer is not ...:
            graph = self.get_graph()
            x: torch.Tensor = torch.zeros((1, self._opts.num_features, *self._opts.im_size)).to(self._ctx[0])
            self.writer.add_graph(graph, x)
            self.writer.flush()

    def _train_loop(self, train_data, *args, **kwargs):
        for batch_idx, batch in enumerate(train_data, start=1):
            self.global_iter += 1
            self._iteration(batch, batch_idx)
