from abc import ABC, abstractmethod
from apheleia.trainer.trainer import Trainer

import torch


class DLTrainer(Trainer, ABC):

    @abstractmethod
    def _iteration(self, batch, batch_idx, *args):
        pass

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
