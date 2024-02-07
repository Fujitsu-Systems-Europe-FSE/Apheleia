from time import time
from tqdm import tqdm
from abc import ABC, abstractmethod
from apheleia.trainer.trainer import Trainer
from apheleia.utils.logger import ProjectLogger

import torch.distributed as dist


class DLTrainer(Trainer, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pbar = tqdm()

    @abstractmethod
    def _iteration(self, batch, batch_idx, *args):
        pass

    def _b_tick(self):
        self._batch_tick = time()

    def _b_duration(self):
        return time() - self._batch_tick

    def _log_iteration(self, batch_idx):
        if self.global_iter % self._log_interval == 0:
            if self._opts.daemon:
                b_time = self._b_duration()
                speed = self._opts.batch_size / b_time
                iter_stats = 'exec time: {:.2f} second(s) speed: {:.2f} samples/s'.format(b_time, speed)
                ProjectLogger().info('[Epoch {}] --[{}/{}]-- {}'.format(self.current_epoch, batch_idx, self.num_iter, iter_stats))
            else:
                self._pbar.update(self._log_interval)
                self._pbar.set_description(f'[Epoch {self.current_epoch}]')
                loss = sum([v[0].get().item() for v in self._metrics_store.train.values()])
                self._pbar.set_postfix_str(f'loss -> {loss:.4f}')

    def _pre_loop_hook(self, *args):
        self._pbar.reset(self.num_iter)

    def _train_loop(self, train_data, *args, **kwargs):
        for batch_idx, batch in enumerate(train_data, start=1):
            self.global_iter += 1
            self._iteration(batch, batch_idx)

    def _post_loop_hook(self, val_data, test_data, *args):
        if not self._opts.distributed or dist.get_rank() == 0:
            if self._validator is not None:
                if self._val_interval is not None and (self.current_epoch % self._val_interval == 0):
                    self._validator.evaluate(val_data, 'Validation')

                if self._test_interval is not None and (self.current_epoch % self._test_interval == 0):
                    self._validator.evaluate(test_data, 'Test')

        # When training is interactive insert newline after tqdm pbar
        if not self._opts.daemon:
            print('')

        self._log_epoch()
        self._try_checkpoint()
