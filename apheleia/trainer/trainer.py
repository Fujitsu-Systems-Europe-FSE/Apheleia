from apheleia.app import App
from abc import ABC, abstractmethod
from apheleia.utils.logger import ProjectLogger
from torch.utils.tensorboard import SummaryWriter
from apheleia.utils.wandb_writer import WandbWriter
from apheleia.metrics.metric_store import MetricStore
from apheleia.utils.error_handling import handle_exception
from codecarbon import EmissionsTracker, OfflineEmissionsTracker

import os
import glob
import json
import time
import torch
import torch.distributed as dist


class TrainerException(Exception):
    pass


class Trainer(ABC):

    def __init__(self, opts, net, optims, scheds, loss, validator, metrics: MetricStore, ctx, model_name, *args, **kwargs):
        self._opts = opts
        self._ctx = ctx
        self._net = net
        self._loss = loss
        self._model_name = model_name
        self._optimizer = optims
        self._scheduler = scheds
        self._validator = validator
        self._metrics_store = metrics

        self._epochs: int = opts.epochs
        self._start_epoch: int = opts.resume or 1
        if self._start_epoch > self._epochs:
            raise Exception(f'Training has already ended.')
        if opts.resume and not opts.models:
            raise Exception(f'Pre-trained model is required to resume training at epoch {opts.resume}')

        self._log_interval: int = opts.log_interval
        self._chkpt_interval: int = opts.chkpt_interval
        self._val_interval: int = opts.val_interval
        self._test_interval: int = opts.test_interval
        self._report_interval: int = opts.report_interval
        self._thumb_interval: int = opts.thumb_interval
        self._stats_interval: int = opts.stats_interval

        self._start_time: int = ...
        self._epoch_tick: int = ...
        self._batch_tick: int = ...

        self.writer: SummaryWriter = ...
        self.current_epoch: int = ...
        self.num_iter: int = ...
        self.global_iter: int = 0
        self.global_tick: int = 0

        self._outdir = self._opts.outdir
        self._outlogs = os.path.join(self._outdir, 'logs_{}'.format(self._model_name))
        self._outchkpts = os.path.join(self._outdir, 'checkpoints_{}'.format(self._model_name))
        self._prepare_outdir()

        self._scaler = torch.cuda.amp.GradScaler(enabled=opts.loss_scaling)

        # Measure Carbon footprint
        # self._tracker = OfflineEmissionsTracker(country_iso_code='FRA', output_dir=self._outdir)
        if self._opts.carbon_footprint:
            self._tracker = EmissionsTracker(project_name=os.path.basename(self._outdir), output_dir=self._outdir)
        else:
            self._tracker = None

    def _prepare_outdir(self):
        os.makedirs(self._outlogs, exist_ok=True)
        os.makedirs(self._outchkpts, exist_ok=True)

        params_dump = os.path.join(self._outdir, 'parameters_dump_{}.json'.format(self._model_name))
        with open(params_dump, 'w') as f:
            json.dump(vars(self._opts), f, indent=4, skipkeys=True, default=lambda x: str(x))

    def _b_tick(self):
        self._batch_tick = time.time()

    def _e_tick(self):
        self._epoch_tick = time.time()

    def _t_tick(self):
        self._start_time = time.time()

    def _b_duration(self):
        return time.time() - self._batch_tick

    def _e_duration(self):
        return time.time() - self._epoch_tick

    def _t_duration(self):
        return time.time() - self._start_time

    def start(self, train_data, val_data, test_data):
        self.num_iter = len(train_data)
        self._t_tick()

        for k in self._net.keys():
            ProjectLogger().info(self._net.get_raw(k))

        try:
            if not self._opts.distributed or dist.get_rank() == 0:
                with WandbWriter(self._opts, self._outlogs, enabled=self._opts.wandb):
                    with SummaryWriter(log_dir=self._outlogs, flush_secs=5) as self.writer:
                        self._metrics_store.sink = self.writer
                        self._train(train_data, val_data, test_data)
            else:
                self._train(train_data, val_data, test_data)
        except Exception as e:
            ProjectLogger().error('Fatal error occurred during training. Checkpointing current epoch.')
            self.do_interrupt_backup()
            handle_exception(e, self._opts.daemon, App().name)

    def _train(self, train_data, val_data, test_data):
        if self._tracker:
            self._tracker.start()

        for self.current_epoch in range(self._start_epoch, self._epochs + 1):
            if self._opts.distributed:
                # let all processes sync up before starting with a new epoch of training
                dist.barrier()

            self._e_tick()
            self._metrics_store.reset()

            self._report_graph()
            self._report_params()
            # self._start_profiler(epoch)

            if self._opts.target_metric is not None and self._metrics_store.best_tgt_metric >= self._opts.target_metric:
                ProjectLogger().info('Target metric has been reached.')
                break

            self._train_loop(train_data)

            self._apply_schedules()
            if not self._opts.distributed or dist.get_rank() == 0:
                if self._validator is not None:
                    if self._val_interval is not None and (self.current_epoch % self._val_interval == 0):
                        self._validator.evaluate(val_data, 'Validation')

                    if self._test_interval is not None and (self.current_epoch % self._test_interval == 0):
                        self._validator.evaluate(test_data, 'Test')

            self._log_epoch()
            if not self._opts.distributed or dist.get_rank() == 0:
                self._try_checkpoint()

        if not self._opts.distributed or dist.get_rank() == 0:
            ProjectLogger().info('Checkpointing model...')
            self._do_checkpoint()
            ProjectLogger().info('Exporting model...')
            self._export_model()
        ProjectLogger().info('Training complete. Done in {:.2f}s.'.format(self._t_duration()))

        if self._tracker:
            emissions = self._tracker.stop()

    @abstractmethod
    def _train_loop(self, *args, **kwargs):
        pass

    def _apply_schedules(self):
        if self._opts.lr_schedules is not None:
            for scheduler_name, scheduler in self._scheduler.items():
                scheduler.step()
                if type(self.writer) == SummaryWriter:
                    self.writer.add_scalar(f'{scheduler_name}-learning_rate', self._optimizer[scheduler_name].param_groups[0]['lr'], self.current_epoch - 1)

    def _export_model(self):
        try:
            networks = self._net.get_all_raw()
            for net in networks:
                net_jit = torch.jit.script(net)
                net_jit.save(os.path.join(self._outdir, '{}-{:04d}.pt'.format(f'{self._opts.arch}-{net.model_name()}', self.current_epoch)))
        except Exception as e:
            ProjectLogger().error(f'Model export failed : {e}')

    def do_interrupt_backup(self):
        out_filename = os.path.join(self._outchkpts, '{}-{:04d}.bak'.format(self._opts.arch, self.current_epoch))
        self._save_checkpoints(out_filename)

    def _do_checkpoint(self):
        out_filename = os.path.join(self._outchkpts, '{}-{:04d}'.format(self._opts.arch, self.current_epoch))
        self._save_checkpoints(out_filename)

    def _save_checkpoints(self, out_filename):
        save_dict = dict(epoch=self.current_epoch)
        for netname in self._net.keys():
            save_dict[f'{netname}_state'] = self._net[netname].state_dict()
            save_dict[f'{netname}_optimizer_state'] = self._optimizer[netname].state_dict()

        torch.save(save_dict, '{}.chpth'.format(out_filename))

    def _clean_checkpoints(self):
        chkpt_files = glob.glob(os.path.join(self._outchkpts, '*.chpth'))
        self._clean_files(chkpt_files)

    def _clean_files(self, files):
        if len(files) >= self._opts.max_chkpts:
            files.sort(key=os.path.getmtime)
            files_to_drop = files[:-self._opts.max_chkpts]
            _ = [os.remove(f) for f in files_to_drop]

    def _try_checkpoint(self):
        if self._metrics_store is None:
            return

        target = self._metrics_store.target
        value = target.get()
        if type(value) == dict:
            value = value[target.target()]

        if self._chkpt_interval is not None and (self.current_epoch % self._chkpt_interval == 0):
            self._do_checkpoint()
            self._clean_checkpoints()
        elif target.count > 0 and value > self._metrics_store.best_tgt_metric:
            self._metrics_store.best_tgt_metric = value
            ProjectLogger().info('Target metric has improved ! Checkpointing.')
            self._do_checkpoint()
            self._clean_checkpoints()

    def _log_epoch(self):
        ProjectLogger().info('[Epoch {}] exec time: {:.2f}'.format(self.current_epoch, self._e_duration()))
        self._metrics_store.flush(self.current_epoch)

    def _log_iteration(self, batch_idx):
        if self.global_iter % self._log_interval == 0:
            b_time = self._b_duration()
            speed = self._opts.batch_size / b_time
            iter_stats = 'exec time: {:.2f} second(s) speed: {:.2f} samples/s'.format(b_time, speed)
            ProjectLogger().info('[Epoch {}] --[{}/{}]-- {}'.format(self.current_epoch, batch_idx, self.num_iter, iter_stats))

    @abstractmethod
    def _report_graph(self):
        pass

    def _report_params(self):
        if self._report_interval > 0 and self.current_epoch % self._report_interval == 0:
            if self.writer is not ...:
                for k, v in self._net.items():
                    names, weights, grads = self.get_params(k)

                    if len(names) > 0:
                        for name, weight, grad in zip(names, weights, grads):
                            model_name = self._net.get_raw(k).model_name()
                            param_name = '.'.join(name.split('.')[1:])

                            self.writer.add_histogram(f'{model_name}_weights_biases/{param_name}', weight, self.current_epoch, max_bins=512)
                            if grad is not None and not torch.isnan(grad.sum()).item() and not torch.isinf(grad.sum()).item():
                                self.writer.add_histogram(f'{model_name}_grads/{param_name}', grad, self.current_epoch, max_bins=512)

                self.writer.flush()

    @abstractmethod
    def _report_stats(self, *args):
        pass

    @abstractmethod
    def get_graph(self) -> torch.nn.Module:
        pass

    def get_params(self, netname):
        names, weights, grads = [], [], []
        for k, v in self._net[netname].named_parameters():
            names.append(k)
            weights.append(v)
            grads.append(v.grad)

        return names, weights, grads
