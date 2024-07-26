from tqdm import tqdm
from apheleia.app import App
from abc import ABC, abstractmethod
from apheleia.utils.logger import ProjectLogger
from torch.utils.tensorboard import SummaryWriter
from apheleia.utils.wandb_writer import WandbWriter
from apheleia.metrics.metric_store import MetricStore
from apheleia.utils.error_handling import handle_exception
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from apheleia.utils.metadata import save_parameters_dump

import os
import glob
import time
import torch
import torch.distributed as dist


class TrainerException(Exception):
    pass


class Trainer(ABC):

    def __init__(self, model_name, opts, net, optims, scheds, ema, loss, validator, metrics: MetricStore, ctx):
        self._opts = opts
        self._ctx = ctx
        self._net = net
        self._loss = loss
        self._model_name = model_name
        self._optimizer = optims
        self._scheduler = scheds
        self._ema = ema
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
        # self.global_tick: int = 0

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
        save_parameters_dump(self._opts, self._outdir, suffix=self._model_name)

    def _e_tick(self):
        self._epoch_tick = time.time()

    def _t_tick(self):
        self._start_time = time.time()

    def _e_duration(self):
        return time.time() - self._epoch_tick

    def _t_duration(self):
        return time.time() - self._start_time

    def start(self, train_data, val_data, test_data):
        self.num_iter = len(train_data)
        self._t_tick()

        for k in self._net.keys():
            n_params = sum([p.numel() for p in self._net[k].parameters() if p.requires_grad])
            ProjectLogger().info(f'{k} model size : {n_params:,} parameters')
            ProjectLogger().info(self._net.get_raw(k))

        if self._opts.num_accum > 1:
            info = f'{self._opts.batch_size} x {self._opts.num_accum} = {self._opts.batch_size * self._opts.num_accum}'
            ProjectLogger().info(f'Gradient accumulation enabled. Effective batch size -> {info}')

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

        self._report_graph()

        self._pbar = tqdm()
        for self.current_epoch in range(self._start_epoch, self._epochs + 1):
            if self._opts.distributed:
                # let all processes sync up before starting with a new epoch of training
                dist.barrier()

            self._e_tick()
            self._metrics_store.reset()

            # self._start_profiler(epoch)

            if self._opts.target_metric is not None and self._metrics_store.best_tgt_metric >= self._opts.target_metric:
                ProjectLogger().info('Target metric has been reached.')
                break

            self._pre_loop_hook(train_data)
            self._train_loop(train_data)
            self._post_loop_hook(val_data, test_data)

        if not self._opts.distributed or dist.get_rank() == 0:
            ProjectLogger().info('Checkpointing model...')
            self._do_checkpoint()
            ProjectLogger().info('Exporting model...')
            self._export_model()
        ProjectLogger().info('Training complete. Done in {:.2f}s.'.format(self._t_duration()))

        if self._tracker:
            emissions = self._tracker.stop()

    def step(self, netname, loss, strict_mode=True):
        self._scaler.scale(loss.div(self._opts.num_accum)).backward()
        if self.global_iter % self._opts.num_accum == 0:
            # clip gradients if needed
            params = self._net[netname].parameters()

            if strict_mode:
                no_grad_layers = [n for n, p in self._net[netname].named_parameters() if p.grad is None]
                if len(no_grad_layers) > 0:
                    ProjectLogger().warning(f'The following layers don\'t have grads : {no_grad_layers}')

            if self._opts.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, self._opts.clip_grad_norm)
            if self._opts.clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(params, self._opts.clip_grad_value)

            # do an optimizer step, rescale FP16 to FP32 if needed
            self._scaler.step(self._optimizer[netname])
            self._scaler.update()

            # Report weights and grads before grads got wiped
            self._report_params()

            # clear accumulated gradients
            self._optimizer[netname].zero_grad(set_to_none=True)
            # update scheduler
            self._apply_schedule(netname)

            # update the moving average with the new parameters from the last optimizer step
            if netname in self._ema:
                self._ema[netname].update()

    @abstractmethod
    def _train_loop(self, *args, **kwargs):
        pass

    def _pre_loop_hook(self, *args):
        pass

    def _post_loop_hook(self, *args):
        pass

    def _apply_schedule(self, scheduler_name):
        if self._scheduler is not None and scheduler_name in self._scheduler:
            scheduler = self._scheduler[scheduler_name]
            scheduler.step()
            if type(self.writer) == SummaryWriter:
                self.writer.add_scalar(f'{scheduler_name}-learning_rate', self._optimizer[scheduler_name].param_groups[0]['lr'], self.global_iter - 1)

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
            if netname in self._ema:
                save_dict[f'{netname}_ema_state'] = self._ema[netname].state_dict()
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
        if (self._opts.distributed and dist.get_rank() > 0) or self._metrics_store is None:
            return

        target = self._metrics_store.target
        if target is None:
            return

        value = target.get()
        if type(value) == dict:
            value = value[target.target()]

        if self._chkpt_interval is not None and (self.current_epoch % self._chkpt_interval == 0):
            self._do_checkpoint()
            self._clean_checkpoints()
        elif (target.count > 0 and (self._metrics_store.best_tgt_metric is None or
              ((target.expected_behavior == 'increasing' and value > self._metrics_store.best_tgt_metric) or
              (target.expected_behavior == 'decreasing' and value < self._metrics_store.best_tgt_metric)))):
            self._metrics_store.best_tgt_metric = value
            ProjectLogger().info('Target metric has improved ! Checkpointing.')
            self._do_checkpoint()
            self._clean_checkpoints()

    def _log_epoch(self):
        ProjectLogger().info('[Epoch {}] exec time: {:.2f}'.format(self.current_epoch, self._e_duration()))
        self._metrics_store.flush(self.current_epoch)

    @staticmethod
    def _log_iteration(self, *args):
        pass

    def _report_graph(self):
        if self.writer is not ...:
            graph = self.get_graph()
            inputs = self.get_dummmy_inputs()

            if graph is None or inputs is None:
                return

            # tensorboard needs graph to be on cpu
            graph.to(torch.device('cpu'))
            self.writer.add_graph(graph, inputs)
            graph.to(self._ctx[0])
            self.writer.flush()

    def _report_params(self):
        if self._report_interval > 0 and self.current_epoch % self._report_interval == 0\
                and self.global_iter % self.num_iter == 0:
            if self.writer is not ...:
                for k, v in self._net.items():
                    names, weights, grads = self.get_params(k)

                    if len(names) > 0:
                        for name, weight, grad in zip(names, weights, grads):
                            model_name = self._net.get_raw(k).model_name()
                            param_name = '.'.join(name.split('.')[1:])

                            if weight is not None and not torch.isnan(weight.sum()).item() and not torch.isinf(weight.sum()).item():
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

    @abstractmethod
    def get_dummmy_inputs(self) -> list[torch.Tensor]:
        pass

    def get_params(self, netname):
        names, weights, grads = [], [], []
        for k, v in self._net[netname].named_parameters():
            names.append(k)
            weights.append(v)
            grads.append(v.grad)

        return names, weights, grads
