from apheleia.factory import Factory
from torch_ema import ExponentialMovingAverage
from apheleia.utils.logger import ProjectLogger
from apheleia.utils.parsing import from_params_str
from apheleia.catalog import OptimizersCatalog, PipelinesCatalog, SchedulesCatalog

import itertools


class OptimizerFactory(Factory):

    def _init_ema(self, names, params):
        ema = {}
        model_pipeline = PipelinesCatalog()[self._namespace][self._model]
        if 'ema' in model_pipeline:
            with_ema = model_pipeline['ema']
            ema_decay = self._opts.ema_decay if hasattr(self._opts, 'ema_decay') and self._opts.ema_decay is not None else 0.999

            for i, enabled in enumerate(with_ema):
                if enabled:
                    ema[names[i]] = ExponentialMovingAverage(params[i], decay=ema_decay)

        return ema

    def _init_optimizers(self, names, params):
        optimizers = {}
        optims = PipelinesCatalog()[self._namespace][self._model]['optimizers']
        # CLI override
        if hasattr(self._opts, 'optimizers') and self._opts.optimizers is not None:
            optims = self._opts.optimizers

        # Still possible ? Could be removed probably
        if type(optims) != list:
            optims = [optims]

        assert len(optims) == len(names) or len(optims) == 1, 'Models MUST use individual optimizers or a common one.'
        if len(optims) == 1:
            params = [list(itertools.chain(*params))]

        for j, model_name in enumerate(names):
            reused = j >= len(optims)
            index = 0 if reused else j

            optim_name = optims[index]
            state = 'RE-USED' if reused else 'USED'
            ProjectLogger().info(f'{optim_name.upper()} will be {state} as optimizer with {names[j]}.')

            args, kwargs = from_params_str(self._opts.optimizers_params, index)
            clazz = OptimizersCatalog()[optim_name]
            try:
                optimizers[names[j]] = optimizers[list(optimizers.keys())[0]] if reused else clazz(params[index], *args, **kwargs)
            except Exception as e:
                ProjectLogger().error(f'{clazz.__name__} is expecting parameters. Refer to PyTorch doc and then use: --optimizers-params <params0>:<key1>=<params1>:<key2>=<params2>')
                raise e

        return optimizers

    def _init_schedulers(self, names, optimizers):
        schedules = {}
        if self._opts.lr_schedules is None:
            return None

        for i, lr_schedule in enumerate(self._opts.lr_schedules):
            ProjectLogger().info(f'{lr_schedule.upper()} will be used as scheduler with {names[i]}.')

            args, kwargs = from_params_str(self._opts.schedules_params, i)
            clazz = SchedulesCatalog()[lr_schedule]
            try:
                schedules[names[i]] = clazz(optimizers[i], *args, **kwargs)
            except Exception as e:
                ProjectLogger().error(f'{clazz.__name__} is expecting parameters. Refer to PyTorch doc and then use: --scheduler-params <params0>:<key1>=<params1>:<key2>=<params2>')
                raise e

        return schedules

    def _load_states(self, optims, ema):
        if self._save:
            for k, v in optims.items():
                key = f'{k}_optimizer_state'
                if key in self._save:
                    v.load_state_dict(self._save[key])
                else:
                    ProjectLogger().warning(f'"{k}" not found in checkpoint file')

            for k, v in ema.items():
                key = f'{k}_ema_state'
                if key in self._save:
                    v.load_state_dict(self._save[key])

    def build(self, nets):
        net_names = list(nets.keys())
        optimizers = self._init_optimizers(net_names, [v.parameters() for v in nets.values()])
        schedulers = self._init_schedulers(net_names, [v for v in optimizers.values()])
        ema = self._init_ema(net_names, [v.parameters() for v in nets.values()])

        if not self._opts.ignore_states:
            self._load_states(optimizers, ema)

        return optimizers, schedulers, ema
