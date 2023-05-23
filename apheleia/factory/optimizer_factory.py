from apheleia.factory import Factory
from apheleia.utils.logger import ProjectLogger
from apheleia.catalog import OptimizersCatalog, PipelinesCatalog, SchedulesCatalog

import json
import itertools


class OptimizerFactory(Factory):

    @staticmethod
    def auto_parse(val):
        # return float(val) if val.isdigit() else bool(val)
        try:
            return json.loads(val)
        except:
            return bool(val)

    @staticmethod
    def build_args(params_list, index):
        args = []
        kwargs = {}

        if params_list is not None:
            params = params_list[index]
            for p in params.split(':'):
                p = p.split('=')
                if len(p) == 2:
                    kwargs[p[0]] = OptimizerFactory.auto_parse(p[1])
                else:
                    args.append(OptimizerFactory.auto_parse(p[0]))

        return args, kwargs

    def _init_optimizers(self, names, params):
        optimizers = {}
        optims = PipelinesCatalog()[self._family][self._model]['optimizers']
        # CLI override
        if hasattr(self._opts, 'optimizers') and self._opts.optimizers is not None:
            optims = self._opts.optimizers

        if type(optims) != list:
            optims = [optims]

        assert len(optims) == len(names) or len(optims) == 1, 'Models MUST use individual optimizers or a common one.'
        if len(optims) == 1:
            params = [list(itertools.chain(*params))]

        # for i, recommended_optim_name in enumerate(optims):
        for j, model_name in enumerate(names):
            reused = j >= len(optims)
            index = 0 if reused else j

            optim_name = optims[index]
            state = 'RE-USED' if reused else 'USED'
            ProjectLogger().info(f'{optim_name.upper()} will be {state} as optimizer with {names[j]}.')

            args, kwargs = OptimizerFactory.build_args(self._opts.optimizers_params, index)
            clazz = OptimizersCatalog()[optim_name]
            try:
                optimizers[names[j]] = optimizers[list(optimizers.keys())[0]] if reused else clazz(params[index], *args, **kwargs)
            except Exception as e:
                ProjectLogger().error(f'{clazz.__name__} is expecting parameters. Refer to PyTorch doc and then use: --optimizer-params <params0>:<key1>=<params1>:<key2>=<params2>')
                raise e

        return optimizers

    def _init_schedulers(self, names, optimizers):
        schedules = {}
        if self._opts.lr_schedules is None:
            return None

        for i, lr_schedule in enumerate(self._opts.lr_schedules):
            ProjectLogger().info(f'{lr_schedule.upper()} will be used as scheduler with {names[i]}.')

            args, kwargs = OptimizerFactory.build_args(self._opts.schedules_params[i])
            clazz = SchedulesCatalog()[lr_schedule]
            try:
                schedules[names[i]] = clazz(optimizers[i], *args, **kwargs)
            except Exception as e:
                ProjectLogger().error(f'{clazz.__name__} is expecting parameters. Refer to PyTorch doc and then use: --scheduler-params <params0>:<key1>=<params1>:<key2>=<params2>')
                raise e

        return schedules

    def _load_states(self, optims):
        if self._save:
            for k, v in optims.items():
                key = f'{k}_optimizer_state'
                if key in self._save:
                    v.load_state_dict(self._save[key])
                else:
                    ProjectLogger().warning(f'"{k}" not found in checkpoint file')

    def build(self, nets):
        net_names = list(nets.keys())
        optimizers = self._init_optimizers(net_names, [v.parameters() for v in nets.values()])
        schedulers = self._init_schedulers(net_names, [v for v in optimizers.values()])
        # schedulers = {k: self._init_scheduler(v) for k, v in optimizers.items()}

        self._load_states(optimizers)

        return optimizers, schedulers
