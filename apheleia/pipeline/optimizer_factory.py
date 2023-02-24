from SegOOD.pipeline import Factory, OPTIMIZERS, TRAINING_PIPELINES
from SegOOD.scheduler import LR_SCHEDULES
from SegOOD.utils.logger import ProjectLogger

import json


class OptimizerFactory(Factory):

    @staticmethod
    def auto_parse(val):
        # return float(val) if val.isdigit() else bool(val)
        try:
            return json.loads(val)
        except:
            return bool(val)

    @staticmethod
    def build_args(params):
        args = []
        kwargs = {}

        if params is not None:
            for p in params.split(':'):
                p = p.split('=')
                if len(p) == 2:
                    kwargs[p[0]] = OptimizerFactory.auto_parse(p[1])
                else:
                    args.append(OptimizerFactory.auto_parse(p[0]))

        return args, kwargs

    def _init_optimizer(self, params):
        recommended_optim_name = TRAINING_PIPELINES[self._family][self.model]['optimizer']
        optim_name = self._opts.optimizer if hasattr(self._opts, 'optimizer') and self._opts.optimizer is not None else recommended_optim_name
        ProjectLogger().info(f'{optim_name.upper()} will be used as optimizer.')

        args, kwargs = OptimizerFactory.build_args(self._opts.optimizer_params)
        kwargs['lr'] = self._opts.lr
        clazz = OPTIMIZERS[optim_name]
        try:
            optimizer = clazz(params, *args, **kwargs)
            return optimizer
        except Exception as e:
            ProjectLogger().error(f'{clazz.__name__} is expecting parameters. Refer to PyTorch doc and then use: --optimizer-params <params0>,<key1>=<params1>,<key2>=<params2>')
            raise e

        # optimizer = None
        # if hasattr(self._opts, 'b1') and hasattr(self._opts, 'b2') and hasattr(self._opts, 'lr') and hasattr(self._opts, 'eps'):
        #     adam_params = {
        #         'eps': self._opts.eps,
        #         'lr': self._opts.lr,
        #         'amsgrad': self._opts.amsgrad
        #     }
        #
        #     if self._opts.b1 and self._opts.b2:
        #         adam_params['betas'] = [self._opts.b1, self._opts.b2]
        #
        #     optimizer = optim.Adam(params, **adam_params)

    def _init_scheduler(self, optimizer):
        if self._opts.lr_schedule is None:
            return None

        ProjectLogger().info(f'{self._opts.lr_schedule.upper()} will be used as scheduler.')

        args, kwargs = OptimizerFactory.build_args(self._opts.schedule_params)
        clazz = LR_SCHEDULES[self._opts.lr_schedule]
        try:
            scheduler = clazz(optimizer, *args, **kwargs)
            return scheduler
        except Exception as e:
            ProjectLogger().error(f'{clazz.__name__} is expecting parameters. Refer to PyTorch doc and then use: --scheduler-params <params0>,<key1>=<params1>,<key2>=<params2>')
            raise e

    def _load_states(self, optims):
        if self._save:
            for k, v in optims.items():
                key = f'{k}_optimizer_state'
                if key in self._save:
                    v.load_state_dict(self._save[key])
                else:
                    ProjectLogger().warning(f'"{k}" not found in checkpoint file')

    def build(self, nets):
        optimizers = {k: self._init_optimizer(v.parameters()) for k, v in nets.items()}
        schedulers = {k: self._init_scheduler(v) for k, v in optimizers.items()}

        self._load_states(optimizers)

        return optimizers, schedulers
