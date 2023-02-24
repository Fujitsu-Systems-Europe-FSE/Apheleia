from torch import nn
from SegOOD.model import ModelStore
from SegOOD.model.common import NeuralNet
from SegOOD.pipeline import Factory, TRAINING_PIPELINES
from SegOOD.utils.logger import ProjectLogger


class ModelFactory(Factory):
    """
    Factory to build models

    Args:
        Factory: generic factory
    """
    def build(self):
        """
        Build the model specified in options

        Returns:
            Model: model built
        """
        # Models, optimizers and schedulers are stored in dict, because some models need their components to be handled separately (e.g. GANs, BiGANs, etc.)
        nets = ModelStore(self._opts)
        nets['model'] = self._init_model()

        self._parallelize_models(nets)
        self._load_weights(nets)

        return nets

    def _init_model(self):
        clazz = TRAINING_PIPELINES[self._family][self.model]['model']
        net = clazz(self._opts)
        assert isinstance(net, NeuralNet), f'{clazz.__class__} must inherit from NeuralNet class'
        net.check_structure()
        return net.to(self._ctx[0])

    def _parallelize_models(self, nets):
        if self._ctx[0].type != 'cpu':
            for k, v in nets.items():
                if hasattr(self._opts, 'distributed') and self._opts.distributed:
                    nets[k] = nn.parallel.DistributedDataParallel(v.to(self._ctx[0]), device_ids=self._ctx, find_unused_parameters=True)
                else:
                    nets[k] = nn.DataParallel(v, device_ids=self._ctx).to(self._ctx[0])
        else:
            ProjectLogger().warning('Models cannot be parallelized on cpu')

    def _load_weights(self, nets):
        if self._save:
            for k, v in nets.items():
                try:
                    v.load_state_dict(self.get_state_dict(f'{k}_state'), strict=False)
                except RuntimeError as e:
                    ProjectLogger().error(f'Cannot load weights properly for model {v.module._get_name()}')
                    raise e

    def get_state_dict(self, model_name):
        state_dict = self._save[model_name]
        if self._ctx[0].type == 'cpu':
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.split('module.')[-1]
                new_state_dict[new_key] = v

            state_dict = new_state_dict
        return state_dict

