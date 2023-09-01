from torch import nn
from apheleia.model import NeuralNet
from apheleia.factory import Factory
from apheleia.catalog import PipelinesCatalog
from apheleia.utils.logger import ProjectLogger
from apheleia.model.model_store import ModelStore


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
        nets.extend(self._init_models())

        self._parallelize_models(nets)
        self._load_weights(nets)

        return nets

    def _init_models(self):
        models = PipelinesCatalog()[self._namespace][self._model]['models']
        if type(models) != list:
            models = [models]

        nets = []
        for model_clazz in models:
            net = model_clazz(self._opts)
            assert isinstance(net, NeuralNet), f'{model_clazz.__class__} must inherit from NeuralNet class'
            net.check_structure()
            nets.append(net.to(self._ctx[0]))

        return nets

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

