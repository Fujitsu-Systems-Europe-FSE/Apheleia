from apheleia.factory import Factory
from apheleia.factory.model_factory import ModelFactory
from apheleia.catalog import PipelinesCatalog, LossesCatalog
from apheleia.factory.optimizer_factory import OptimizerFactory


class TrainingPipelineFactory(Factory):
    """
    Trainer factory build trainer with its dependencies

    Args:
        Factory: generic factory
    """

    def __init__(self, opts, ctx):
        super(TrainingPipelineFactory, self).__init__(opts, ctx)

        self._model_factory = ModelFactory(self._opts, self._ctx)
        self._optim_factory = OptimizerFactory(self._opts, self._ctx)

        if hasattr(self._opts, 'resume') and self._opts.resume:
            self._opts.resume = self._save['epoch'] + 1

    def _init_loss(self):
        losses = PipelinesCatalog()[self._family][self._model]['losses']
        assert len(losses), f'No losses defined for current arch {self._family} -> {self._model}'
        loss_name = self._opts.loss if hasattr(self._opts, 'loss') and self._opts.loss is not None else losses[0]
        selected_loss = LossesCatalog()[self._family][loss_name]
        loss = selected_loss['class']
        assert loss_name in PipelinesCatalog()[self._family][self._model]['losses'], f'Invalid loss for current arch {self._family} -> {self._model}'

        if 'type' in selected_loss and hasattr(self._opts, 'dataset_type'):
            loss_type = selected_loss['type']
            assert loss_type == self._opts.dataset_type, f'Invalid loss for current dataset {self._opts.dataset_class} -> {self._opts.dataset_type}'

        return loss(self._opts)

    def _init_metrics(self, loss):
        metrics = PipelinesCatalog()[self._family][self._model]['metrics']
        return metrics(self._opts, loss)

    def _init_validator(self, net, metrics):
        validator = PipelinesCatalog()[self._family][self._model]['validator']
        return validator(self._opts, net, metrics, self._ctx)

    def _init_trainer(self, nets, optimizers, schedulers, loss, validator, metrics):
        trainer = PipelinesCatalog()[self._family][self._model]['trainer']
        return trainer(self._opts, nets, optimizers, schedulers, loss, validator, metrics, self._ctx)

    def build(self, with_loss=True):
        loss = self._init_loss() if with_loss else None
        metrics = self._init_metrics(loss)

        nets = self._model_factory.build()
        validator = self._init_validator(nets, metrics) if 'validator' in PipelinesCatalog()[self._family][self._model] else None
        optimizers, schedulers = self._optim_factory.build(nets)

        trainer = self._init_trainer(nets, optimizers, schedulers, loss, validator, metrics)

        return trainer
