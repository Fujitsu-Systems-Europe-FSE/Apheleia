from apheleia.utils.logger import ProjectLogger
from apheleia.factory.model_factory import ModelFactory
from apheleia.factory.training_pipeline_factory import TrainingPipelineFactory

import os
import torch


def init_infer(args, ctx, model_name=None):
    try:
        model = torch.jit.load(args.model, map_location=ctx[0])
        model.eval()
        return model
    except Exception:
        if args.arch is None:
            raise Exception('Model architecture must be given when loading a checkpoint. Use --arch option.')

        model_factory = ModelFactory(args, ctx)
        models = model_factory.build()
        models.eval()

        if model_name is not None:
            return models[model_name]

        return models


def train(args, ctx, setup_env):
    train_data, val_data, test_data = setup_env(args)
    if args.runs > 1:
        root_path = args.outdir
        basename = os.path.basename(root_path)
        for i in range(1, args.runs + 1):
            args.outdir = os.path.join(root_path, f'{basename}-{i}')
            init_train(args, ctx, train_data, val_data, test_data)
    else:
        init_train(args, ctx, train_data, val_data, test_data)


def init_train(opts, ctx, *args):
    trainer = TrainingPipelineFactory(opts, ctx).build()

    try:
        trainer.start(*args)
    except KeyboardInterrupt as interrupt:
        ProjectLogger().warning('Keyboard interrupt received. Checkpointing current epoch.')
        trainer.do_interrupt_backup()
        raise interrupt
