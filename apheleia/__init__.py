from apheleia.utils.logger import ProjectLogger
from apheleia.factory.training_pipeline_factory import TrainingPipelineFactory

import os


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
