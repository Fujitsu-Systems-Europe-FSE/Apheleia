from pathlib import Path
from functools import partial
from types import SimpleNamespace
from apheleia.utils.logger import ProjectLogger
from apheleia.dataset.utilities import load_values
from apheleia.factory.model_factory import ModelFactory
from apheleia.utils.parsing import from_params_str, to_params_str
from apheleia.factory.training_pipeline_factory import TrainingPipelineFactory

import os
import wandb
import torch


def init_infer(args, ctx, model_name=None):
    try:
        model = torch.jit.load(Path(args.models).expanduser(), map_location=ctx[0])
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
        init_multiruns_train(args, ctx, train_data, val_data, test_data)
    elif args.sweep:
        args.wandb = True
        sweep_config = load_values(Path(args.sweep).expanduser())
        sweep_id = wandb.sweep(sweep_config, project=args.arch) if args.resume_sweep is None else args.resume_sweep
        wandb.agent(sweep_id, project=args.arch, function=partial(init_sweep_train, args, ctx, train_data, val_data, test_data), count=10)
    else:
        init_train(args, ctx, train_data, val_data, test_data)


def init_multiruns_train(opts, ctx, *args):
    if opts.sweep is not None:
        ProjectLogger().warning('--sweep cannot be executed with --runs option.')

    root_path = opts.outdir
    basename = os.path.basename(root_path)
    for i in range(1, opts.runs + 1):
        opts.outdir = os.path.join(root_path, f'{basename}-{i}')
        init_train(opts, ctx, *args)


def parse_composite_params(params):
    '''
    Used for optimizers and schedulers params
    :return: 
    '''
    if params is None:
        return {}

    assert len(params) == 1, 'Multi optimizers/schedulers not yet supported for sweep run'
    args, kwargs = from_params_str(params, 0)
    parsed_params = {e: True for e in args}
    parsed_params.update(kwargs)

    return parsed_params


def init_sweep_train(opts, ctx, *args):
    # wandb.init project arg is ignored in sweep mode and MUST be set at wandb.sweep level
    wandb.init()
    # Apply sweep hyperparams to namespace
    new_opts = opts.__dict__.copy()
    new_optimizer_params = parse_composite_params(opts.optimizers_params)
    new_schedule_params = parse_composite_params(opts.schedules_params)

    for k, v in dict(wandb.config).items():
        if k.startswith('optimizers-params-'):
            new_optimizer_params[k[18:]] = v
        elif k.startswith('schedulers-params-'):
            new_schedule_params[k[18:]] = v
        else:
            prev_value = new_opts[k]
            new_opts[k] = v
            ProjectLogger().warning(f'Sweep changing "{k}" hyperparam from {prev_value} to {v}')

    str_optim_params = to_params_str(new_optimizer_params)
    if opts.optimizers_params is not None and opts.optimizers_params[0] != str_optim_params:
        new_opts['optimizers_params'] = [str_optim_params]
        ProjectLogger().warning(f'Sweep changing optimizer hyperparams from {opts.optimizers_params[0]} to {str_optim_params}')

    str_sched_params = to_params_str(new_schedule_params)
    if opts.schedules_params is not None and opts.schedules_params[0] != str_sched_params:
        new_opts['schedules_params'] = [str_sched_params]
        ProjectLogger().warning(f'Sweep changing schedule hyperparams from {opts.schedules_params[0]} to {str_sched_params}')

    opts = SimpleNamespace(**new_opts)

    root_path = opts.outdir
    basename = os.path.basename(root_path)
    opts.outdir = os.path.join(root_path, f'{basename}-{wandb.run.name}')

    init_train(opts, ctx, *args)

def init_train(opts, ctx, *args):
    trainer = TrainingPipelineFactory(opts, ctx).build()

    try:
        trainer.start(*args)
    except KeyboardInterrupt as interrupt:
        ProjectLogger().warning('Keyboard interrupt received. Checkpointing current epoch.')
        trainer.do_interrupt_backup()
        raise interrupt
