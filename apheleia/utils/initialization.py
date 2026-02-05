from apheleia.utils.logger import ProjectLogger

import torch
import random
import numpy as np


def get_ctx(args):
    devices_id = [int(i) for i in args.gpus.split(',') if i.strip()]
    if torch.cuda.is_available():
        if len(devices_id) == 0:
            devices_id = list(range(torch.cuda.device_count()))

        ctx = [torch.device(f'cuda:{i}') for i in devices_id if i >= 0]
        ctx = ctx if len(ctx) > 0 else [torch.device('cpu')]
    else:
        ProjectLogger().error('Cannot access GPU.')
        ctx = [torch.device('cpu')]

    ProjectLogger().info('Used context: {}'.format(', '.join([str(x) for x in ctx])))
    return ctx


def seed(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        # we are just setting cudnn to deterministic, cannot set whole PyTorch engine (e.g. cros_entropy is not deterministic)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        args.seed = torch.initial_seed()

    random.seed(args.seed)
    # torch 2^64 seed is too high for numpy 2^32 - 1. So we are folding seed in a deterministic way
    MAX_SEED = 2 ** 32 - 1
    safe_seed = int(args.seed) % MAX_SEED
    np.random.seed(safe_seed)

    ProjectLogger().info(f'torch seed initialized to {args.seed}')
