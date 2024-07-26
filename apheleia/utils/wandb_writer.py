from apheleia.utils.logger import ProjectLogger

import os
import wandb


class WandbWriter:

    def __init__(self, opts, tensorboard_logdir, enabled=True):
        self._opts = opts
        self._logdir = tensorboard_logdir
        self._project_name = opts.arch
        self._enabled = enabled

    def __enter__(self):
        if self._enabled:
            try:
                wandb.tensorboard.patch(root_logdir=self._logdir, pytorch=True)
            except Exception:
                ProjectLogger().warning('WandB tensorboard already patched.')

            # Initialize Wandb if it's not done already
            if wandb.run is None:
                wandb.init(project=self._project_name, name=os.path.basename(self._opts.outdir), sync_tensorboard=True)

            wandb.config.update(self._opts)
            return wandb

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._enabled:
            wandb.finish()
