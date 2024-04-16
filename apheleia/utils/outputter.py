from pathlib import Path
from datetime import datetime
from apheleia.utils.patterns import Singleton

import shutil


class Outputter(metaclass=Singleton):
    def __init__(self, opts):
        if hasattr(opts, 'outdir'):
            current_time = datetime.now().strftime('%y_%m_%d-%H_%M_%S')

            if opts.action == 'train':
                model_name = opts.arch
                default_dir = Path.cwd() / f'training-{model_name}-{opts.epochs}e-{current_time}'
            else:
                default_dir = Path.cwd() / f'{opts.action}-{current_time}'

            self._outdir = default_dir if opts.outdir is None else Path(opts.outdir)
            self._outdir = self._outdir.expanduser()
            opts.outdir = self._outdir

            outdir_exists = self._outdir.is_dir()

            if outdir_exists and not opts.overwrite:
                if not hasattr(opts, 'resume') or not opts.resume:
                    raise Exception('Output directory already exists.')
            elif outdir_exists and opts.overwrite:
                if hasattr(opts, 'resume') and opts.resume:
                    raise Exception('Cannot overwrite with resume option.')
                shutil.rmtree(self._outdir)

            self._outdir.mkdir(parents=True, exist_ok=True)
