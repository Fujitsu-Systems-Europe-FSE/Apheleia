from datetime import datetime
from apheleia.utils.patterns import Singleton

import os
import shutil


class Outputter(metaclass=Singleton):
    def __init__(self, opts):
        if hasattr(opts, 'outdir'):
            current_time = datetime.now().strftime('%y_%m_%d-%H_%M_%S')

            if opts.action == 'train':
                model_name = opts.arch
                default_dir = os.path.join(os.getcwd(), f'training-{model_name}-{opts.epochs}e-{current_time}')
            else:
                default_dir = os.path.join(os.getcwd(), f'{opts.action}-{current_time}')

            self._outdir = opts.outdir or default_dir
            self._outdir = os.path.expanduser(self._outdir)
            opts.outdir = self._outdir

            outdir_exists = os.path.isdir(self._outdir)

            if outdir_exists and not opts.overwrite:
                if not hasattr(opts, 'resume') or not opts.resume:
                    raise Exception('Output directory already exists.')
            elif outdir_exists and opts.overwrite:
                if hasattr(opts, 'resume') and opts.resume:
                    raise Exception('Cannot overwrite with resume option.')
                shutil.rmtree(self._outdir)

            os.makedirs(self._outdir, exist_ok=True)
