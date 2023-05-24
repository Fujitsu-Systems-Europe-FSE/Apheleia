from apheleia import train
from typing import Callable
from functools import partial
from daemonocle import Daemon
from datetime import timedelta
from apheleia.utils.patterns import Singleton
from apheleia.utils.outputter import Outputter
from apheleia.utils.initialization import seed
from apheleia.utils.logger import ProjectLogger
from apheleia.utils.initialization import get_ctx
from apheleia.command_line_parser import CommandLineParser
from apheleia.utils.error_handling import handle_exception

import os
import torch.distributed as dist


class App(metaclass=Singleton):

    def __init__(self, name, with_dataset=True):
        self.name = name
        self.cli = CommandLineParser(with_dataset)
        self._bootstrap_hooks = {}

    def add_bootstrap(self, cli_action_name: str, fun: Callable):
        if cli_action_name in self.cli._subparsers:
            if cli_action_name == 'train':
                fun = partial(train, setup_env=fun)
            self._bootstrap_hooks[cli_action_name] = fun
        else:
            ProjectLogger().error(f'{cli_action_name} not found in cli subparsers. Cannot add hook.')

    def _mainloop(self, args):
        try:
            ctx = get_ctx(args)
            if args.action == 'train' and args.distributed:
                ProjectLogger().info('Waiting for entire world to start...')
                dist.init_process_group(backend='nccl', init_method=f'tcp://{args.master}', rank=args.rank, world_size=args.world_size, timeout=timedelta(minutes=5))
            seed(args)
            self._bootstrap_hooks[args.action](args, ctx)
        except Exception as e:
            ProjectLogger().error('Fatal error occurred.')
            handle_exception(e, args.daemon if hasattr(args, 'daemon') else False, self.name)

    def run(self):
        try:
            args = self.cli.parse()
            _ = Outputter(args)
            _ = ProjectLogger(args, self.name)

            if hasattr(args, 'daemon') and args.daemon:
                pid_file = os.path.join(os.path.sep, 'tmp', f'{self.name.lower()}.pid')
                if os.path.isfile(pid_file):
                    ProjectLogger().error('Daemon already running.')
                    exit(1)

                daemon = Daemon(worker=partial(self._mainloop, args), pid_file=pid_file)
                daemon.do_action('start')
            else:
                self._mainloop(args)
        except KeyboardInterrupt:
            ProjectLogger().warning('Stopped by user.')
