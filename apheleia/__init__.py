from typing import Callable
from functools import partial
from datetime import timedelta
from daemonocle import Daemon
from apheleia.command_line_parser import CommandLineParser
from apheleia.utils.error_handling import handle_exception
from apheleia.utils.initialization import get_ctx, seed
from apheleia.utils.patterns import Singleton
from apheleia.utils.outputter import Outputter
from apheleia.utils.logger import ProjectLogger

import os
import torch.distributed as dist


class Project(metaclass=Singleton):

    def __init__(self, project_name):
        self.project_name = project_name
        self._bootstrap_hooks = {}

    def add_bootstrap(self, cli_action_name: str, fun: Callable):
        self._bootstrap_hooks[cli_action_name] = fun

    def mainloop(self, args):
        try:
            ctx = get_ctx(args)
            if args.action == 'train' and args.distributed:
                ProjectLogger().info('Waiting for entire world to start...')
                dist.init_process_group(backend='nccl', init_method=f'tcp://{args.master}', rank=args.rank, world_size=args.world_size, timeout=timedelta(minutes=5))
            self._bootstrap_hooks[args.action](args, ctx)
        except Exception as e:
            ProjectLogger().error('Fatal error occurred.')
            handle_exception(e, args.daemon if hasattr(args, 'daemon') else False)

    def run(self):
        try:
            cli = CommandLineParser()
            args = cli.parse()
            _ = Outputter(args)
            _ = ProjectLogger(args, self.project_name)

            if hasattr(args, 'daemon') and args.daemon:
                pid_file = os.path.join(os.path.sep, 'tmp', f'{self.project_name.lower()}.pid')
                if os.path.isfile(pid_file):
                    ProjectLogger().error('Daemon already running.')
                    exit(1)

                daemon = Daemon(worker=partial(self.mainloop, args), pid_file=pid_file)
                daemon.do_action('start')
            else:
                self.mainloop(args)
        except KeyboardInterrupt:
            ProjectLogger().warning('Stopped by user.')
