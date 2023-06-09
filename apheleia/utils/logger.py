from apheleia.utils.patterns import Singleton

import os
import sys
import logging


class ProjectLogger(metaclass=Singleton):
    def __init__(self, *args):
        conf = {}
        self._name = os.path.basename(sys.argv[0])
        if len(args) > 0:
            opts, name = args
            self._name = name
            self._foreground = not opts.daemon if hasattr(opts, 'daemon') else True

            if not self._foreground:
                if hasattr(opts, 'outdir') and opts.outdir:
                    conf['filename'] = os.path.expanduser(os.path.join(opts.outdir, 'output.log'))
                else:
                    raise Exception('Cannot run in background. No output directory defined.')

        logging.basicConfig(**conf)
        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(logging.INFO)

    def error(self, msg):
        self._logger.log(logging.ERROR, msg)

    def warning(self, msg):
        self._logger.log(logging.WARNING, msg)

    def debug(self, msg):
        self._logger.log(logging.DEBUG, msg)

    def info(self, msg):
        self._logger.log(logging.INFO, msg)
