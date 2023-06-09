from apheleia.utils.logger import ProjectLogger

import os
import traceback


def handle_exception(e, daemon, app_name):
    if daemon:
        traceback_dir = '/tmp/log'
        os.makedirs(traceback_dir, exist_ok=True)
        traceback_file = os.path.join(traceback_dir, f'{app_name}.log')
        ProjectLogger().error(f'Traceback saved at {traceback_file}')
        with open(traceback_file, 'w') as f:
            f.write(traceback.format_exc())
    else:
        raise e
