from pathlib import Path
from types import SimpleNamespace


import json


def save_parameters_dump(opts: SimpleNamespace, outdir: str | Path, suffix=''):
    suffix = f'_{suffix}' if suffix != '' else suffix
    params_dump = Path(outdir) / f'parameters_dump{suffix}.json'
    with open(params_dump, 'w') as f:
        json.dump(vars(opts), f, indent=4, skipkeys=True, default=lambda x: str(x))


def load_parameters_dump(path: str | Path):
    dump_file = Path(path).expanduser()
    with open(dump_file, 'r') as f:
        params = json.load(f)

    return SimpleNamespace(**params)