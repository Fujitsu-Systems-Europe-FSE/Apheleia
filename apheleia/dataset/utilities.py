from pathlib import Path
from typing import Iterable

import json
import torch
import numpy as np


class WelfordRunningStats:
    """
    Calculate dataset statistics iteratively using Welford algorithm
    """
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def calc_mean_std(self, dataset: Iterable):
        for e in dataset:
            im = e
            y = None
            if type(e) == tuple:
                im, y = e

            if type(im) == torch.Tensor:
                im = im.numpy()
            self.update(im)
        return self.mean, self.std_dev()

    def update(self, image: np.ndarray):
        flat_image = image.flatten()
        count = flat_image.size
        delta = flat_image - self.mean
        self.mean += np.sum(delta) / (self.n + count)
        delta2 = flat_image - self.mean
        self.M2 += np.sum(delta * delta2)
        self.n += count

    def variance(self):
        if self.n < 2:
            return float('nan')
        return self.M2 / (self.n - 1)

    def std_dev(self):
        return np.sqrt(self.variance())


def load_values(values_file):
    values_file = Path(values_file).expanduser()
    with open(values_file, 'r') as f:
        values_dict = json.load(f)

    return values_dict


def save_values(outdir, values_dict):
    values_dump = Path(outdir).expanduser() / 'dataset_values.json'
    save_json(values_dump, values_dict)


def save_json(outfile, values):
    with open(outfile, 'w') as f:
        json.dump(values, f, indent=4,  skipkeys=True, default=lambda x: str(x))
