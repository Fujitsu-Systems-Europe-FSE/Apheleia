from pathlib import Path
from typing import Iterable
from pickle import dump, load

import json
import torch
import numpy as np


class WelfordRunningStats:
    """
    Calculate dataset statistics iteratively using Welford algorithm
    """
    def __init__(self):
        self.ns = None
        self.means = None
        self.mins = None
        self.maxs = None
        self.M2s = None

    # def calc_mean_std(self, dataset: Iterable):
    #     for e in dataset:
    #         im = e
    #         y = None
    #         if type(e) == tuple:
    #             im, y = e
    #
    #         if type(im) == torch.Tensor:
    #             im = im.numpy()
    #         self.update(im)
    #     return self.means, self.std_devs()

    def _init_arrays(self, num_channels):
        if self.means is None:
            self.means = [0] * num_channels
            self.ns = [0] * num_channels
            self.M2s = [0] * num_channels
            self.mins = [0] * num_channels
            self.maxs = [0] * num_channels

    def update(self, image: np.ndarray | torch.Tensor):
        if type(image) != np.ndarray:
            image = image.numpy()

        if image.ndim > 2:
            self._init_arrays(image.shape[0])
            for i, c in enumerate(image):
                self._update_channel(i, c)
        else:
            self._init_arrays(1)
            self._update_channel(0, image)

    def _update_channel(self, channel_idx, image: np.ndarray):
        flat_image = image.flatten()
        count = flat_image.size
        delta = flat_image - self.means[channel_idx]
        self.means[channel_idx] += np.sum(delta) / (self.ns[channel_idx] + count)
        delta2 = flat_image - self.means[channel_idx]
        self.M2s[channel_idx] += np.sum(delta * delta2)
        self.ns[channel_idx] += count
        self.mins[channel_idx] = min(self.mins[channel_idx], flat_image.min())
        self.maxs[channel_idx] = max(self.maxs[channel_idx], flat_image.max())

    def variance(self, idx):
        if self.ns[idx] < 2:
            return float('nan')
        return self.M2s[idx] / (self.ns[idx] - 1)

    def std_devs(self):
        return [np.sqrt(self.variance(i)) for i in range(len(self.means))]

    def bounds(self):
        return [(mini, maxi) for mini, maxi in zip(self.mins, self.maxs)]


def load_values(values_file):
    values_file = Path(values_file).expanduser()
    with open(values_file, 'r') as f:
        values_dict = json.load(f)

    return values_dict


def save_values(outdir, values_dict):
    values_dump = Path(outdir).expanduser() / 'preprocessor_values.json'
    save_json(values_dump, values_dict)


def save_json(outfile, values):
    with open(outfile, 'w') as f:
        json.dump(values, f, indent=4,  skipkeys=True, default=lambda x: str(x))


def save_preprocessors(outdir, preprocessors):
    path = Path(outdir).expanduser() / 'preprocessor.pkl'
    dump(preprocessors, open(path, 'wb'))
    save_values(outdir, preprocessors.__dict__)


def load_preprocessors(preproc_file):
    path = Path(preproc_file).expanduser()
    return load(open(path, 'rb'))
