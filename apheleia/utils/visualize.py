from typing import List

import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def gradients_norm_hist(writer, title, grads_norm: List[np.ndarray], epoch, n_bins=100, labels=None):
    if labels is not None:
        assert len(labels) == len(grads_norm), 'Number of labels mismatch number of gradients norm'

    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    fig.suptitle('Gradients norm histogram')

    color_table = list(mcolors.TABLEAU_COLORS.keys())

    for i in range(len(grads_norm)):
        label = labels if labels is None else labels[i]
        ax.hist(grads_norm[i], bins=n_bins, color=color_table[i % len(grads_norm)], alpha=.5, label=label)

    ax.set_xlabel('norm value')
    ax.set_ylabel('count')
    ax.legend()

    writer.add_figure(f'gradients/{title}', fig, global_step=epoch)
    writer.flush()

def tensors_to_densities(writer, tag, tensors, epoch, channels_names, tensors_name=None, feats_subset_size=-1):
    tensors = prepare_tensors(tensors)
    tensors, channels_names = reduce_feats(feats_subset_size, tensors, channels_names)

    ncols = min(4, len(channels_names))
    nrows = max(1, math.ceil(len(channels_names) / 4))

    def plot_ax(ax, idx, tensors, tensors_name, channels_names):
        ax.axis('off')
        if idx < len(channels_names):
            for j, t in enumerate(tensors):
                plot_opts = dict(fill=True, ax=ax, alpha=(.4 if j > 0 else .7))
                if tensors_name is not None:
                    plot_opts['label'] = tensors_name[j]
                sns.kdeplot(t[:, idx], **plot_opts)

            ax.set_title(channels_names[idx])
            ax.set_xscale('symlog')
            ax.set_yscale('symlog')

    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 7.2))
    try:
        _ = [plot_ax(ax, i, tensors, tensors_name, channels_names) for i, ax in enumerate(axes.flat)]
        handles, labels = axes.flat[0].get_legend_handles_labels()
    except AttributeError as _:
        plot_ax(axes, 0, tensors, tensors_name, channels_names)
        handles, labels = axes.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center')
    fig.tight_layout()

    graph = fig_as_numpy(fig)
    writer.add_image(tag, graph.transpose((2, 0, 1)), epoch)
    writer.flush()

def tensors_to_hist1d(writer, tensors, epoch, channels_names, tensors_name=None, feats_subset_size=-1, tag='dist/histograms'):
    tensors = prepare_tensors(tensors)
    tensors, channels_names = reduce_feats(feats_subset_size, tensors, channels_names)

    ncols = min(4, len(channels_names))
    nrows = max(1, math.ceil(len(channels_names) / 4))

    def hist_ax(ax, idx, tensors, tensors_name, channels_names):
        # ax.axis('off')
        if idx < len(channels_names):
            # dists = [t[:, i] for t in tensors]
            for j, t in enumerate(tensors):
                plot_opts = dict(bins=100, density=True, alpha=(.4 if j > 0 else .7))
                if tensors_name is not None:
                    plot_opts['label'] = tensors_name[j]
                ax.hist(t[:, idx], **plot_opts)
            ax.set_title(channels_names[idx])

    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 7.2))
    try:
        _ = [hist_ax(ax, i, tensors, tensors_name, channels_names) for i, ax in enumerate(axes.flat)]
        handles, labels = axes.flat[0].get_legend_handles_labels()
    except AttributeError as _:
        hist_ax(axes, 0, tensors, tensors_name, channels_names)
        handles, labels = axes.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center')
    fig.tight_layout()
    graph = fig_as_numpy(fig)
    writer.add_image(tag, graph.transpose((2, 0, 1)), epoch)
    writer.flush()

def prepare_tensors(tensors):
    tensors = list(tensors)
    for i, t in enumerate(tensors):
        tensors[i] = t.squeeze() if t.ndim == 3 else t
        if isinstance(tensors[i], torch.Tensor):
            tensors[i] = tensors[i].cpu().numpy()
    return tensors

def reduce_feats(subset_size, tensors, channels_names):
    if subset_size > -1:
        n_feats = min(subset_size, len(channels_names))
        selected_feats = np.random.choice(range(len(channels_names)), n_feats, replace=False)

        channels_names = np.take(channels_names, selected_feats, axis=0).tolist()
        tensors = [np.take(t, selected_feats, axis=-1) for t in tensors]

    return tensors, channels_names

def fig_as_numpy(fig):
    fig.canvas.draw()

    graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    graph = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return graph
