from typing import List

import numpy as np
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


def fig_as_numpy(fig):
    fig.canvas.draw()

    graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    graph = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return graph
