"""Loss curve plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(histories, out_path, title=None, logy=True, xlim=None,
                     inset_iters=None):
    """Overlay loss-vs-iteration curves.

    histories: list of (label, loss_array).
    inset_iters: if set, add a zoomed inset over the first N iterations.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, history in histories:
        ax.plot(range(1, len(history) + 1), history, label=label, linewidth=1.6)

    if inset_iters:
        axins = ax.inset_axes([0.45, 0.45, 0.5, 0.45])
        for label, history in histories:
            n = min(inset_iters, len(history))
            axins.plot(range(1, n + 1), history[:n], linewidth=1.4)
        if logy:
            axins.set_yscale("log")
        axins.set_title(f"first {inset_iters} iterations", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.grid(alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training MSE")
    if logy:
        ax.set_yscale("log")
    if xlim:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
