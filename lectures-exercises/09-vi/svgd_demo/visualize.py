import numpy as np
import matplotlib.pyplot as plt

def plot_particles(particles, ax=None, **kwargs):
    """Plot particles on given axis."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(particles[:, 0], particles[:, 1], **kwargs)
    return ax

def animate_particles(history, target_logp=None, xlim=(-4, 4), ylim=(-4, 4),
                      figsize=(6, 6), save_path=None):
    """Create particle evolution visualization."""
    iterations = sorted(history.keys())
    n_frames = len(iterations)
    n_cols = min(4, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes).flatten()

    if target_logp is not None:
        xx, yy = np.meshgrid(np.linspace(*xlim, 100), np.linspace(*ylim, 100))
        grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        zz = np.exp(target_logp(grid)).reshape(xx.shape)

    for idx, (ax, it) in enumerate(zip(axes, iterations)):
        if target_logp is not None:
            ax.contourf(xx, yy, zz, levels=20, alpha=0.3, cmap='viridis')
        ax.scatter(history[it][:, 0], history[it][:, 1], s=10, alpha=0.7)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'Iteration {it}')
        ax.set_aspect('equal')

    for ax in axes[len(iterations):]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
