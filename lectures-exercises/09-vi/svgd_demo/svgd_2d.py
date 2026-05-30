import numpy as np


def median_bandwidth(particles):
    """Compute bandwidth using the median heuristic."""
    n = particles.shape[0]
    pairwise_dists = np.sum((particles[:, None, :] - particles[None, :, :]) ** 2, axis=-1)
    h = np.median(pairwise_dists) / np.log(n + 1)
    return np.sqrt(0.5 * h)


def rbf_kernel(x, h=None):
    """RBF kernel with median heuristic for bandwidth."""
    n = x.shape[0]
    pairwise_dists = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)

    if h is None:
        h = median_bandwidth(x)

    K = np.exp(-pairwise_dists / (2 * h ** 2))
    grad_K = -K[:, :, None] * (x[:, None, :] - x[None, :, :]) / (h ** 2)
    return K, grad_K, h


def svgd_forces(particles, grad_log_p, h=None):
    """Compute SVGD attraction and repulsion forces separately.

    Returns:
        attractions: (n, d) array - score-weighted kernel term
        repulsions: (n, d) array - kernel gradient term
        h: bandwidth used
    """
    n = particles.shape[0]
    K, grad_K, h = rbf_kernel(particles, h)
    grad_log_p_vals = grad_log_p(particles)

    # Attraction: K @ grad_log_p / n
    attractions = (K @ grad_log_p_vals) / n

    # Repulsion: sum of kernel gradients / n
    repulsions = grad_K.sum(axis=0) / n

    return attractions, repulsions, h

def svgd_step(particles, grad_log_p, step_size, h=None):
    """One SVGD update step."""
    n = particles.shape[0]
    K, grad_K, h = rbf_kernel(particles, h)
    grad_log_p_vals = grad_log_p(particles)
    phi = (K @ grad_log_p_vals + grad_K.sum(axis=0)) / n
    return particles + step_size * phi, h

def run_svgd(particles, grad_log_p, n_iter=1000, step_size=0.1,
             record_every=10, h=None):
    """Run SVGD for n_iter steps, recording particle history."""
    history = {0: particles.copy()}
    for i in range(1, n_iter + 1):
        particles, h = svgd_step(particles, grad_log_p, step_size, h)
        if i % record_every == 0:
            history[i] = particles.copy()
    return particles, history
