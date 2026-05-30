"""Utility functions for Jupyter notebooks."""

from IPython.display import HTML, display
import numpy as np
import pytensor.tensor as pt


# --- Pathological distributions (NumPy versions for visualization) ---

def funnel_logp(y, x):
    """Log density of Neal's funnel: p(y, x) = N(y|0,9) * N(x|0, exp(y))"""
    return -0.5 * (y**2 / 9 + np.log(9) + x**2 / np.exp(y) + y + np.log(2 * np.pi))


def rosenbrock_logp(x1, x2, a=1.0, b=3.0):
    """Log density of Rosenbrock (banana) distribution."""
    return -(a - x1)**2 - b * (x2 - x1**2)**2


def two_beans_logp(x0, x1):
    """Log density of two beans distribution (disconnected bimodal)."""
    norm = np.sqrt(x0**2 + x1**2)
    term1 = -0.5 * ((norm - 3) / 0.4)**2
    term2 = np.logaddexp(-0.5 * ((x0 - 3) / 0.4)**2, -0.5 * ((x0 + 3) / 0.4)**2)
    return term1 + term2


# --- Pathological distributions (PyTensor versions for PyMC) ---

def funnel_logp_pt(y, x):
    """PyTensor log density of Neal's funnel."""
    return -0.5 * (y**2 / 9 + pt.log(9) + x**2 / pt.exp(y) + y + pt.log(2 * np.pi))


def rosenbrock_logp_pt(x1, x2, a=1.0, b=3.0):
    """PyTensor log density of Rosenbrock (banana) distribution."""
    return -(a - x1)**2 - b * (x2 - x1**2)**2


def two_beans_logp_pt(value):
    """PyTensor log density of two beans distribution for PyMC Potential."""
    value = pt.atleast_2d(value)
    x0 = value[:, 0]
    x1 = value[:, 1]
    norm = pt.sqrt(x0**2 + x1**2)
    term1 = -0.5 * ((norm - 3.) / 0.4)**2
    term2 = pt.logsumexp(
        pt.stack([-0.5 * ((x0 - 3) / 0.4)**2, -0.5 * ((x0 + 3) / 0.4)**2], axis=1),
        axis=1
    )
    return pt.squeeze(term1 + term2)


