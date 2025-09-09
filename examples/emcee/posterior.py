import dill as pickle
import numpy as np

_config = None
NDIM = None


def init_posterior(config_path):
    """Initialize the global CalibrationConfig on this rank."""
    global _config
    global NDIM
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    _config = cfg
    NDIM = cfg.ndim


def starting_location(nwalkers):
    """Generate starting locations for the walkers."""
    return _config.starting_location(nwalkers)


def log_posterior(theta):
    """
    log-probability for a single walker.

    Parameters
    ----------
    theta : ndarray, shape (ndim)

    Returns
    -------
    log_prob : float
    """
    return _config.log_posterior(theta)


def log_posterior_batch(thetas):
    """
    Vectorized log-probability for all walkers at once.

    Parameters
    ----------
    thetas : ndarray, shape (nwalkers, ndim)

    Returns
    -------
    log_probs : ndarray, shape (nwalkers,)
    """
    return np.array([_config.log_posterior(theta) for theta in thetas])
