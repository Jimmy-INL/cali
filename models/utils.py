"""Maths utils"""
import numpy as np
from numba import jit


@jit
def pct_changes(x):
    """Percentage change of a series

    Return a series of percentage changes from a series. Accepts multivariate series

    Parameters
    ----------
    x: array [float, features]
        Numpy array of floats. The series to be converted to percentages should be the first index

    Returns
    -------
    percentage changes: array [float, features]
        Percentage changes of the input series on index 0

    Notes
    ----------
    Assumes the series is positive
    """
    return (x[1:]-x[:-1])/x[:-1]


@jit
def inv_pct_change(pct, x0):
    """Inverse percentage change

        Return the next value from a previous value and a percentage change

        Parameters
        ----------
        pct: float
            Percentage change, in the range
        x0: float
            The previous value

        Returns
        -------
        x1: float
            The next value given the percentage change
        """
    return x0*(1+pct)


def sample_time_series(price_samples, n_samples, window):
    """Generate a set of input-target samples from time-series

        Randomly sample a set of time-series to produce a set of training samples. The training samples are
        a window of historical (differenced) values, and a integer +-1 that indicates the following direction
        of the price movement

        Parameters
        ----------
        price_samples: array[time-series, samples]
            The set of time-series samples to sample from
        n_samples: int
            Number of samples to generate
        window: int
            The size of the window to sample as input features

        Returns
        -------
        xs: array[window, samples]
            Input samples. Essentially a set of samples of windows of the time series.
        ys: [int]
            A list of target values in {-1, 1} indicating the next direction of the move in price, corresponding
            to the windows in xs
        """
    pctc = pct_changes(price_samples).T

    s_samples = np.random.randint(0, high=pctc.shape[0], size=n_samples)
    t_samples = np.random.randint(window + 1, high=pctc.shape[1], size=n_samples)

    ss = np.stack([pctc[i[0], i[1] - window - 1:i[1]] for i in zip(s_samples, t_samples)])
    xs = ss[:, :window]
    ys = (ss[:, -1] > 0).astype(int)  # Target classes are +ve/-ve pct changes

    return xs, ys


@jit
def safe_exponent(x):
    """Safe exponent to stay in bounds

        Safely use exponent when there is no guarantee that the final value will not cause an overflow

        Parameters
        ----------
        x: float
            Exponent

        Returns
        -------
        exp(x): float
            Either the value of the exponent of the max float value
        """
    if x > 709:
        return 8.2184074615549724e+307
    else:
        return np.exp(x)
