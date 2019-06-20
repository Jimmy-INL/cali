"""
Brock and Hommes Model
"""
from numba import jit
import numpy as np
from numpy.random import RandomState

from models._BaseAbm import _BaseAbm
from models import utils


calibrated_params = {'names': ['beta', 'R', 'b1', 'b2', 'g1 ', 'g2', 'C', 'eta',
                               'n1', 'alpha', 'sigma', 'epsilon_sigma', 'epsilon_mu'],
                     'bounds': [[0.0, 10.0],
                                [1.01, 1.1],
                                [-2.0, 2.0],
                                [-2.0, 2.0],
                                [-2.0, 2.0],
                                [-2.0, 2.0],
                                [0.0, 5.0],
                                [0.0, 1.0],
                                [0.0, 1.0],
                                [0.0001, 2.0],
                                [0.0001, 100.0],
                                [0.0001, 100.0],
                                [-1.0, 1.0]]}


@jit
def run_func(n_steps, n_agents, param_vec, seed_series, rng_seed, burn_in):
    """Run the Brocke and Hommes model

            Run the Brocke and Hommes model from a set of parameters and an initializing price history

            Parameters
            ----------
            n_steps: int
                Number of time-steps to run the simulation for
            n_agents : int unused
                Number of agents to generate for ABM.
            param_vec: ['beta',  'R', 'b1', 'b2', 'g1 ', 'g2', 'C', 'eta', 'n1', 'alpha', 'sigma', 'epsilon_sigma', 'epsilon_mu']
                Parameters are roughly: TODO: Write
            seed_series: array[float..]
                Historical price history to use as a warm start for the run of the parameter
            rng_seed: int
                Random number generator seed
            burn_in: int
                Number of steps to allow for model "burn-in", i.e. the returned price series will remove this number
                of leading steps if the run was successful. If the run goes out of bounds the return time series will
                ignore this burn-in and return the full run for debugging purposes

            Returns
            -------
             prices: array[prices]
                Numpy array in the shape [prices x 1]
             success_flag: boolean
                Flag indicating whether the ABM reached the requested number of time steps without blowing-up/crashing
            """
    beta = param_vec[0]
    R = param_vec[1]
    b1 = param_vec[2]
    b2 = param_vec[3]
    g1 = param_vec[4]
    g2 = param_vec[5]
    C = param_vec[6]
    eta = param_vec[7]
    n1 = param_vec[8]
    alpha = param_vec[9]
    sigma = param_vec[10]
    epsilon_sigma = param_vec[11]
    epsilon_mu = param_vec[12]

    prng = RandomState(rng_seed)

    total_steps = n_steps + burn_in

    # Initial wealth
    u1 = 0  # initial wealth type 1
    u2 = 0  # initial wealth type 2

    pdiff = np.zeros(total_steps)

    p = np.zeros((total_steps + 1, 1))
    p[0, 0] = seed_series[-1]

    eps = prng.normal(loc=epsilon_mu, scale=epsilon_sigma, size=total_steps)  # additive noise

    n2 = 1.0 - n1

    x = seed_series[-1] - np.mean(seed_series[-4:])
    y = x

    aversion = alpha * (sigma ** 2)

    for t in range(total_steps):

        # Agent Forecasts
        f1_0 = g1 * x + b1
        f2_0 = g2 * x + b2

        f1_1 = g1 * y + b1
        f2_1 = g2 * y + b2

        # Price forecast
        y = x
        x = ((n1 * f1_0 + n2 * f2_0) + eps[t]) / R

        if np.isnan(x):
            return p[: t + 1], False

        # Agent Utilities
        rx = R * y
        x_deviation = x - rx
        risk_aversion = x_deviation / aversion
        u1 = risk_aversion * (f1_1 - rx) + eta * u1 - C
        u2 = risk_aversion * (f2_1 - rx) + eta * u2

        # Update probabilities
        n1 = utils.safe_exponent(beta * u1)
        n2 = utils.safe_exponent(beta * u2)

        norm = (n1 + n2)

        if norm > 0.0:
            n1 /= norm
            n2 /= norm

        new_price = p[t, 0] + x  # utils.inv_pct_change(p[t, 0], x)

        # Stop run if price totally crashes or too large a swing
        if np.abs(x / p[t, 0]) > 4.0 or new_price < 1e-12:
            return p[: t + 1], False

        pdiff[t] = x
        p[t + 1, 0] = new_price

    return p[burn_in + 1:], True


class ABM(_BaseAbm):
    """Brocke and Hommes ABM

        Initialize a Brocke and Hommes ABM

        Parameters
        ----------
        n_agents: int
            Number of agents to run the model with. This may well be a dummy value, but is used in most models
        burn_in: int optional
            Number of steps to use as burn in phase of the model i.e. the returned time series starts from this
            point in the full simulation. By design, if the model fails to meet the required number of steps it
            ignores the burn_in for debug purposes
        """

    def __init__(self, n_agents, burn_in=100):
        super().__init__("Brocke and Hommes", calibrated_params, n_agents, burn_in)

    def run(self, n_steps, starting_price_series, param_vec, rng_seed=0, **kwargs):
        """Run model

            Run the model and return prices series and success flag

            Parameters
            ----------
            n_steps : int
                Number of steps to run model, i.e total length of return time series if run is successful
            starting_price_series: array[float]
                Starting price-series that the model runs from, used as steps in the past to start the model from
            param_vec: array[float]
                Array of ABM parameters, as outlined by the ABMs calibrated_params
            rng_seed: int
                Random number generator seed
            percentage_returns: boolean
                if set to true the percentage returns are calculated
            \*\* kwargs:
                Additional keyword argument specific to an ABM

            Returns
            -------
             prices: array[prices]
                Numpy array in the shape [prices x 1].
             success_flag: boolean
                Flag indicating whether the ABM reached the requested number of time steps without blowing-up/crashing
            """
        return run_func(n_steps, self.n_agents, param_vec, starting_price_series, rng_seed, self.burn_in)
