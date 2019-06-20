"""Standardized Sampling Format and Implementations"""

import numpy as np
from diversipy import random_uniform


def gen_problem(params):
    """Generate a Problem Specification

    The problem specification is an expanded dictionary of parameter names and bounds. Containing
    additional fields convenient for use in generating bounds

    Parameters
    ----------
    params: dict
        The dictionary  should contain the keys and lists ``{'names':[.....], 'bounds':[.....]}`` where
        ``'names'`` is a list of feature names and ``'bounds'`` the corresponding tuple of upper and lower bounds

    Returns
    -------
    problem: dict
        Expanded param dictionary containing additional fields:

        - num_vars: Dimensionality of the problem
        - lower_bounds: 1-d array of lower bounds
        - upper_bounds: 1-d array of upper bounds
        - bounded_range: 1-d array of bound range sizes

    Raises
    ------
    AssertionError
        Checks that the input parameters are valid

    """
    assert 'names' in params, "Parameter definition must contain a list of feature names!"
    assert 'bounds' in params, "Parameter definition must contain a list of feature bounds!"
    assert len(params['names']) == len(params['bounds']), "Name and bound lists must be the same size!"

    problem = {'num_vars': len(params['names']),
               'names': params['names'],
               'bounds': np.array([np.array(i) for i in params['bounds']])}  # Not assuming that this is passed as numpy

    problem['lower_bounds'] = problem['bounds'][:, 0]
    problem['upper_bounds'] = problem['bounds'][:, 1]
    problem['bounded_range'] = problem['upper_bounds'] - problem['lower_bounds']

    return problem


class BaseSampler:
    """Base Sampler Functionality

    Class wrapping common sampler functionality. The different sampler implementations we are pulling in
    have very varied approaches, so for now this approach is safest

    Parameters
    ----------
    name: str
        Full name for the sampler
    problem_setting: dict, optional
        Full problem definition as generated by ``gen_problem``
    parameters: dict, optional
        Problem parameters in the form ``{'names':[], 'bounds':[[float, float],..] }``

    Raises
    ------
    AssertionError
        The constructor expects one of ``problem_setting`` or ``parameters`` but not both
    """

    def __init__(self, name, problem_setting=None, parameters=None):
        self.name = name

        if parameters:
            assert (not problem_setting), 'Only pass one off parameters or a problem definition'
            self.p = gen_problem(parameters)

        elif problem_setting:
            self.p = problem_setting

        else:
            raise Exception()

    def sample(self, sample_size, rng_seed=0):
        """Generate Samples

        This wrapper should standardise sampler calls independent of the library being used

        Parameters
        ----------
        sample_size: int
            Number of samples to generate
        rng_seed: int, optional
            Seed used for random number generation. In some cases may be redundant. Default is 0.
        """
        pass

    def rescale(self, samples):
        """Rescale sample to problem bounds

        Rescales samples in the range [0,1 ] to the fit the bounds as described by the problem attribute

        Parameters
        ----------
        samples: array[float]
            Numpy array of samples in the range [0,1]

        Returns
        -------
        scaled_samples: array[float]
            Numpy array of rescaled parameters
        """
        return samples * self.p['bounded_range'] + self.p['lower_bounds']


class RandomUniformSampler(BaseSampler):
    """Random Uniform sampler

    Produces samples drawn uniformly from the problem bounds

    Parameters
    ----------
    problem_setting: dict, optional
        Full problem definition as generated by ``gen_problem``
    parameters: dict, optional
        Problem parameters in the form ``{'names':[], 'bounds':[[float, float],..] }``
    """

    def __init__(self, problem_setting=None, parameters=None):
        super().__init__('Random Uniform',
                         problem_setting=problem_setting,
                         parameters=parameters)

    def sample(self, sample_size, rng_seed=0):
        """Generate Uniform Random Samples

        Produce samples drawn uniformly from the classes problem bounds

        Parameters
        ----------
        sample_size: int
            Number of samples to generate
        rng_seed: int, optional
            Seed used for random number generation. In some cases may be redundant.

        Returns
        -------
        samples: array[float]
            Numpy array of samples
        """
        np.random.seed(rng_seed)
        parameter_combinations = random_uniform(sample_size, self.p['num_vars'])
        range_multiplier = np.tile(self.p['bounded_range'], (sample_size, 1))
        return np.multiply(parameter_combinations, range_multiplier) + self.p['lower_bounds']
