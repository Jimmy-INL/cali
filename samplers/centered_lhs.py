"""PyDOE LHS samplers
https://pythonhosted.org/pyDOE/randomized.html
"""

import pyDOE

from samplers.base_sampler import BaseSampler


class LhsCenter(BaseSampler):
    """PyDOE Latin Hyper-Squares Center

    Center the points within the sampling intervals
    """

    def __init__(self, problem_setting=None, parameters=None):
        super(LhsCenter, self).__init__('pyDOE LHS Centre', problem_setting=problem_setting, parameters=parameters)

    def sample(self, sample_size, rng_seed=0):
        s = pyDOE.lhs(self.p['num_vars'], samples=sample_size, criterion='c')
        return self.rescale(s)


calibration_params = {'names': ['beta', 'R', 'b1', 'b2', 'g1 ', 'g2', 'C', 'eta',
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
