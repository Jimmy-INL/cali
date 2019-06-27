class _BaseAbm:
    """Base ABM shared functionality

    Basic structure expected of an ABM as expected by other components of the platform
    New ABM's should impliment or extend this basic format

    Parameters
    ----------
    name : str
        ABMs full name
    calibrated_params: dict
        Dictionary describing the model parameters and suggested bounds. Should be in the format
        {'names': array[str,..], 'bounds': array[[int, int],..]}
    n_agents: int
        Number of agents to run the model with. This may well be a dummy value, but is used in most models
    burn_in: int
        Number of steps to use as burn in phase of the model i.e. the returned time series starts from this
        point in the full simulation. By design, if the model fails to meet the required number of steps it
        ignores the burn_in for debug purposes
    """

    def __init__(self, name, calibrated_params, n_agents, burn_in):
        self.name = name
        self.calibrated_params = calibrated_params
        self.n_agents = n_agents
        self.burn_in = burn_in

    def run(self, n_steps, starting_price_series, param_vec, rng_seed=0, **kwargs):
        """Basic ABM running signature

            Shared minimal ABM running signature. Should be implemented by all child implementations and
            extended with keywords

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
            **kwargs:
                Additional keyword argument specific to an ABM
            Returns
            -------
             prices: array[time_series]
                Numpy array in the shape [time-series x features]
            success_flag: boolean
                Flag indicating whether the ABM reached the requested number of time steps without blowing-up/crashing
        """
        pass
