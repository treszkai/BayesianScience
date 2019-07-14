import logging
from math import log, exp
from typing import Union, List, Dict, Any

import numpy as np
import scipy.stats as st

import utils

logger = logging.getLogger('BayesianScience')

logger.setLevel(logging.DEBUG)


class MyObject(object):
    def _update_dict(self, args):
        """Update the attributes of an object with a dictionary

        Typical usage:

        def __init__(self, foo, bar):
            self._update_dict(locals())
            # equivalent to:
            #   self.foo = foo
            #   self.bar = bar
        """
        self.__dict__.update({k: v for k, v in args.items() if k != 'self'})


class AbsModel(MyObject):
    def likelihood(self, obs):
        """P[obs] or P[obs | params]"""
        raise NotImplementedError

    def unnormalized_posterior(self, obs):
        return self.likelihood(obs)

    def marginal_likelihood(self, obs):
        return self.unnormalized_posterior(obs)


class AbsParameterfreeModel(AbsModel):
    def likelihood(self, obs):
        """P[obs]"""
        raise NotImplementedError

    def marginal_likelihood(self, obs):
        return self.unnormalized_posterior(obs)


class AbsBayesianModel(AbsModel):
    """Abstract class for a Bayesian model"""

    def prior(self):
        """P[params]"""
        raise NotImplementedError

    def likelihood(self, obs):
        raise NotImplementedError

    def unnormalized_posterior(self, obs):
        """P[params, Obs = obs]"""
        return self.prior() * self.likelihood(obs)

    def posterior(self, obs):
        """P[params | obs]"""
        return self.unnormalized_posterior(obs) / self.marginal_likelihood(obs)


class AbsContinuousBayesianModel(AbsBayesianModel):
    """Abstract class for a Bayesian model with a single continuous parameter"""

    def __init__(self, param_space: np.ndarray):
        """Assumes uniformly spaced 1-dimensional linspace for param_space"""
        assert len(param_space.shape) == 1
        self.param_space = param_space
        super().__init__()

    @property
    def param_step(self):
        return utils.linspace_step(self.param_space)

    def marginal_likelihood(self, obs):
        """P[obs]"""
        return np.sum(self.unnormalized_posterior(obs)) * self.param_step


class PoissonModel(AbsParameterfreeModel):
    def __init__(self, param):
        self.param = param
        super().__init__()

    def marginal_likelihood(self, obs):
        assert int(obs) == obs, "Observation must be an integer"
        return st.poisson.pmf(obs, self.param)

    def __str__(self):
        return f"PoissonModel({self.param:.2f})"


class BayesianPoissonModel(AbsContinuousBayesianModel):
    """Poisson model with a Gamma prior on the parameter"""

    def __init__(self, param_space: np.ndarray, prior_loc: float, prior_scale: float, prior_shape:float = 2.):
        self._update_dict(locals())
        super().__init__(param_space)

    def prior(self) -> np.ndarray:
        return st.gamma.pdf(self.param_space, a=self.prior_shape, loc=self.prior_loc, scale=self.prior_scale)

    def likelihood(self, obs):
        assert int(obs) == obs, "Observation must be an integer"
        return st.poisson.pmf(obs, self.param_space)


class IndependentMultivarModel(AbsModel):
    def __init__(self, models: Dict[Any, AbsModel]):
        assert len(models) > 0
        self._update_dict(locals())
        super().__init__()

    def unnormalized_posterior(self, obs: dict):
        partial_unnormalized_posteriors = {key: self.models[key].unnormalized_posterior(obs[key])
                                           for key in self.models}

        logger.debug(f"Likelihoods of submodels: {partial_unnormalized_posteriors}")

        return exp(sum(log(p) for p in partial_unnormalized_posteriors.values()))

    def marginal_likelihood(self, obs: dict):
        partial_marginal_likelihoods = {key: self.models[key].marginal_likelihood(obs[key])
                                        for key in self.models}

        logger.debug(f"Likelihoods of submodels: {partial_marginal_likelihoods}")

        return exp(sum(log(p) for p in partial_marginal_likelihoods.values()))
