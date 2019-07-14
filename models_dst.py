import logging
from math import log, exp
from typing import Dict, Optional
import numpy as np

from scipy import stats

from models import IndependentMultivarModel, \
    BayesianPoissonModel, \
    PoissonModel, \
    AbsContinuousBayesianModel

logger = logging.getLogger('BayesianScience')
logger.setLevel(logging.DEBUG)


DAYS_OF_WEEK = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
WEEKDAYS = DAYS_OF_WEEK[1:6]


class NullHypothesis(IndependentMultivarModel):
    """The DST doesn't have any effect on AMI counts

    x_i ~ Poisson(λ=trend_counts[i]), i = 1..5 (Mon..Fri)
    """

    def __init__(self, trend_counts: Dict[str, float]):
        models = {day: PoissonModel(trend_counts[day]) for day in WEEKDAYS}
        super().__init__(models)


class MondayIncreaseHypothesis(IndependentMultivarModel):
    """The spring DST adjustment increases AMI counts only on the following Monday

    increased_count ~ Gamma(shape=2,
                            loc=trend_counts[1],
                            scale=expected_monday_increase * trend_counts[1])
    x_1 ~ Poisson(λ=increased_count)
    x_i ~ Poisson(λ=trend_counts[i]), i = 2..5 (Tue..Fri)
    """

    def __init__(self, trend_counts: Dict[str, float], expected_monday_increase: float,
                 prior_shape:Optional[float] = None):
        monday_trend = trend_counts["Mon"]
        param_space = np.arange(int(0.7 * monday_trend), int(2.5 * monday_trend))

        if prior_shape:
            prior_shape_arg = dict(prior_shape=prior_shape)
        else:
            prior_shape_arg = dict()

        models = {"Mon": BayesianPoissonModel(param_space=param_space,
                                             prior_loc=monday_trend,
                                             prior_scale=monday_trend * expected_monday_increase,
                                             **prior_shape_arg)}
        for day in WEEKDAYS[1:]:
            models[day] = PoissonModel(trend_counts[day])

        super().__init__(models)


class WeekdaysIncreaseHypothesis(IndependentMultivarModel):
    """The spring DST adjustment increases AMI counts on all weekdays of the following week.

    BUG! This overestimates the posterior; a hierarchical model
      (such as WeekdaysModel) would be more appropriate.

                          { expected_monday_increase       if  i = Mon
    expected_increase_i = { ... (decreasing linearly)
                          { .2 * expected_monday_increase  if  i = Fri

    increased_count_i ~ Gamma(shape=2.0 (by default),
                            loc=trend_counts[i],
                            scale=expected_monday_increase * trend_counts[1])
    x_i ~ Poisson(λ=increased_count_i)

    for i = Mon..Fri
    """

    def __init__(self, trend_counts: Dict[str, float], expected_monday_increase: float, **kwargs):
        models = {}
        for day, factor in zip(WEEKDAYS, np.linspace(1, 0, len(WEEKDAYS), endpoint=False)):
            trend = trend_counts[day]
            param_space = np.arange(int(0.7 * trend), int(2.5 * trend))
            expected_increase = expected_monday_increase * factor

            models[day] = BayesianPoissonModel(param_space=param_space,
                                               prior_loc=trend,
                                               prior_scale=trend * expected_increase,
                                               **kwargs)

        super().__init__(models)


class WeekdaysModel(AbsContinuousBayesianModel):
    """Model for AMI counts after the spring DST adjustment on all weekdays of the following week.

    which_model ~ Discrete([0.5, 0.5])

    case which_model of
      0:
        theta ~ Normal(loc=0.0,
                       stdev=0.01)

      1:
        theta ~ Gamma(shape=1.0,
                      loc=0,
                      scale=expected_monday_increase)

              { 1.0 * theta   if  i = Mon
    theta_i = { ... (decreasing linearly)
              { 0.2 * theta   if  i = Fri

    x_i ~ Poisson(λ=trend_counts[i] * (1 + theta_i))

    D = {x_i}_{i=1}^5
    P(D | theta) = Π_i P(x_i | theta)

    for i = Mon..Fri
    """

    GAMMA_SHAPE = 1.0
    NORMAL_SCALE = 0.01

    def __init__(self, trend_counts: Dict[str, float], expected_monday_increase: float, **kwargs):
        self._update_dict(locals())

        self.param_space = np.linspace(-0.1, 2.0, 1000)

        models = {}

        for day, factor in zip(WEEKDAYS, np.linspace(1, 0, len(WEEKDAYS), endpoint=False)):
            day_trend = trend_counts[day]
            day_param_space = day_trend * (1 + self.param_space)

            models[day] = BayesianPoissonModel(param_space=day_param_space,
                                               prior_loc=None,  # prior is defined in this class
                                               prior_scale=None,
                                               **kwargs)

        self.models = models

        super().__init__(self.param_space)

    def prior(self):
        m0 = stats.norm.pdf(self.param_space,
                            loc=0.,
                            scale=self.NORMAL_SCALE)

        m1 = stats.gamma.pdf(self.param_space,
                             a=self.GAMMA_SHAPE,
                             loc=0.,
                             scale=self.expected_monday_increase)

        prior = 0.5 * m0 + 0.5 * m1

        logger.debug("Total probability mass in prior: {:.3f}".format(
                        prior.sum() * self.param_step))

        return prior

    def likelihood(self, obs: dict) -> np.ndarray:
        partial_likelihoods = {key: self.models[key].likelihood(obs[key])
                               for key in self.models}

        return np.exp(sum(np.log(p) for p in partial_likelihoods.values()))
