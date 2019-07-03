from models import IndependentMultivarModel, BayesianPoissonModel, PoissonModel

from typing import Dict
import numpy as np


DAYS_OF_WEEK = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
WEEKDAYS = DAYS_OF_WEEK[1:6]


class NullHypothesis(IndependentMultivarModel):
    """The DST doesn't have any effect on AMI counts

    x_i ~ Poisson(λ=trend_counts[i]), i = 1..5 (Mon..Fri)
    """

    def __init__(self, trend_counts: Dict[str, float]):
        models = {day: PoissonModel(trend_counts[day]) for day in WEEKDAYS}
        super().__init__(models)


class MondayHypothesis(IndependentMultivarModel):
    """The spring DST adjustment increases AMI counts only on the following Monday

    increased_count ~ Gamma(shape=2,
                            loc=trend_counts[1],
                            scale=expected_monday_increase * trend_counts[1])
    x_1 ~ Poisson(λ=increased_count)
    x_i ~ Poisson(λ=trend_counts[i]), i = 2..5 (Tue..Fri)
    """

    def __init__(self, trend_counts: Dict[str, float], expected_monday_increase: float):
        monday_trend = trend_counts["Mon"]
        param_space = np.arange(int(0.7 * monday_trend), int(2.5 * monday_trend))

        models = {}
        models["Mon"] = BayesianPoissonModel(param_space=param_space,
                                             prior_loc=monday_trend,
                                             prior_scale=monday_trend * expected_monday_increase)
        for day in WEEKDAYS[1:]:
            models[day] = PoissonModel(trend_counts[day])

        super().__init__(models)


class WeekdaysHypothesis(IndependentMultivarModel):
    """The spring DST adjustment increases AMI counts on all weekdays of the following week.

                          { expected_monday_increase       if  i = Mon
    expected_increase_i = { ... (decreasing linearly)
                          { .2 * expected_monday_increase  if  i = Fri

    increased_count_i ~ Gamma(shape=2.0 (by default),
                            loc=trend_counts[i],
                            scale=expected_monday_increase * trend_counts[1])
    x_i ~ Poisson(λ=trend_counts[i])

    for i = Mon..Fri
    """

    def __init__(self, trend_counts: Dict[str, float], expected_monday_increase: float):
        models = {}
        for day, factor in zip(WEEKDAYS, np.linspace(1, 0, len(WEEKDAYS), endpoint=False)):
            trend = trend_counts[day]
            param_space = np.arange(int(0.7 * trend), int(2.5 * trend))
            expected_increase = expected_monday_increase * factor

            models[day] = BayesianPoissonModel(param_space=param_space,
                                               prior_loc=trend,
                                               prior_scale=trend * expected_increase)

        super().__init__(models)
