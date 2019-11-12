from numbers import Real
from typing import Tuple, Union, List, Dict, Iterable
import logging

import numpy as np
from scipy import stats, optimize

DAYS_OF_WEEK = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
WEEKDAYS = DAYS_OF_WEEK[1:6]

logger = logging.getLogger('BayesianScience')

def parse_second_column(data_str):
    return [float(line.split(", ")[1]) for line in data_str.split("\n")]


def parse_data(data_str: str, days: List[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Parse string with lines "obs_i/trend_i" and return (obs,trend) dictionaries"""
    dict_obs = {}
    dict_trend = {}

    for day, data_day_raw in zip(days, data_str.split("\n")):
        s_obs, s_trend = data_day_raw.split('/')
        dict_obs[day] = int(s_obs)
        dict_trend[day] = float(s_trend)

    return dict_obs, dict_trend


def linspace_step(linspace: np.ndarray) -> float:
    """Calculate the step size of a linearly spaced interval"""
    return (linspace[-1] - linspace[0]) / (len(linspace) - 1)


def central_credible_interval(param_space: np.ndarray, pdf: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Calculate the central credible interval for a continuous univariate distribution

    This means an interval [lb,ub] such that
      P[X < lb] <= alpha/2, and
      P[X > ub] <= alpha/2

    Note: When calculated for the posterior, it is called the “central posterior interval”.
    """
    cdf = np.cumsum(pdf) * linspace_step(param_space)
    axis = 0
    lb_idx = np.where(cdf > alpha / 2)[axis][0]
    lb = param_space[lb_idx]
    ub_idx = np.where(cdf > 1 - alpha / 2)[axis][0]
    ub = param_space[ub_idx]
    return lb, ub


def central_credible_interval_discrete(param_space: np.ndarray, pmf: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Calculate the central credible interval for a discrete univariate distribution

    This means an interval [lb,ub] such that
      P[X < lb] <= alpha/2, and
      P[X > ub] <= alpha/2

    Note: When calculated for the posterior, it is called the “central posterior interval”.
    """
    cdf = np.cumsum(pmf)
    axis = 0
    lb_idx = np.where(cdf > alpha / 2)[axis][0]
    lb = param_space[lb_idx]
    ub_idx = np.where(cdf > 1 - alpha / 2)[axis][0]
    ub = param_space[ub_idx]
    return lb, ub


def highest_density_region(param_space: np.ndarray,
                           pdf: np.ndarray,
                           alpha: float) -> List[Tuple[float, float]]:
    """Calculate the highest density region (HDR) of a 1-dim PDF

    Assumes the whole PDF is covered in param_space

    Return value is a list of intervals which together make up the HDR.
    """
    # inspired by https://stackoverflow.com/a/22290087/8424390

    def errfn(p, xs, ps, alpha):
        # TODO: this integration method is _very_ inaccurate
        # use scipy.integrate.simps instead
        prob = np.sum(ps[ps > p]) * linspace_step(xs)
        return (prob + alpha - 1.0) ** 2

    p = optimize.fmin(errfn, x0=0, args=(param_space, pdf, alpha))[0]

    logger.debug('HDR: p = %3g' % p)

    in_interval = (pdf > p)
    intervals_begin = param_space[np.where(np.r_[1, 1 - in_interval[:-1]] * in_interval)[0]]
    intervals_end = param_space[np.where(np.r_[1 - in_interval[1:], 1] * in_interval)[0]]

    assert len(intervals_begin) == len(intervals_end), "calculation error"

    hd_intervals = list(zip(intervals_begin, intervals_end))

    assert all(begin < end for (begin, end) in hd_intervals), \
        "calculation error"

    assert all(end < next_begin
               for ((_, end), (next_begin, _))
               in zip(hd_intervals, hd_intervals[1:])), \
        "calculation error"

    return hd_intervals


def calc_log_likelihood(param_space: np.ndarray,
                        ami_trend: Dict[str, Real],
                        ami_observed: Dict[str, int]) -> np.ndarray:
    """Calculates P[obs | theta] for the Poisson weekday model

    Parameter: increase in AMI counts on the Monday following DST change; e.g. 0.2 means a +20% increase

    :param param_space: NumPy array of the parameters to be considered
    :param ami_observed: Observed AMI counts; a dictionary mapping day name to an int
    :param ami_trend: AMI counts predicted by the trend; a dictionary mapping day name to a number
    :param num_affected_days: number of affected workdays after a DST change 1..5 (default: all weekdays)
    :return: the log likelihood, calculated at the points of param_space
    """
    log_likelihood = np.zeros_like(param_space)

    for day, factor in zip(WEEKDAYS, np.linspace(1, 0, len(WEEKDAYS), endpoint=False)):
        day_increased_counts = (1 + param_space * factor) * ami_trend[day]
        log_likelihood += stats.poisson.logpmf(ami_observed[day], day_increased_counts)

    return log_likelihood


def calc_posterior(param_space: np.ndarray,
                   prior: np.ndarray,
                   log_likelihoods: Union[np.ndarray, Iterable[np.ndarray]]) -> np.ndarray:
    if type(log_likelihoods) is np.ndarray:
        log_likelihoods = [log_likelihoods]

    unnormalized_log_posterior = np.log(prior)
    for log_likelihood in log_likelihoods:
        assert param_space.shape == prior.shape == log_likelihood.shape
        unnormalized_log_posterior += log_likelihood

    unnormalized_posterior = np.exp(unnormalized_log_posterior - np.max(unnormalized_log_posterior))

    return unnormalized_posterior / (np.sum(unnormalized_posterior) * linspace_step(param_space))
