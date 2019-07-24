import math
from numbers import Real
from textwrap import wrap
from typing import Tuple, Union, List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DAYS_OF_WEEK = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
WEEKDAYS = DAYS_OF_WEEK[1:6]


def parse_second_column(data_str):
    return [float(line.split(", ")[1]) for line in data_str.split("\n")]


def parse_data(data_str: str, days: List[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Parse string with lines "obs_i/trend_i" and return (obs,trend) dictinaries """
    dict_obs = {}
    dict_trend = {}

    for day, data_day_raw in zip(days, data_str.split("\n")):
        s_obs, s_trend = data_day_raw.split('/')
        dict_obs[day] = int(s_obs)
        dict_trend[day] = float(s_trend)

    return dict_obs, dict_trend


def linspace_step(linspace: np.ndarray) -> float:
    return (linspace[-1] - linspace[0]) / (len(linspace) - 1)


def central_credible_interval(param_space: np.ndarray, pdf: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Calculate the central credible interval for a univariate distribution

    This means an interval [lb,ub] such that
      P[X < lb] <= alpha/2, and
      P[X > ub] <= alpha/2

    The distribution can be either continuous or discrete,
    so the `probs` argument can be either a pdf or a pmf.

    Note: When calculated for the posterior, it is called the “central posterior interval”.
    """
    cdf = np.cumsum(pdf) * linspace_step(param_space)
    axis = 0
    lb_idx = np.where(cdf > alpha / 2)[axis][0]
    lb = param_space[lb_idx]
    ub_idx = np.where(cdf > 1 - alpha / 2)[axis][0]
    ub = param_space[ub_idx]
    return lb, ub


def highest_density_region() -> List[Tuple[float, float]]:
    """Calculate the highest density region"""
    # https://stackoverflow.com/a/22290087/8424390
    raise NotImplementedError


def calc_irr(trend: Union[float, np.ndarray], observed: int) -> Union[float, np.ndarray]:
    """Calculates the incidence rate ratio (IRR)"""
    return observed/trend - 1.0


def calc_inv_irr(irr: Union[float, np.ndarray], trend: float) -> Union[float, np.ndarray]:
    """The inverse of calc_irr

    irr = calc_irr(trend, obs)
    obs == calc_inv_irr(irr, trend)
    """
    return trend * (irr + 1.0)


def calc_log_likelihood(param_space: np.ndarray,
                        ami_trend: Dict[str, Real],
                        ami_observed: Dict[str, int],
                        num_affected_days: Optional[int] = None) -> np.ndarray:
    """Calculates P[obs | theta] for a Poisson model

    Parameter: increase in AMI counts on the Monday following DST change; e.g. 0.2 means a +20% increase

    :param param_space: NumPy array of the parameters to be considered
    :param ami_observed: Observed AMI counts; a dictionary mapping day name to an int
    :param ami_trend: AMI counts predicted by the trend; a dictionary mapping day name to a number
    :param num_affected_days: number of affected workdays after a DST change 1..5 (default: all weekdays)
    :return: the log likelihood, calculated at the points of param_space
    """
    log_likelihood = np.zeros_like(param_space)

    affected_days = WEEKDAYS[:num_affected_days]

    for day, factor in zip(affected_days, np.linspace(1, 0, len(affected_days), endpoint=False)):
        day_increased_counts = calc_inv_irr(param_space * factor, ami_trend[day])
        log_likelihood += stats.poisson.logpmf(ami_observed[day], day_increased_counts)

    return log_likelihood


def calc_posterior(param_space: np.ndarray,
                   prior: np.ndarray,
                   log_likelihoods: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    if type(log_likelihoods) is np.ndarray:
        log_likelihoods = [log_likelihoods]

    unnormalized_log_posterior = np.log(prior)
    for log_likelihood in log_likelihoods:
        assert param_space.shape == prior.shape == log_likelihood.shape
        unnormalized_log_posterior += (log_likelihood)

    unnormalized_posterior = np.exp(unnormalized_log_posterior - np.max(unnormalized_log_posterior))

    return unnormalized_posterior / (np.sum(unnormalized_posterior) * linspace_step(param_space))


# Plotting functions

def plot_poissons_error(ys, ys2, p=0.025, kw2={}, extra_code=""):
    yerr = np.vstack([[y - stats.poisson.ppf(p, y) for y in ys.values()],
                      [stats.poisson.ppf(1 - p, y) - y for y in ys.values()]])
    fig, ax = plt.subplots(1, 1)
    plt.errorbar(ys.keys(), ys.values(), yerr=yerr, fmt='o',
                 elinewidth=1, capsize=10, ms=10,
                 label='trend')
    plt.plot(list(ys2), list(ys2.values()), 'x', c='red', ms=10, mew=4, alpha=0.7, zorder=10, **kw2)
    ax.legend(loc='best', frameon=False)
    plt.grid(True, axis='y', alpha=0.3)
    exec(extra_code)
    plt.show()
    return fig


def plot_posterior(param_space,
                   prior,
                   log_likelihoods: Union[np.ndarray, Dict[str, np.ndarray]],
                   paper_title=None,
                   alpha=0.05):
    if type(log_likelihoods) is np.ndarray:
        log_likelihoods = {paper_title or 'this': log_likelihoods}

    posterior = calc_posterior(param_space,
                               prior,
                               log_likelihoods.values())

    ax = plt.gca()
    lb, ub = param_space[[0, -1]]
    ci_lb, ci_ub = central_credible_interval(param_space, posterior, alpha)

    plt.plot(param_space, posterior, label='Posterior')
    plt.plot(param_space, prior, c='red', lw=1, alpha=0.7, label='Prior')

    ax.set_xlim(lb, ub)
    ax.set_ylim(-0.01)

    ax.axvspan(xmin=ci_lb, xmax=ci_ub, lw=1, alpha=0.1,
               label=f'{round(100*(1-alpha))}% central credible interval')
    ax.legend(loc='upper right', frameon=True)

    ax.set_title("\n".join(wrap(f"Posterior probability after {' & '.join(log_likelihoods.keys())} paper", 60)))

    ax.set_xlabel("Relative increase in AMI count")
    ax.set_ylabel("Posterior probability")
    ax.set_yticklabels(["", "0.0"])

    plt.grid(True, alpha=0.5)

    return ci_lb, ci_ub


def process_study_results(all_data, param_space, prior, log_likelihoods,
                          paper_id, ami_trend, ami_obs, num_years,
                          **kwargs):

    # plot study results with error bars
    plot_poissons_error(ami_trend, ami_obs, kw2=dict(label='actual'),
                        extra_code="plt.title('Observed and trend values with 95% error bars')");

    # calculate the log likelihood
    log_likelihood = calc_log_likelihood(param_space, ami_trend, ami_obs)

    # register study results
    if all_data is not None:
        all_data[paper_id]['trend'] = ami_trend
        all_data[paper_id]['obs'] = ami_obs
        all_data[paper_id]['num_years'] = num_years
        log_likelihoods[paper_id] = log_likelihood

    plt.figure()
    plt.plot(param_space, np.exp(log_likelihood))
    plt.title("Likelihood of different parameters")
    plt.xlabel("Relative increase in AMI count")
    plt.ylabel("Likelihood")
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_yticklabels(["0.0"])
    plt.gca().set_xlim(param_space[[0, -1]])
    plt.grid(True, alpha=0.5)
    plt.show()

    plt.figure()
    plot_posterior(param_space, prior, log_likelihood, paper_title=paper_id)
    plt.show()

