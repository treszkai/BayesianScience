from numbers import Real
from typing import Tuple

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def parse_second_column(data_str):
    return [float(line.split(", ")[1]) for line in data_str.split("\n")]


def poisson_plot(ys, ys2, p=0.025, kw2={}, extra_code=""):
    yerr = np.vstack([[y - st.poisson.ppf(p, y) for y in ys.values()],
                      [st.poisson.ppf(1 - p, y) - y for y in ys.values()]])
    fig, ax = plt.subplots(1, 1)
    plt.errorbar(ys.keys(), ys.values(), yerr=yerr, fmt='o-',
                 elinewidth=1,
                 label='trend', lw=0.5)
    plt.plot(list(ys2), list(ys2.values()), 'x-', c='orange', lw=0.5, **kw2)
    ax.legend(loc='best', frameon=False)
    plt.grid(True, axis='y', alpha=0.3)
    exec(extra_code)
    plt.show()
    return fig


def linspace_step(linspace: np.ndarray) -> float:
    return (linspace[-1] - linspace[0]) / (len(linspace) - 1)


def central_credible_interval(param_space: np.ndarray, pdf: np.ndarray, alpha: float) -> Tuple[Real, Real]:
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
