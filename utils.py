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