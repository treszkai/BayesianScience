from utils import *
import matplotlib.pyplot as plt
from textwrap import wrap

logger = logging.getLogger('BayesianScience')


def plot_hdi_line(hdi_min, hdi_max, ax=None, horizontal=True):
    if ax is None:
        ax = plt.gca()

    if horizontal:
        line_xys = ([hdi_min, hdi_max], [0, 0])
    else:
        line_xys = ([0, 0], [hdi_min, hdi_max])

    hdi_line, = ax.plot(*line_xys,
                        lw=5.0,
                        color='k',
                        label=f'95% HDI: [{hdi_min:.3f}, {hdi_max:.3f}]')
    hdi_line.set_clip_on(False)
    

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
                   posterior,
                   paper_title=None,
                   alpha=0.05,
                   legend=True):

    ax = plt.gca()
    rr_space = param_space + 1
    lb, ub = rr_space[[0, -1]]
    ci_lb, ci_ub = central_credible_interval(rr_space, posterior, alpha)

    logger.info(f'{paper_title}, {ci_lb:.4f}, {ci_ub:.4f}')

    plt.plot(rr_space, posterior, label='Posterior probability')
    plt.plot(rr_space, prior, c='red', lw=1, alpha=0.7, label='Prior probability')

    ax.set_xlim(lb, ub)
    ax.set_ylim(-0.01)

    ax.axvspan(xmin=ci_lb, xmax=ci_ub, lw=1, alpha=0.1,
               label=f'{round(100 * (1 - alpha))}% central credible interval')

    if legend:
        ax.legend(loc='upper right', frameon=True)

    ax.set_title("\n".join(wrap(f"Posterior probability after {paper_title} paper", 60)))

    ax.set_xlabel("Risk ratio on Monday")
    ax.set_ylabel("Posterior probability")
    ax.set_yticklabels([])

    plt.grid(True, alpha=0.5)

    return ci_lb, ci_ub


def plot_likelihood(param_space, log_likelihood):
    plt.figure()
    plt.plot(param_space, np.exp(log_likelihood))
    plt.title("Likelihood of different parameters")
    plt.xlabel("Relative increase in AMI counts")
    plt.ylabel("Likelihood")
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_yticklabels(["0.0"])
    plt.gca().set_xlim(param_space[[0, -1]])
    plt.grid(True, alpha=0.5)
    plt.show()


def plot_results(param_space, prior, all_data, paper_id):
    ami_trend = all_data[paper_id]['trend']
    ami_obs = all_data[paper_id]['obs']

    log_likelihood = calc_log_likelihood(param_space, ami_trend, ami_obs)

    # plot study results with error bars
    plot_poissons_error(ami_trend, ami_obs, kw2=dict(label='actual'),
                        extra_code="plt.title('Observed and trend values with 95% error bars')");

    plot_likelihood(param_space, log_likelihood)

    posterior = calc_posterior(param_space,
                               prior,
                               log_likelihood)

    plt.figure()
    plot_posterior(param_space, prior, posterior, paper_title=paper_id)
    plt.savefig('figs/%s_posterior.svg' % paper_id)
    plt.show()
