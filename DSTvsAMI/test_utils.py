import numpy as np
import pytest
import scipy.stats as st

import utils


def test_central_credible_interval_cont():
    xs = np.linspace(0, 1, 100000)
    a, b = 2, 6  # arbitrary positive numbers
    dist = st.beta(a, b)
    pdf = dist.pdf(xs)
    alpha = 0.05
    lb, ub = utils.central_credible_interval(xs, pdf, alpha)
    true_lb, true_ub = dist.ppf([alpha / 2, 1 - alpha / 2])
    print(f"Calculated: [{lb:.5f}, {ub:.5f}]")
    print(f"True:       [{true_lb:.5f}, {true_ub:.5f}]")

    assert lb == pytest.approx(true_lb, abs=1e-4, rel=0)
    assert ub == pytest.approx(true_ub, abs=1e-4, rel=0)


def mix_norm_pdf(x_min, x_max):
    def mix_norm_pdf_generic(x, loc, scale, weight):
        return np.dot(weight, st.norm.pdf(x, loc, scale))

    loc = np.array([-1, 3])  # mean values
    scale = np.array([.5, .8])  # standard deviations
    weight = np.array([.4, .6])  # mixture probabilities

    xs = np.linspace(x_min, x_max, 200)
    ps = np.array([mix_norm_pdf_generic(x, loc, scale, weight) for x in xs])

    return xs, ps


def test_hdr():
    xs, ps = mix_norm_pdf(-3, 6)
    expected = [(-2, 0), (1.43, 4.55)]
    assert utils.highest_density_region(xs, ps, alpha=0.05) == \
           [pytest.approx(exp_sub, abs=0.1) for exp_sub in expected]

    xs = np.linspace(0, 10, 10000)
    ps = st.halfnorm.pdf(xs)
    assert utils.highest_density_region(xs, ps, alpha=0.05) == \
        [(0, pytest.approx(st.norm.ppf(0.975), abs=0.01))]

    xs = np.linspace(-10, 0, 10000)
    ps = st.halfnorm.pdf(-xs)
    assert utils.highest_density_region(xs, ps, alpha=0.05) == \
        [(pytest.approx(st.norm.ppf(0.025), abs=0.01), 0)]
