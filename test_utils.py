import numpy as np
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
    assert np.allclose(true_lb, lb, atol=1e-5, rtol=0), "Lower bound diff: {:.2e}".format(lb - true_lb)
    assert np.allclose(true_ub, ub, atol=1e-5, rtol=0), "Upper bound diff: {:.2e}".format(ub - true_ub)
