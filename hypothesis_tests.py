from poibin.poibin import PoiBin
import numpy as np
from scipy.stats import norm, chi2

def geom_r_test(r):
    log_r = np.log(r)
    x = np.mean(log_r)
    s = np.std(log_r)
    z = x / (s/np.sqrt(len(r)))
    return z, 2 * min([1 - norm.cdf(z),norm.cdf(z)])

def lin_r_test(o,outcome):
    linear_r = o * outcome - 1
    x = np.mean(linear_r)
    s = np.std(linear_r)
    z = x / (s/np.sqrt(len(o)))
    return z, 2 * min([1 - norm.cdf(z),norm.cdf(z)])

def poi_bin_test(p,w):
    pb = PoiBin(p)
    p_lower = pb.cdf(w)
    p_upper = 1 - p_lower
    return w, 2 * min([p_lower, p_upper])

def brier_test(p, outcome, Z):
    o = outcome
    bs = np.mean((p-o)**2)

    e_bs = np.mean(p - p**2)

    var_bs = np.sum(p*(1-p)**4+(1-p)*p**4)
    term = (p - p**2)
    var_bs += np.sum(term[:, None] * term[None, :]) - np.sum(term**2)
    var_bs /= len(p)**2
    var_bs -= e_bs**2
    z = (bs - e_bs) / np.sqrt(var_bs)
    pval = 2 * np.min([norm.cdf(z),1-norm.cdf(z)])
    return z, pval, 1 * ((e_bs - Z * np.sqrt(var_bs) > 0) and (e_bs + Z * np.sqrt(var_bs) < 1))

def pcs_test(p, outcome, num_bins=10):
    o = outcome
    percentiles = np.linspace(0, 100, num_bins+1)
    bin_edges = np.percentile(p, percentiles)
    bin_edges = np.unique(bin_edges)
    p_bins = np.digitize(p, bin_edges, right=True) - 1

    p_expected = np.zeros(num_bins)
    p_observed = np.zeros(num_bins)

    for i in range(num_bins):
        p_indices = (p_bins == i)
        if np.any(p_indices):
            p_expected[i] = np.sum(p[p_indices])
            p_observed[i] = np.sum(o[p_indices])

    test_statistic = np.sum((p_observed - p_expected)**2/p_expected)
    return test_statistic, 1 - chi2.cdf(test_statistic, num_bins - 1)