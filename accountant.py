"""Pure-numpy RDP accountant for the Sampled Gaussian Mechanism (DP-SGD).

Use it BEFORE renting a GPU to see what epsilon your (sigma, batch size, steps)
gives. Integer Renyi orders only (Mironov et al., 2019), with the tighter
RDP->(eps, delta) conversion of Balle et al. (2020) (the one Opacus uses).
Final numbers for the thesis should come from Opacus' own accountant
(final_epsilon.py cross-checks both).
"""
import math
import numpy as np
from scipy.special import gammaln, logsumexp

ORDERS = list(range(2, 257))

def _log_binom(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def rdp_sgm(q, sigma, alpha):
    """RDP of order `alpha` (int >= 2) for ONE step of the sampled Gaussian mechanism."""
    if q == 0:
        return 0.0
    if q >= 1.0:
        return alpha / (2 * sigma ** 2)
    ks = np.arange(alpha + 1)
    log_terms = (_log_binom(alpha, ks) + (alpha - ks) * math.log1p(-q) + ks * math.log(q)
                 + (ks ** 2 - ks) / (2 * sigma ** 2))
    return float(logsumexp(log_terms)) / (alpha - 1)

def epsilon(q, sigma, steps, delta, orders=ORDERS):
    """(epsilon, best_order) after `steps` DP-SGD steps at sampling rate q."""
    best = (float('inf'), None)
    for a in orders:
        rdp = steps * rdp_sgm(q, sigma, a)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        if eps < best[0]:
            best = (eps, a)
    return best

def epsilon_from_history(history, delta, orders=ORDERS):
    """history = list of (sigma, q, steps) segments (e.g. one per FL round)."""
    best = (float('inf'), None)
    for a in orders:
        rdp = sum(steps * rdp_sgm(q, s, a) for (s, q, steps) in history)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        if eps < best[0]:
            best = (eps, a)
    return best

def find_sigma(q, steps, delta, target_eps, lo=0.3, hi=20.0, tol=1e-3):
    """Smallest sigma whose epsilon <= target_eps (bisection)."""
    for _ in range(60):
        mid = (lo + hi) / 2
        if epsilon(q, mid, steps, delta)[0] > target_eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi
