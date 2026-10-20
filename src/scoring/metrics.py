"""Forecast scoring: CRPS, MAE, interval coverage, PIT and skill scores.

All functions are vectorised over items. Gaussian forecasts are given as (mu, sigma);
non-Gaussian forecasts (mixtures, simulated aggregates) are given as sample arrays.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def crps_gaussian(y, mu, sigma):
    """Closed-form CRPS of N(mu, sigma^2) at y (Gneiting et al. 2005). Lower is better."""
    y, mu, sigma = np.broadcast_arrays(np.asarray(y, float), np.asarray(mu, float), np.asarray(sigma, float))
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - _INV_SQRT_PI)


def crps_sample(y, samples):
    """CRPS from forecast samples: E|X - y| - 0.5 E|X - X'|.

    `samples` has shape (n_items, n_draws) or (n_draws,) for a single y.
    Uses the sorted-sample identity, O(n log n) per item.
    """
    x = np.sort(np.atleast_2d(np.asarray(samples, float)), axis=1)
    y = np.atleast_1d(np.asarray(y, float))
    n = x.shape[1]
    term1 = np.mean(np.abs(x - y[:, None]), axis=1)
    i = np.arange(1, n + 1)
    term2 = np.sum((2 * i - n - 1) * x, axis=1) / (n * n)  # = 0.5 E|X - X'|
    out = term1 - term2
    return out if out.size > 1 else out[0]


def central_interval(mu, sigma, level):
    """Central `level` predictive interval of N(mu, sigma^2)."""
    z = stats.norm.ppf(0.5 + level / 2)
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    return mu - z * sigma, mu + z * sigma


def coverage(y, lo, hi):
    """Share of y inside [lo, hi]."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def pit_gaussian(y, mu, sigma):
    """Probability integral transform values; uniform if the forecast is calibrated."""
    return stats.norm.cdf((np.asarray(y, float) - np.asarray(mu, float)) / np.asarray(sigma, float))


def mae(y, point):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(point, float))))


def skill(score_model: float, score_baseline: float) -> float:
    """Skill score 1 - S_model / S_baseline. Positive means the model beats the baseline."""
    return 1.0 - score_model / score_baseline


def score_gaussian(y, mu, sigma, levels=(0.5, 0.9)) -> dict[str, float]:
    """Summary scores for a set of Gaussian item forecasts."""
    out = {
        "n": int(np.size(y)),
        "crps": float(np.mean(crps_gaussian(y, mu, sigma))),
        "mae": mae(y, mu),
    }
    for lv in levels:
        lo, hi = central_interval(mu, sigma, lv)
        out[f"coverage_{int(lv * 100)}"] = coverage(y, lo, hi)
    return out
