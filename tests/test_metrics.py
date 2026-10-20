import numpy as np
import pytest
from scipy import integrate, stats

from src.scoring import metrics


@pytest.mark.parametrize("y,mu,sigma", [(0.0, 0.0, 1.0), (0.7, -0.2, 0.5), (-3.0, 1.0, 2.0)])
def test_crps_gaussian_matches_numerical_integral(y, mu, sigma):
    # CRPS = integral of (F(x) - 1{x >= y})^2 dx
    f = lambda x: (stats.norm.cdf(x, mu, sigma) - (x >= y)) ** 2
    lo, hi = mu - 12 * sigma, mu + 12 * sigma
    numeric = integrate.quad(f, lo, y, limit=200)[0] + integrate.quad(f, y, hi, limit=200)[0]
    assert metrics.crps_gaussian(y, mu, sigma) == pytest.approx(numeric, rel=1e-6)


def test_crps_sample_converges_to_gaussian():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.3, 0.8, size=200_000)
    assert metrics.crps_sample(0.9, draws) == pytest.approx(metrics.crps_gaussian(0.9, 0.3, 0.8), rel=5e-3)


def test_crps_sample_vectorised_over_items():
    rng = np.random.default_rng(1)
    draws = rng.normal(0, 1, size=(3, 50_000))
    out = metrics.crps_sample(np.array([0.0, 1.0, -2.0]), draws)
    expected = metrics.crps_gaussian(np.array([0.0, 1.0, -2.0]), 0, 1)
    np.testing.assert_allclose(out, expected, rtol=1e-2)


def test_calibrated_gaussian_has_nominal_coverage():
    rng = np.random.default_rng(2)
    mu = rng.normal(size=20_000)
    sigma = rng.uniform(0.2, 2.0, size=20_000)
    y = rng.normal(mu, sigma)
    s = metrics.score_gaussian(y, mu, sigma)
    assert s["coverage_50"] == pytest.approx(0.5, abs=0.015)
    assert s["coverage_90"] == pytest.approx(0.9, abs=0.01)


def test_skill_score_sign():
    assert metrics.skill(0.5, 1.0) == pytest.approx(0.5)
    assert metrics.skill(2.0, 1.0) < 0


def test_crps_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        metrics.crps_gaussian(0, 0, 0)
