import numpy as np
import pandas as pd
import pytest

from src.model import features as ft, hierarchical as hm


def _sim(seed=0, n=400, beta=0.4, sigma=0.3, tau=0.2, n_ep=6):
    rng = np.random.default_rng(seed)
    eps = rng.integers(0, n_ep, n)
    x = rng.normal(size=n)
    eta = rng.normal(0, sigma, n_ep)[eps]
    rho = 0.6 + beta * x + eta + rng.normal(0, tau, n)
    se = rng.uniform(0.05, 0.3, n)
    return np.column_stack([np.ones(n), x]), rho + rng.normal(0, se), se, [f"e{e}" for e in eps]


def test_recovers_coefficients_and_tau():
    X, y, se, eps = _sim()
    fit = hm.fit(X, y, se, eps, ("intercept", "x"))
    assert fit.coefficients()["x"][0] == pytest.approx(0.4, abs=0.05)
    assert fit.tau == pytest.approx(0.2, abs=0.06)


def test_noisy_items_get_little_influence():
    X, y, se, eps = _sim(n=200)
    base = hm.fit(X, y, se, eps, ("intercept", "x")).coefficients()["x"][0]
    X2, y2, se2 = np.vstack([X, [1.0, 2.0]]), np.append(y, 500.0), np.append(se, 1000.0)
    bumped = hm.fit(X2, y2, se2, eps + ["e0"], ("intercept", "x")).coefficients()["x"][0]
    assert bumped == pytest.approx(base, abs=0.02)


def test_new_episode_is_wider_than_known_episode():
    X, y, se, eps = _sim(sigma=0.4)
    fit = hm.fit(X, y, se, eps, ("intercept", "x"))
    _, known = fit.predict(X[:5], "e0")
    _, new = fit.predict(X[:5], None)
    assert (new > known).all()
    assert new[0] ** 2 == pytest.approx(known[0] ** 2 + fit.sigma ** 2, rel=0.25)


def test_unidentified_effect_stays_near_prior():
    n = 100
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(n), np.zeros(n)])          # the 'cut' column never varies
    fit = hm.fit(X, 0.8 + rng.normal(0, 0.1, n), np.full(n, 0.1), ["a"] * n, ("intercept", "cut"))
    m, s = fit.coefficients()["cut"]
    assert abs(m) < 1e-6 and s == pytest.approx(hm.EFFECT_PRIOR_SD, rel=0.01)


def test_predictive_interval_coverage_on_new_episode():
    hits = []
    for seed in range(40):
        X, y, se, eps = _sim(seed=seed, n=240, n_ep=8)
        keep = np.array([e != "e7" for e in eps])
        fit = hm.fit(X[keep], y[keep], se[keep], [e for e in eps if e != "e7"], ("intercept", "x"))
        mu, sd = fit.predict(X[~keep], None)
        total = np.sqrt(sd ** 2 + se[~keep] ** 2)
        hits.append(np.mean(np.abs(y[~keep] - mu) <= 1.645 * total))
    # Slightly under nominal: sigma is estimated from only a few simulated episodes.
    assert 0.75 <= np.mean(hits) <= 0.97


def test_sample_new_episode_matches_predictive_sd():
    X, y, se, eps = _sim()
    fit = hm.fit(X, y, se, eps, ("intercept", "x"))
    draws = fit.sample_new_episode(X[:3], 40000, seed=1)
    mu, sd = fit.predict(X[:3], None)
    assert draws.mean(axis=0) == pytest.approx(mu, abs=0.02)
    assert draws.std(axis=0) == pytest.approx(sd, rel=0.03)


def test_broad_classes():
    assert ft.broad_japan(3, 3) == "food" and ft.broad_japan(16, 0) == "food_away"
    assert ft.broad_japan(6, 1) == "durable" and ft.broad_japan(6, 3) == "nondurable" and ft.broad_japan(19, 0) == "service"
    assert ft.broad_singapore("1.01.1.3") == "food" and ft.broad_singapore("1.11.2") == "food_away"
    assert ft.broad_india("6.1.03.1.1.02.X") == "durable" and ft.broad_india("1.1.05.2.1.01.0") == "food"


def test_charm_share_counts_prices_ending_in_8_or_9():
    r = pd.DataFrame({"item_code": ["01001"] * 4, "month": ["2024-01", "2024-02", "2024-03", "2024-04"],
                      "value": [198.0, 299.4, 305.0, 500.0]})
    assert ft.charm_share(r, "2024-01", "2024-04")["1001"] == pytest.approx(0.5)
