import numpy as np
import pandas as pd
import pytest

from src.passthrough import estimator as est, inference as inf

P = lambda s: pd.Period(s, "M")
WIN = est.Window(P("2019-09"), P("2019-10"), (0, 1, 2), 2, P("2017-09"), P("2019-08"))
MONTHS = pd.period_range("2015-01", "2020-01", freq="M")


def simulate(rho, drift=0.0, noise=0.0, n_treated=40, n_control=40, tau=0.02, shock=0.0, seed=0, cost=None):
    """Log prices with item trends and seasonality; treated items step up at the event."""
    rng = np.random.default_rng(seed)
    n = n_treated + n_control
    ids = [f"t{i}" for i in range(n_treated)] + [f"c{i}" for i in range(n_control)]
    t = np.arange(len(MONTHS))
    season = rng.normal(0, 0.005, size=(n, 12))
    y = np.empty((len(MONTHS), n))
    for j in range(n):
        path = drift * (1 + j / n) * t + np.cumsum(season[j, MONTHS.month - 1])
        y[:, j] = path + rng.normal(0, noise, len(t)) if noise else path
    post = MONTHS >= WIN.event
    y[np.ix_(post, range(n_treated))] += rho * tau
    y[post] += shock                     # common non-tax shock hits every item
    if cost is not None:
        y[:, :] += 0.5 * cost.reindex(MONTHS).to_numpy()[:, None]
    levels = pd.DataFrame(y, index=MONTHS, columns=ids)
    is_control = pd.Series([False] * n_treated + [True] * n_control, index=ids)
    category = pd.Series("a", index=ids)
    taus = pd.Series([tau] * n_treated + [0.0] * n_control, index=ids)
    return levels, is_control, category, taus


@pytest.mark.parametrize("rho,drift", [(1.0, 0.0), (1.0, 0.003), (0.4, 0.003), (0.0, 0.0), (-0.2, 0.001)])
def test_recovers_known_rho(rho, drift):
    levels, ctrl, cat, tau = simulate(rho, drift=drift)
    net = est.net_of_controls(est.deviations(levels, WIN), ctrl, cat)
    got = est.headline(net, WIN)[~ctrl] / tau[~ctrl]
    assert got.to_numpy() == pytest.approx(rho, abs=1e-9)


def test_common_shock_removed_by_controls_only_with_did():
    levels, ctrl, cat, tau = simulate(0.5, shock=0.01)
    dev = est.deviations(levels, WIN)
    with_did = est.headline(est.net_of_controls(dev, ctrl, cat), WIN)[~ctrl] / tau[~ctrl]
    without = est.headline(est.net_of_controls(dev, ctrl, cat, est.Spec(did=False)), WIN)[~ctrl] / tau[~ctrl]
    assert with_did.to_numpy() == pytest.approx(0.5, abs=1e-9)
    assert without.to_numpy() == pytest.approx(0.5 + 0.01 / 0.02, abs=1e-9)


def test_cost_term_removes_cost_shock():
    rng = np.random.default_rng(3)
    c = pd.DataFrame({"g": np.cumsum(rng.normal(0, 0.01, len(MONTHS)))}, index=MONTHS)
    levels, ctrl, _, tau = simulate(0.5, cost=c["g"])
    group = pd.Series("g", index=levels.columns)
    with_cost = est.deviations(levels, WIN, est.Spec(), c, group)
    plain = est.deviations(levels, WIN, est.Spec(cost=False))
    target = (0.5 * tau[~ctrl]).to_numpy()
    assert np.abs(with_cost.loc[~ctrl, 0] - target).max() < np.abs(plain.loc[~ctrl, 0] - target).min() + 1e-9
    assert with_cost.loc[~ctrl, 0].to_numpy() == pytest.approx(target, abs=1e-3)


def test_uses_only_pre_window_for_trend():
    """A huge move after the pre-window must not alter the fitted trend."""
    levels, ctrl, cat, tau = simulate(1.0, drift=0.002)
    bumped = levels.copy()
    bumped.loc[P("2019-12"):] += 0.5
    a = est.deviations(levels, WIN)[0]
    b = est.deviations(bumped, WIN)[0]
    assert a.to_numpy() == pytest.approx(b.to_numpy(), abs=1e-12)


def test_excluded_months_ignored_in_trend():
    win = est.Window(WIN.base, WIN.event, WIN.horizons, WIN.headline_h, WIN.pre_start, WIN.pre_end,
                     exclude=((P("2018-03"), P("2018-05")),))
    levels, *_ = simulate(1.0, drift=0.001, seed=4)
    bumped = levels.copy()
    bumped.loc[P("2018-03"):P("2018-05")] += 0.3
    pd.testing.assert_frame_equal(est.deviations(levels, win), est.deviations(bumped, win))


def test_item_with_too_little_history_is_nan():
    levels, *_ = simulate(1.0)
    levels.loc[:P("2019-04"), "t0"] = np.nan
    assert est.deviations(levels, WIN).loc["t0"].isna().all()


def test_placebo_bases_respect_data_and_exclusions():
    first = P("2015-01")
    bases = inf.placebo_bases(WIN, first)
    assert bases[0] == first + 24 and bases[-1] == P("2019-06")   # end = base + 3 months <= real base
    out = inf.placebo_bases(WIN, first, ((P("2018-01"), P("2018-02")),))
    assert all(b > P("2018-01") + 24 or b < P("2017-11") for b in out) and len(out) < len(bases)


def test_placebo_interval_covers_estimate_scale_and_needs_dates():
    draws = np.random.default_rng(1).normal(0, 0.01, 200)
    r = inf.placebo_interval(0.5, draws, scale=0.02)
    assert r.se == pytest.approx(0.5, rel=0.15)
    assert r.low < 0.5 < r.high
    assert np.isnan(inf.placebo_interval(0.5, draws[:5]).se)


def test_placebo_noise_gives_nominal_coverage_on_simulated_items():
    hits, tau = [], 0.02
    for seed in range(20):
        levels, ctrl, cat, taus = simulate(0.6, drift=0.001, noise=0.003, n_treated=10, seed=seed)
        bases = inf.placebo_bases(WIN, P("2015-01"))
        pl = inf.placebo_matrix(levels, WIN, bases, ctrl, cat)
        net = est.net_of_controls(est.deviations(levels, WIN), ctrl, cat)
        y = est.headline(net, WIN)
        for item in y.index[~ctrl]:
            r = inf.placebo_interval(y[item] / tau, pl.loc[item].to_numpy(), tau)
            hits.append(r.low <= 0.6 <= r.high)
    assert 0.85 <= np.mean(hits) <= 1.0


def test_wild_interval_nominal_coverage():
    rng = np.random.default_rng(0)
    hits = []
    for rep in range(400):
        g = np.repeat(np.arange(30), 8)
        tau = rng.choice([-0.1, -0.06, 0.02], g.size)
        shock = rng.normal(0, 0.01, 30)[g]
        y = 0.7 * tau + shock + rng.normal(0, 0.01, g.size)
        lo, hi = inf.wild_interval(y, tau, g, reps=399, seed=rep)
        hits.append(lo <= 0.7 <= hi)
    assert np.mean(hits) == pytest.approx(0.95, abs=0.03)


def test_wild_interval_needs_enough_clusters():
    y, tau = np.ones(6), np.ones(6)
    assert np.isnan(inf.wild_interval(y, tau, np.array([0, 0, 1, 1, 2, 2]))[0])


def test_ols_through_origin_handles_mixed_tax_sizes_and_gaps():
    tau = np.array([-0.1, -0.05, np.nan])
    y = 0.6 * np.nan_to_num(tau) + np.array([0, 0, np.nan])
    assert est.ols_through_origin(y, tau) == pytest.approx(0.6)


def test_ols_through_origin_weights_downplay_noisy_items():
    tau = np.array([0.1, 0.1, 0.1])
    y = np.array([0.05, 0.05, 5.0])
    assert est.ols_through_origin(y, tau, np.array([1.0, 1.0, 0.0])) == pytest.approx(0.5)
    assert est.ols_through_origin(y, tau) == pytest.approx(5.1 / 0.3)


def test_precision_weights_floor_at_median_variance():
    placebo = pd.DataFrame({"d1": [0.0, 1.0, 10.0], "d2": [1e-9, -1.0, -10.0]}, index=["still", "mid", "wild"])
    w = inf.precision_weights(placebo)
    assert w["still"] == pytest.approx(w["mid"])    # the near-constant item gets no extra weight
    assert w["wild"] < w["mid"]
