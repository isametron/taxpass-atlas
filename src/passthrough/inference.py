"""Uncertainty for pass-through estimates.

Placebo: re-run the whole estimator at earlier dates with no tax change, each with its
own pre-window strictly before it. The spread of those outcomes is the noise an estimate
carries, including shocks common to all items. Items have one observation per event, so
this is the only item-level interval available. Placebo windows overlap, so the number
of dates overstates the number of independent draws.

Wild bootstrap: percentile-t with Webb six-point weights. Episode-level estimates cluster
by category; category-level estimates resample items. It captures cross-item dispersion
but treats the control adjustment as known, so results report both intervals and the
headline interval is their union.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.passthrough import estimator as est

_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
MIN_PLACEBO_DATES = 12
MIN_WILD_UNITS = 4


def placebo_bases(win: est.Window, first_month: pd.Period,
                  exclude: tuple[tuple[pd.Period, pd.Period], ...] = ()) -> list[pd.Period]:
    """Placebo base months: every window fits in the data, ends by the real base month,
    and touches no excluded range. Pre-windows start at least at first_month."""
    lead = (win.base - win.pre_start).n
    tail = (win.end - win.base).n
    out = []
    for base in pd.period_range(first_month + lead, win.base - tail, freq="M"):
        start, end = base - lead, base + tail
        if not any(start <= b and a <= end for a, b in exclude):
            out.append(base)
    return out


def placebo_matrix(levels: pd.DataFrame, win: est.Window, bases: list[pd.Period], is_control: pd.Series,
                   category: pd.Series, spec: est.Spec = est.Spec(),
                   cost: pd.DataFrame | None = None, cost_group: pd.Series | None = None) -> pd.DataFrame:
    """Items x placebo dates: headline net deviation (log points) with no tax event."""
    rows = {}
    for base in bases:
        fake = win.shifted(base)
        dev = est.deviations(levels, fake, spec, cost, cost_group)
        rows[str(base)] = est.headline(est.net_of_controls(dev, is_control, category, spec), fake)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PlaceboResult:
    se: float
    low: float
    high: float
    n: int
    p: float


def placebo_interval(estimate: float, draws: np.ndarray, scale: float = 1.0, level: float = 0.95) -> PlaceboResult:
    """Interval for `estimate` from placebo outcomes `draws`, both divided by `scale` first."""
    x = draws[~np.isnan(draws)] / scale
    n = x.size
    if n < MIN_PLACEBO_DATES or np.isnan(estimate):
        return PlaceboResult(*[float("nan")] * 3, n, float("nan"))
    se = float(x.std(ddof=1))
    half = stats.t.ppf(0.5 + level / 2, n - 1) * se
    p = float((1 + (np.abs(x) >= abs(estimate)).sum()) / (n + 1))
    return PlaceboResult(se, estimate - half, estimate + half, n, p)


def precision_weights(placebo: pd.DataFrame) -> pd.Series:
    """Inverse placebo variance by item, floored at the median item's variance so that a
    few items whose index barely moves cannot dominate a pooled estimate."""
    sd = placebo.std(axis=1, ddof=1)
    return 1.0 / sd.clip(lower=sd.median()) ** 2


def pooled_placebo_draws(placebo: pd.DataFrame, tau: pd.Series, weights: pd.Series | None = None) -> np.ndarray:
    """Pooled pass-through a set of items would show at each placebo date."""
    t = tau.reindex(placebo.index).to_numpy()
    w = None if weights is None else weights.reindex(placebo.index).to_numpy()
    return np.array([est.ols_through_origin(placebo[c].to_numpy(), t, w) for c in placebo.columns])


def wild_interval(y: np.ndarray, tau: np.ndarray, clusters: np.ndarray, reps: int = 9999,
                  seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile-t interval for through-origin OLS rho, with Webb weights by cluster."""
    ok = ~np.isnan(y) & ~np.isnan(tau)
    y, tau = y[ok], tau[ok]
    groups, inverse = np.unique(clusters[ok], return_inverse=True)
    g = groups.size
    if g < MIN_WILD_UNITS:
        return float("nan"), float("nan")
    q = (tau ** 2).sum()
    rho = (tau * y).sum() / q
    score = np.bincount(inverse, tau * (y - rho * tau), g)
    tt = np.bincount(inverse, tau ** 2, g)
    adj = g / (g - 1)
    se = np.sqrt(adj * (score ** 2).sum()) / q
    if se == 0:
        return rho, rho
    rng = np.random.default_rng(seed)
    v = rng.choice(_WEBB, size=(reps, g))
    delta = (v @ score) / q
    boot_score = v * score - delta[:, None] * tt
    boot_se = np.sqrt(adj * (boot_score ** 2).sum(axis=1)) / q
    t = delta[boot_se > 0] / boot_se[boot_se > 0]
    lo, hi = np.quantile(t, [(1 - level) / 2, (1 + level) / 2])
    return float(rho - hi * se), float(rho - lo * se)
