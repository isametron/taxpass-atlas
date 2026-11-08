"""Item-level pass-through for one tax event.

For each item:  rho = (tax-attributable change in log price) / dlog(1 + tau).

The tax-attributable change is the cumulative log change from the base month to
event + h, minus
  1. the item's own seasonal trend, fitted on the pre-window (month-of-year means of
     monthly log changes),
  2. a cost-control term (group-level beta on tax-free producer prices), and
  3. the median of the same quantity across untreated items in the item's category,
     or across all untreated items when the category has too few.

The headline is the average over h = 0..headline_h.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_PRE_CHANGES = 12


@dataclass(frozen=True)
class Spec:
    seasonal: bool = True
    did: bool = True
    cost: bool = True
    min_controls: int = 3


@dataclass(frozen=True)
class Window:
    base: pd.Period
    event: pd.Period
    horizons: tuple[int, ...]
    headline_h: int
    pre_start: pd.Period
    pre_end: pd.Period
    exclude: tuple[tuple[pd.Period, pd.Period], ...] = ()

    @property
    def end(self) -> pd.Period:
        return self.event + max(self.horizons)

    def shifted(self, base: pd.Period) -> "Window":
        """The same window moved to a placebo base month, without the real exclusions."""
        step = (base - self.base).n
        return Window(base, self.event + step, self.horizons, self.headline_h,
                      self.pre_start + step, self.pre_end + step)

    def headline_horizons(self) -> list[int]:
        return [h for h in self.horizons if h <= self.headline_h]


def wide_log_levels(panel: pd.DataFrame) -> pd.DataFrame:
    """Months x items matrix of log index levels on a complete monthly index."""
    wide = panel.pivot(index="month", columns="item_id", values="log_index")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    full = pd.period_range(wide.index.min(), wide.index.max(), freq="M")
    return wide.reindex(full)


def _kept_change_months(index: pd.PeriodIndex, win: Window) -> np.ndarray:
    """Months whose change (t-1 to t) lies wholly inside the pre-window and outside exclusions."""
    keep = (index > win.pre_start) & (index <= win.pre_end)
    for a, b in win.exclude:
        keep &= ~((index >= a) & (index <= b))           # t excluded
        keep &= ~(((index - 1) >= a) & ((index - 1) <= b))  # t-1 excluded
    return keep


def _seasonal_means(changes: pd.DataFrame, keep: np.ndarray, seasonal: bool) -> pd.DataFrame:
    """Calendar month (1..12) x series expected monthly change. Empty months use the overall mean."""
    kept = changes[keep]
    overall = kept.mean()
    months = range(1, 13)
    if not seasonal:
        return pd.DataFrame({c: overall[c] for c in changes.columns}, index=months)
    by_month = kept.groupby(kept.index.month).mean().reindex(months)
    return by_month.fillna(overall)


def _cost_betas(changes: pd.DataFrame, expected: pd.DataFrame, keep: np.ndarray,
                cost_changes: pd.DataFrame, cost_expected: pd.DataFrame,
                cost_group: pd.Series) -> dict[str, float]:
    """Pooled pre-window OLS slope of item surprises on cost surprises, by cost group."""
    idx = changes.index[keep]
    cal = idx.month
    betas = {}
    for group in cost_expected.columns:
        items = cost_group.index[cost_group == group].intersection(changes.columns)
        c = cost_changes.loc[idx, group].to_numpy() - cost_expected[group].reindex(cal).to_numpy()
        r = changes.loc[idx, items].to_numpy() - expected[items].reindex(cal).to_numpy()
        valid = ~np.isnan(r) & ~np.isnan(np.tile(c[:, None], (1, r.shape[1])))
        cc = np.tile(c[:, None], (1, r.shape[1]))
        denom = (cc[valid] ** 2).sum()
        betas[group] = float((r[valid] * cc[valid]).sum() / denom) if denom > 0 else 0.0
    return betas


def deviations(levels: pd.DataFrame, win: Window, spec: Spec = Spec(),
               cost: pd.DataFrame | None = None, cost_group: pd.Series | None = None) -> pd.DataFrame:
    """Items x horizons: cumulative change minus own seasonal trend and cost term."""
    changes = levels.diff()
    keep = _kept_change_months(levels.index, win)
    expected = _seasonal_means(changes, keep, spec.seasonal)
    enough = changes[keep].notna().sum() >= MIN_PRE_CHANGES

    use_cost = spec.cost and cost is not None and cost_group is not None
    if use_cost:
        cost_changes = cost.reindex(levels.index).diff()
        cost_expected = _seasonal_means(cost_changes, keep, spec.seasonal)
        betas = _cost_betas(changes, expected, keep, cost_changes, cost_expected, cost_group)

    out = {}
    base_level = levels.loc[win.base]
    for h in win.horizons:
        t = win.event + h
        months = pd.period_range(win.base + 1, t, freq="M")
        trend = expected.reindex(months.month).sum()
        dev = levels.loc[t] - base_level - trend
        if use_cost:
            surprise = (cost_changes.loc[months] - cost_expected.reindex(months.month).to_numpy()).sum()
            adj = cost_group.map({g: betas[g] * surprise[g] for g in betas})
            dev = dev - adj.reindex(dev.index).fillna(0.0)
        out[h] = dev.where(enough)
    return pd.DataFrame(out)


def net_of_controls(dev: pd.DataFrame, is_control: pd.Series, category: pd.Series,
                    spec: Spec = Spec()) -> pd.DataFrame:
    """Subtract the median untreated deviation, by category when it has enough controls."""
    if not spec.did:
        return dev.copy()
    is_control = is_control.reindex(dev.index).fillna(False).astype(bool)
    category = category.reindex(dev.index)
    pooled = dev[is_control].median()
    out = dev.astype(float).copy()
    for cat, members in category.groupby(category).groups.items():
        ctrl = dev.loc[members][is_control.loc[members]]
        counts = ctrl.notna().sum()
        adj = ctrl.median().where(counts >= spec.min_controls, pooled)
        out.loc[members] = dev.loc[members].to_numpy() - adj.to_numpy()
    return out


def headline(net: pd.DataFrame, win: Window) -> pd.Series:
    """Average over h = 0..headline_h; NaN unless every horizon is available."""
    return net[win.headline_horizons()].mean(axis=1, skipna=False)


def ols_through_origin(y: np.ndarray, tau: np.ndarray, weights: np.ndarray | None = None) -> float:
    """rho minimising sum w*(y - rho*tau)^2, the pooled pass-through for a set of items."""
    w = np.ones_like(tau) if weights is None else weights
    ok = ~np.isnan(y) & ~np.isnan(tau) & ~np.isnan(w) & (w > 0)
    return float((w[ok] * tau[ok] * y[ok]).sum() / (w[ok] * tau[ok] ** 2).sum()) if ok.any() else float("nan")
