"""Pre-specified extraction of realised pass-through from Japan CPI.

realised rho_i = [dlog p_i(base -> Apr..Jun) - own seasonal trend - cost term] / dlog(1 + tau)

This is the formula in config/japan_2027_protocol.yaml. There is no difference-in-differences
step: the cut hits the whole food basket, and non-food items are affected by different shocks.
The same code runs on historical data for the fake-shock test and on 2027 data for scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import episode
from src.passthrough import estimator as est
from src.paths import DATA_PROCESSED

SPEC = est.Spec(did=False)
COST_GROUP = "beverages_and_foods"
NATIONAL, TOKYO = "00000", "13100"


def window() -> est.Window:
    ep = episode("japan_2027")
    P = lambda s: pd.Period(s, "M")
    return est.Window(P(ep.base_month), P(ep.event_month), ep.horizons, ep.headline_h, P(ep.pre_window[0]), P(ep.pre_window[1]))


def levels(area: str, path=None) -> pd.DataFrame:
    """Months x item codes: log index on the 2025 base (linked before 2025)."""
    v = pd.read_parquet(path or DATA_PROCESSED / "japan_cpi2025.parquet")
    v = v[(v.area == area) & (v.tab == "1") & (v.item_code >= "1000") & (v.value > 0)]
    wide = v.pivot(index="month", columns="item_code", values="value")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    return np.log(wide.reindex(pd.period_range(wide.index.min(), wide.index.max(), freq="M")))


def cost_series(path=None) -> pd.DataFrame:
    raw = pd.read_parquet(path or DATA_PROCESSED / "japan_cgpi.parquet")
    wide = raw[raw.series == COST_GROUP].pivot(index="month", columns="series", values="index")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    return np.log(wide)


def net_change(lv: pd.DataFrame, win: est.Window, cost: pd.DataFrame) -> pd.Series:
    """Headline tax-attributable log change by item (before dividing by the tax change)."""
    group = pd.Series(COST_GROUP, index=lv.columns)
    return est.headline(est.deviations(lv, win, SPEC, cost, group), win)


def realised_rho(lv: pd.DataFrame, win: est.Window, cost: pd.DataFrame, dlog_tax: float) -> pd.Series:
    return net_change(lv, win, cost) / dlog_tax
