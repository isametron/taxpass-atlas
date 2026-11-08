"""Item features shared by the historical estimates and the Japan 2027 targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import episode
from src.paths import DATA_PROCESSED, REPORTS

EPISODES = ("japan_2019", "singapore_2023", "singapore_2024", "india_gst2")
BROAD = ("food", "food_away", "durable", "service")     # nondurable goods are the reference class
_INDIA_DURABLE = ("6.1.01.1", "6.1.01.2", "6.1.01.3", "6.1.01.4", "6.1.01.5", "6.1.03.1", "6.1.04.1")
_SG_DURABLE = ("1.04.1", "1.04.2", "1.04.4", "1.04.5", "1.06.1.1", "1.06.1.2", "1.07.1", "1.08.1", "1.08.2", "1.10.3")
_SG_NONDURABLE = ("1.02", "1.05.1", "1.06.1.5", "1.06.1.6", "1.08.3.2", "1.09.3", "1.10.2", "1.04.6.1")


def broad_japan(gs_code: int, durable_code: int) -> str:
    if gs_code in (1, 2, 3):
        return "food"
    if gs_code in (9, 16):
        return "food_away"
    if gs_code <= 8:
        return "durable" if durable_code == 1 else "nondurable"
    return "service"


def broad_singapore(code: str) -> str:
    if code.startswith("1.01"):
        return "food"
    if code.startswith("1.11"):
        return "food_away"
    if code.startswith(_SG_DURABLE):
        return "durable"
    if code.startswith(_SG_NONDURABLE):
        return "nondurable"
    return "service"


def broad_india(code: str) -> str:
    if code.startswith("1."):
        return "food"
    if code.startswith(_INDIA_DURABLE):
        return "durable"
    if code.startswith(("6.1.04.2", "6.1.06.3", "6.1.05.2")):
        return "service"
    return "nondurable"


def stickiness(levels: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Share of months in [start, end] whose index did not change: a menu-cost proxy."""
    d = levels.diff().loc[pd.Period(start, "M"):pd.Period(end, "M")]
    return (d.abs() < 1e-4).sum() / d.notna().sum()


def charm_share(retail: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Share of months whose Tokyo average yen price ends in 8 or 9 (by item)."""
    r = retail[(retail.month >= start) & (retail.month <= end)]
    last = np.floor(r.value + 0.5).astype(int) % 10
    return last.isin([8, 9]).groupby(r.item_code.str[-4:]).mean()


def load_retail() -> pd.DataFrame | None:
    path = DATA_PROCESSED / "japan_retail.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _panel_levels(name: str) -> pd.DataFrame:
    from src.passthrough.estimator import wide_log_levels
    panel = pd.read_parquet(DATA_PROCESSED / f"panel_{name}.parquet")
    return wide_log_levels(panel[panel.role != "excluded"])


def training_table() -> pd.DataFrame:
    """One row per treated item per episode, with rho, its placebo se and raw features."""
    info = pd.read_parquet(DATA_PROCESSED / "japan_items_2025.parquet").set_index("code")
    retail = load_retail()
    frames = []
    for name in EPISODES:
        ep = episode(name)
        d = pd.read_csv(REPORTS / "estimates" / f"{name}_items.csv", dtype={"item_id": str})
        d = d[d.rho.notna() & d.se.notna()].copy()
        d["episode"] = name
        d["country"] = ep.country
        if ep.country == "JP":
            d["broad"] = [broad_japan(info.gs_code.get(c, 6), info.durable_code.get(c, 0)) for c in d.item_id]
        elif ep.country == "SG":
            d["broad"] = d.item_id.map(broad_singapore)
        else:
            d["broad"] = d.item_id.map(broad_india)
        d["cut"] = float(ep.direction == "cut")
        d["display"] = float(ep.tax_inclusive_display_required)
        d["tax_size"] = np.log(100 * d.dlog_tax.abs())
        levels = _panel_levels(name)
        d["sticky"] = d.item_id.map(stickiness(levels, ep.pre_window[0], ep.pre_window[1]))
        if ep.country == "JP" and retail is not None:
            d["charm"] = d.item_id.map(charm_share(retail, "2018-01", "2019-09"))
        else:
            d["charm"] = np.nan
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


class FeatureMap:
    """Design-matrix builder, standardised on the training rows only."""

    columns = ("intercept", *[f"broad_{b}" for b in BROAD], "cut", "display", "tax_size", "sticky", "charm")

    def __init__(self, train: pd.DataFrame):
        self.scale = {c: (train[c].mean(), train[c].std(ddof=1) or 1.0) for c in ("tax_size", "sticky", "charm")}

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        out = [np.ones(len(df))]
        out += [(df.broad == b).to_numpy(float) for b in BROAD]
        out += [df.cut.to_numpy(float), df.display.to_numpy(float)]
        for c in ("tax_size", "sticky", "charm"):
            mean, sd = self.scale[c]
            out.append(np.nan_to_num((df[c].to_numpy(float) - mean) / sd))   # missing -> training mean
        return np.column_stack(out)
