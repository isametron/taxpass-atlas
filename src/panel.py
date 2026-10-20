"""Build estimation panels: one tidy table per episode in a common schema.

Columns:
  episode, item_id, item_name, category, month, log_index,
  role        treated | control | excluded
  reason      why an item is a control or excluded
  dlog_tax    ln((1+new)/(1+old)) for the item; 0 for controls

`category` is the clustering unit for the wild bootstrap (COICOP division or the
country's equivalent).

  python -m src.panel            build every episode whose inputs exist
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import dlog_tax, episode
from src.paths import CONCORDANCES, DATA_PROCESSED

COLUMNS = ["episode", "item_id", "item_name", "category", "month", "log_index",
           "role", "reason", "dlog_tax"]


def _window(ep_name: str, cpi: pd.DataFrame) -> pd.DataFrame:
    """Keep months from the pre-window start to the last horizon."""
    ep = episode(ep_name)
    last = (pd.Period(ep.event_month, "M") + max(ep.horizons)).strftime("%Y-%m")
    return cpi[(cpi.month >= ep.pre_window[0]) & (cpi.month <= last)]


def singapore(ep_name: str) -> pd.DataFrame:
    ep = episode(ep_name)
    cpi = pd.read_parquet(DATA_PROCESSED / "singapore_cpi.parquet")
    flags = pd.read_csv(CONCORDANCES / "singapore_gst_flags.csv", dtype=str).fillna("")
    cpi = _window(ep_name, cpi[cpi.is_leaf])
    df = cpi.merge(flags[["series_no", "gst_status", "reason", "administered", "low_registration"]],
                   on="series_no", how="inner")

    role = np.where(df.gst_status == "standard", "treated", "control")
    reason = df.reason.where(df.gst_status != "standard", "")
    excluded = (df.administered != "") | (df.low_registration != "")
    role = np.where(excluded, "excluded", role)
    reason = np.where(df.administered != "", df.administered,
                      np.where(df.low_registration != "", "below GST registration threshold: " + df.low_registration, reason))

    return pd.DataFrame({
        "episode": ep_name,
        "item_id": df.series_no,
        "item_name": df.name,
        "category": df.series_no.str.split(".").str[:2].str.join("."),
        "month": df.month,
        "log_index": np.log(df["index"]),
        "role": role,
        "reason": reason,
        "dlog_tax": np.where(role == "treated", ep.dlog("standard"), 0.0),
    })[COLUMNS]


def japan_2019() -> pd.DataFrame:
    """National 2020-base item indices; 10%-rate items treated, reduced-rate and exempt items as controls."""
    ep = episode("japan_2019")
    path = DATA_PROCESSED / "japan_cpi2020.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing: run `python -m src.fetch.estat cpi2020`")
    cpi = pd.read_parquet(path)
    classes = pd.read_parquet(DATA_PROCESSED / "japan_cpi2020_classes.parquet")
    # tab '1' is the index level in e-Stat CPI tables; confirm on first fetch.
    cpi = cpi[(cpi.area == "00000") & (cpi.tab == "1")]
    cpi = _window("japan_2019", cpi.rename(columns={"value": "index"}))
    flags = pd.read_csv(CONCORDANCES / "japan_rate_flags.csv",
                        dtype={"code_2025": str, "code_2020": str}).fillna({"administered": "", "exempt_reason": ""})
    flags = flags[flags.code_2020.notna()]
    df = cpi.merge(flags, left_on="item_code", right_on="code_2020", how="inner")
    names = classes[classes.class_id == "cat01"].set_index("code")["name"]

    d = np.array([dlog_tax(o, n) for o, n in zip(df.rate_2019_old, df.rate_2019_new)])
    role = np.where(d != 0, "treated", "control")
    reason = np.where(role == "control", df.rate_class, "")
    role = np.where(df.administered != "", "excluded", role)
    reason = np.where(df.administered != "", df.administered, reason)
    return pd.DataFrame({
        "episode": "japan_2019",
        "item_id": df.item_code,
        "item_name": df.item_code.map(names),
        "category": df.item_code.str[0],      # first digit of the item code = 10大費目-like block
        "month": df.month,
        "log_index": np.log(df["index"]),
        "role": role,
        "reason": reason,
        "dlog_tax": np.where(role == "treated", d, 0.0),
    })[COLUMNS]


def complete_items(df: pd.DataFrame, ep_name: str, min_pre_share: float = 0.8) -> pd.DataFrame:
    """Keep items observed at the base month and every horizon, and in most pre-window months.

    Months inside the episode's `exclude` ranges do not count toward the pre-window.
    """
    ep = episode(ep_name)
    post = [ep.base_month] + [(pd.Period(ep.event_month, "M") + h).strftime("%Y-%m") for h in ep.horizons]
    pre = pd.period_range(ep.pre_window[0], ep.pre_window[1], freq="M").strftime("%Y-%m")
    pre = [m for m in pre if not any(a <= m <= b for a, b in ep.exclude)]
    seen = df.groupby("item_id").month.agg(set)
    keep = seen[seen.map(lambda s: set(post) <= s and len(s & set(pre)) >= min_pre_share * len(pre))].index
    return df[df.item_id.isin(keep)]


BUILDERS = {
    "singapore_2023": lambda: singapore("singapore_2023"),
    "singapore_2024": lambda: singapore("singapore_2024"),
    "japan_2019": japan_2019,
    # india_gst2: needs the MoSPI item pull and an item-name match to the slab map.
}


def main() -> None:
    for name, build in BUILDERS.items():
        try:
            df = complete_items(build(), name)
        except FileNotFoundError as e:
            print(f"{name}: skipped ({e})")
            continue
        df.to_parquet(DATA_PROCESSED / f"panel_{name}.parquet", index=False)
        counts = df.drop_duplicates("item_id").role.value_counts().to_dict()
        print(f"{name}: {df.month.min()}..{df.month.max()}, items {counts}")


if __name__ == "__main__":
    main()
