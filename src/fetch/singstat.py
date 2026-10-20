"""Singapore CPI from the SingStat Table Builder API (no key needed).

The API pages by observation, not by series, so a series can span two pages; pages
are merged by series number.

Fetch:  python -m src.fetch.singstat
Parse only (from data/raw):  python -m src.fetch.singstat --offline
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import requests

from src.config import sources
from src.fetch.manifest import write_json
from src.paths import DATA_PROCESSED, DATA_RAW

RAW_DIR = DATA_RAW / "singapore"
HEADERS = {"User-Agent": "Mozilla/5.0 (taxpass-atlas research)", "Accept": "*/*"}
# "1.0 Food" aggregates 1.01 (food) and 1.11 (food serving), so numbering alone hides it.
AGGREGATES = {"1.0"}


def fetch_table(table_id: str) -> dict:
    cfg = sources()["singapore"]
    url = f"{cfg['tablebuilder_api']}/{table_id}"
    limit = cfg["page_limit"]
    series: dict[str, dict] = {}
    meta: dict = {}
    offset = 0
    while True:
        r = requests.get(url, params={"offset": offset, "limit": limit}, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()["Data"]
        meta = {k: v for k, v in data.items() if k != "row"}
        n = 0
        for row in data["row"]:
            s = series.setdefault(row["seriesNo"], {"name": row["rowText"], "uom": row.get("uoM"), "obs": {}})
            for c in row["columns"]:
                s["obs"][c["key"]] = c["value"]
                n += 1
        if n < limit:
            break
        offset += limit
    out = {"meta": meta, "series": series}
    write_json(RAW_DIR / f"{table_id}.json", out, url, {"limit": limit})
    return out


def tidy(raw: dict) -> pd.DataFrame:
    """Long panel: series_no, name, depth, is_leaf, month (YYYY-MM), index."""
    series = raw["series"]
    codes = list(series)
    rows = []
    for code, s in series.items():
        is_leaf = code not in AGGREGATES and not any(o.startswith(code + ".") for o in codes)
        for key, val in s["obs"].items():
            if val in (None, "", "na", "-"):
                continue
            rows.append({
                "series_no": code,
                "name": s["name"],
                "depth": code.count("."),
                "is_leaf": is_leaf,
                "month": pd.to_datetime(key, format="%Y %b").strftime("%Y-%m"),
                "index": float(val),
            })
    return pd.DataFrame(rows).sort_values(["series_no", "month"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="parse data/raw only, no network")
    args = ap.parse_args()
    table_id = sources()["singapore"]["cpi_monthly_2024_base"]
    if args.offline:
        raw = json.loads((RAW_DIR / f"{table_id}.json").read_text(encoding="utf-8"))
    else:
        raw = fetch_table(table_id)
    df = tidy(raw)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "singapore_cpi.parquet"
    df.to_parquet(out, index=False)
    print(f"{df.series_no.nunique()} series, {df.month.min()}..{df.month.max()} -> {out.name}")


if __name__ == "__main__":
    main()
