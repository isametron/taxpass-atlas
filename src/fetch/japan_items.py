"""Japan CPI 2025-base item information: national and Tokyo weights and goods/services class.

Source: Statistics Bureau, 2025-base CPI handbook, table IV-1 (品目情報一覧).

  python -m src.fetch.japan_items            download and parse
  python -m src.fetch.japan_items --offline  parse data/raw only
"""

from __future__ import annotations

import argparse

import openpyxl
import pandas as pd
import requests

from src.fetch.manifest import record
from src.paths import DATA_PROCESSED, DATA_RAW

URL = "https://www.stat.go.jp/data/cpi/2025/kaisetsu/zuhyou/4-1.xlsx"
RAW = DATA_RAW / "japan" / "cpi2025_item_info.xlsx"
TOTAL_WEIGHT = 3543757090       # national all-items weight printed in the table


def fetch() -> None:
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (taxpass-atlas research)"}, timeout=120)
    r.raise_for_status()
    RAW.write_bytes(r.content)
    record(RAW, URL)


def tidy(path=RAW) -> pd.DataFrame:
    """code, name, weight_national, weight_tokyo, gs_code (財・サービス分類), durable_code."""
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[1]
    rows = []
    for r in ws.iter_rows(min_row=6, values_only=True):
        if r[8] in (None, ""):
            continue
        rows.append({"code": str(r[8]).strip(), "name": str([c for c in r[:7] if c][-1]).strip(),
                     "weight_national": float(r[10]), "weight_tokyo": float(r[12]),
                     "gs_code": int(r[19]), "durable_code": int(r[20]) if r[20] not in (None, "", "-") else 0})
    df = pd.DataFrame(rows)
    if len(df) != 589 or df.weight_national.sum() != TOTAL_WEIGHT:
        raise ValueError(f"unexpected item table: {len(df)} items, weight sum {df.weight_national.sum()}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if not args.offline:
        fetch()
    df = tidy()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED / "japan_items_2025.parquet", index=False)
    print(f"{len(df)} items, weights sum to the published total")


if __name__ == "__main__":
    main()
