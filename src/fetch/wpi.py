"""India Wholesale Price Index (base 2011-12), commodity level, from the Office of the
Economic Adviser. Used as the producer-cost control for the GST 2.0 estimates.

The file name changes every month (monthly_index_YYYYMM.xls), so the link is read from
the download page. No key needed.

  python -m src.fetch.wpi            download the latest file
  python -m src.fetch.wpi --offline  parse data/raw only
"""

from __future__ import annotations

import argparse
import re

import pandas as pd
import requests
import truststore

from src.config import sources
from src.fetch.manifest import record
from src.paths import DATA_PROCESSED, DATA_RAW

RAW = DATA_RAW / "india" / "wpi_monthly_index_1112.xls"
HEADERS = {"User-Agent": "Mozilla/5.0 (taxpass-atlas research)"}


def fetch() -> None:
    # eaindustry.nic.in omits an intermediate certificate. The OS trust store fills it in
    # (as browsers and curl do); certifi's bundle cannot. Verification stays on.
    truststore.inject_into_ssl()
    cfg = sources()["india"]["wpi"]
    page = requests.get(cfg["download_page"], headers=HEADERS, timeout=60)
    page.raise_for_status()
    links = re.findall(r'href="(indx_download_1112/monthly_index_\d{6}\.xls)"', page.text)
    if not links:
        raise RuntimeError("monthly_index_*.xls link not found on the OEA download page")
    url = cfg["download_page"].rsplit("/", 1)[0] + "/" + sorted(links)[-1]
    r = requests.get(url, headers=HEADERS, timeout=180)
    r.raise_for_status()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(r.content)
    record(RAW, url)


def tidy(path=RAW) -> pd.DataFrame:
    """Long panel: comm_code, comm_name, weight, month, index."""
    wide = pd.read_excel(path, dtype={"COMM_CODE": str})
    wide["COMM_NAME"] = wide["COMM_NAME"].astype(str).str.strip()
    long = wide.melt(id_vars=["COMM_NAME", "COMM_CODE", "COMM_WT"], var_name="col", value_name="index")
    m = long["col"].str.extract(r"INDX(\d{2})(\d{4})")
    long["month"] = m[1] + "-" + m[0]
    long["index"] = pd.to_numeric(long["index"], errors="coerce")
    out = long.dropna(subset=["index", "month"]).rename(
        columns={"COMM_NAME": "comm_name", "COMM_CODE": "comm_code", "COMM_WT": "weight"})
    return out[["comm_code", "comm_name", "weight", "month", "index"]].sort_values(
        ["comm_code", "month"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if not args.offline:
        fetch()
    df = tidy()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED / "india_wpi.parquet", index=False)
    print(f"{df.comm_code.nunique()} commodities, {df.month.min()}..{df.month.max()}")


if __name__ == "__main__":
    main()
