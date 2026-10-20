"""India all-India item-level CPI (base 2012) from the MoSPI API platform.

Needs a free MoSPI API account (sign up at https://api.mospi.gov.in) with
MOSPI_EMAIL and MOSPI_PASSWORD in .env. The token lasts 15 minutes, which is
enough for one full pull; without a token the API returns only 10 records, so
this script refuses to run unauthenticated.

  python -m src.fetch.mospi            fetch 2014-01..2025-12 item indices
  python -m src.fetch.mospi --offline  parse data/raw only
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import sources
from src.fetch.manifest import write_json
from src.paths import DATA_PROCESSED, DATA_RAW, ROOT

RAW_DIR = DATA_RAW / "india"
YEARS = range(2014, 2026)
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}


def _token() -> str:
    load_dotenv(ROOT / ".env")
    email, pw = os.environ.get("MOSPI_EMAIL"), os.environ.get("MOSPI_PASSWORD")
    if not (email and pw):
        raise SystemExit("Set MOSPI_EMAIL and MOSPI_PASSWORD in .env (free account at api.mospi.gov.in).")
    cfg = sources()["india"]
    r = requests.post(cfg["mospi_api"] + cfg["endpoints"]["login"], json={"email": email, "password": pw}, timeout=60)
    r.raise_for_status()
    body = r.json()
    token = body.get("token") or body.get("data", {}).get("token")
    if not token:
        raise RuntimeError(f"login returned no token: {sorted(body)}")
    return token


def fetch_items() -> list[dict]:
    cfg = sources()["india"]
    url = cfg["mospi_api"] + cfg["endpoints"]["item_index"]
    headers = {"Authorization": _token()}
    records = []
    for year in YEARS:
        page = 1
        while True:
            params = {"Year": year, "Format": "JSON", "page": page}
            r = requests.get(url, params=params, headers=headers, timeout=120)
            r.raise_for_status()
            body = r.json()
            records.extend(body["data"])
            meta = body.get("meta_data", {})
            if meta.get("recordPerPage", 0) <= cfg["unauthenticated_record_limit"] and meta.get("totalPages", 1) > 1:
                raise RuntimeError("API is returning the unauthenticated 10-record page; check the token header")
            if page >= meta.get("totalPages", 1):
                break
            page += 1
    write_json(RAW_DIR / "cpi_item_index_2012.json", records, url, {"years": f"{YEARS.start}-{YEARS.stop - 1}"})
    return records


def tidy(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df[df["baseyear"].astype(str) == "2012"]
    df["month"] = [f"{int(y):04d}-{MONTHS[m]:02d}" for y, m in zip(df["year"], df["month"])]
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    out = df.rename(columns={"item": "item_name"})[["item_name", "month", "index", "status"]]
    return out.dropna(subset=["index"]).sort_values(["item_name", "month"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        records = json.loads((RAW_DIR / "cpi_item_index_2012.json").read_text(encoding="utf-8"))
    else:
        records = fetch_items()
    df = tidy(records)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED / "india_cpi_items.parquet", index=False)
    print(f"{df.item_name.nunique()} items, {df.month.min()}..{df.month.max()}")


if __name__ == "__main__":
    main()
