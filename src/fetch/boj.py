"""Japan Corporate Goods Price Index (CGPI) from the BOJ Time-Series Data Search API.

Open API, no key (BOJ manual, 18 Feb 2026). Used as the producer-cost control in the
Japan 2019 estimates and in the 2027 drift formula.

  python -m src.fetch.boj            fetch the series in config/sources.yaml
  python -m src.fetch.boj --offline  parse data/raw only
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import requests

from src.config import sources
from src.fetch.manifest import write_json
from src.paths import DATA_PROCESSED, DATA_RAW

RAW = DATA_RAW / "japan" / "boj_cgpi.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (taxpass-atlas research)", "Accept-Encoding": "gzip"}


def fetch() -> dict:
    cfg = sources()["japan"]["cgpi"]
    url = cfg["api"]
    params = {"format": "json", "lang": "en", "db": cfg["db"], "startDate": cfg["start"],
              "code": ",".join(cfg["series"].values())}
    r = requests.get(url, params=params, headers=HEADERS, timeout=120)
    r.raise_for_status()
    body = r.json()
    if body.get("STATUS") != 200:
        raise RuntimeError(f"BOJ API {body.get('STATUS')}: {body.get('MESSAGE')}")
    write_json(RAW, body, url, params)
    return body


def tidy(body: dict) -> pd.DataFrame:
    names = {v: k for k, v in sources()["japan"]["cgpi"]["series"].items()}
    rows = []
    for s in body["RESULTSET"]:
        v = s["VALUES"]
        for date, val in zip(v["SURVEY_DATES"], v["VALUES"]):
            if val is None:
                continue
            d = str(date)
            rows.append({"series": names.get(s["SERIES_CODE"], s["SERIES_CODE"]),
                         "series_code": s["SERIES_CODE"], "month": f"{d[:4]}-{d[4:6]}",
                         "index": float(val)})
    return pd.DataFrame(rows).sort_values(["series", "month"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    body = json.loads(RAW.read_text(encoding="utf-8")) if args.offline else fetch()
    df = tidy(body)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED / "japan_cgpi.parquet", index=False)
    for name, g in df.groupby("series"):
        print(f"{name}: {g.month.min()}..{g.month.max()} ({len(g)} months)")


if __name__ == "__main__":
    main()
