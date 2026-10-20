"""Japan CPI from the e-Stat API (needs ESTAT_APP_ID in .env).

Commands:
  python -m src.fetch.estat discover            list CPI tables (statsCode 00200573)
  python -m src.fetch.estat discover --stats-code 00200571   list Retail Price Survey tables
  python -m src.fetch.estat cpi2025             2025-base, national + Tokyo ward-area
  python -m src.fetch.estat cpi2020             2020-base, national + Tokyo ward-area
  python -m src.fetch.estat cpi2025 --offline   parse data/raw only

Raw pages are saved as returned (minus the appId) and recorded in data/raw/MANIFEST.json.
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

RAW_DIR = DATA_RAW / "japan"
PAGE = 100_000


def _app_id() -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("ESTAT_APP_ID", "").strip()
    if not key:
        raise SystemExit("ESTAT_APP_ID is not set. Copy .env.example to .env and add your e-Stat appID.")
    return key


def _get(endpoint: str, params: dict) -> dict:
    cfg = sources()["japan"]
    r = requests.get(f"{cfg['estat_api']}/{endpoint}", params={"appId": _app_id(), **params}, timeout=120)
    r.raise_for_status()
    body = r.json()
    root = next(iter(body.values()))
    root.get("PARAMETER", {}).pop("APP_ID", None)  # never write the key to data/raw
    status = root["RESULT"]["STATUS"]
    if status not in (0, 1, 2):  # 0 ok, 1 no data, 2 partial; others are errors
        raise RuntimeError(f"e-Stat error {status}: {root['RESULT'].get('ERROR_MSG')}")
    return body


def discover(stats_code: str) -> pd.DataFrame:
    body = _get("getStatsList", {"statsCode": stats_code})
    tables = body["GET_STATS_LIST"]["DATALIST_INF"].get("TABLE_INF", [])
    tables = tables if isinstance(tables, list) else [tables]
    return pd.DataFrame([{
        "id": t["@id"],
        "title": t["TITLE"]["$"] if isinstance(t["TITLE"], dict) else t["TITLE"],
        "statistics_name": t.get("STATISTICS_NAME"),
        "cycle": t.get("CYCLE"),
        "updated": t.get("UPDATED_DATE"),
    } for t in tables])


def fetch_table(name: str, stats_data_id: str, filters: dict) -> list[dict]:
    """Fetch all pages; returns the list of raw page bodies and saves them."""
    pages, start = [], 1
    while True:
        params = {"statsDataId": stats_data_id, "limit": PAGE, "startPosition": start, **filters}
        body = _get("getStatsData", params)
        pages.append(body)
        nxt = body["GET_STATS_DATA"]["STATISTICAL_DATA"]["RESULT_INF"].get("NEXT_KEY")
        if not nxt:
            break
        start = int(nxt)
    url = f"{sources()['japan']['estat_api']}/getStatsData"
    write_json(RAW_DIR / f"{name}.json", pages, url, {"statsDataId": stats_data_id, **filters})
    return pages


def _as_list(x):
    return x if isinstance(x, list) else [x]


def parse_time(code: str) -> str:
    """e-Stat monthly time code 'YYYY00MMMM' (e.g. '2019001010') -> 'YYYY-MM'."""
    if len(code) != 10 or code[4:6] != "00" or code[6:8] != code[8:10] or not "01" <= code[6:8] <= "12":
        raise ValueError(f"not a monthly e-Stat time code: {code}")
    return f"{code[:4]}-{code[6:8]}"


def tidy(pages: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (values, classes). values: tab, item_code, area, month, value."""
    first = pages[0]["GET_STATS_DATA"]["STATISTICAL_DATA"]
    classes = []
    for obj in _as_list(first["CLASS_INF"]["CLASS_OBJ"]):
        for c in _as_list(obj["CLASS"]):
            classes.append({
                "class_id": obj["@id"], "class_name": obj["@name"], "code": c["@code"],
                "name": c["@name"], "level": c.get("@level"), "parent": c.get("@parentCode"),
                "unit": c.get("@unit"),
            })
    rows = []
    for page in pages:
        for v in _as_list(page["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"]):
            val = v.get("$")
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue  # '-', '***', '…' mark missing or confidential cells
            rows.append({
                "tab": v.get("@tab"), "item_code": v.get("@cat01"), "area": v.get("@area"),
                "month": parse_time(v["@time"]), "value": val,
            })
    return pd.DataFrame(rows), pd.DataFrame(classes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["discover", "cpi2025", "cpi2020"])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--stats-code", default=sources()["japan"]["stats_code"])
    args = ap.parse_args()
    cfg = sources()["japan"]

    if args.what == "discover":
        df = discover(args.stats_code)
        print(df.to_string(max_colwidth=70))
        return

    if args.what == "cpi2025":
        table_id = cfg["cpi_2025_base"]
        areas = [cfg["area"]["national"], cfg["area"]["tokyo_ward_2025"]]
        filters = {"cdArea": ",".join(areas)}
    else:
        table_id = cfg["cpi_2020_base"]
        if not table_id:
            raise SystemExit("cpi_2020_base is not set in config/sources.yaml; run `discover` first.")
        areas = [cfg["area"]["national"], cfg["area"]["tokyo_ward_2020"]]
        filters = {"cdArea": ",".join(areas), "cdTimeFrom": "2016000101"}

    if args.offline:
        pages = json.loads((RAW_DIR / f"{args.what}.json").read_text(encoding="utf-8"))
    else:
        pages = fetch_table(args.what, table_id, filters)
    values, classes = tidy(pages)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    values.to_parquet(DATA_PROCESSED / f"japan_{args.what}.parquet", index=False)
    classes.to_parquet(DATA_PROCESSED / f"japan_{args.what}_classes.parquet", index=False)
    print(f"{len(values)} values, {values.item_code.nunique()} items, "
          f"{values.month.min()}..{values.month.max()}")


if __name__ == "__main__":
    main()
