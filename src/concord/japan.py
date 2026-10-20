"""Japan CPI 2020-base -> 2025-base item concordance.

Source: e-Stat's "2025年基準消費者物価指数における地域コード及び分類項目コード一覧"
(2025DBcode.xlsx). It lists 2025-base and 2020-base codes side by side, one row per
item. Row alignment gives the mapping. The relation of each row is classified as:

  identity  same code, same name
  renamed   same code, different name (wording change; reviewed by hand)
  recoded   different codes on the same row
  new       2025-base only (no 2020 counterpart; parent category used for transfer)
  dropped   2020-base only

Run:  python -m src.concord.japan  [path/to/2025DBcode.xlsx]
"""

from __future__ import annotations

import sys
from pathlib import Path

import openpyxl
import pandas as pd

from src.paths import CONCORDANCES, DATA_RAW

CODELIST = DATA_RAW / "japan" / "2025DBcode.xlsx"
OUT = CONCORDANCES / "japan_cpi_2020_to_2025.csv"
AREA_OUT = CONCORDANCES / "japan_cpi_area_codes.csv"

# Column positions (0-based) in Sheet1; header rows end at row index 6.
_AREA_COLS = (1, 2, 3, 4)   # 2025 code, 2025 name, 2020 code, 2020 name
_ITEM_COLS = (6, 7, 8, 9)
_FIRST_DATA_ROW = 7


# Renamed food/beverage rows where the wording change is more than 男子用→男性用 style.
# These are in the 2027 target set, so the link is checked against the 2025-26 overlap
# of both bases before 2019 estimates are transferred.
_REVIEWED_RENAMES = {
    "1951": "炭酸飲料 split into 炭酸飲料Ａ/Ｂ; link to Ａ, check overlap",
    "1952": "ノンアルコールビール relabelled 炭酸飲料Ｃ; check overlap before linking",
    "6090": "健康保持用摂取品Ａ relabelled サプリメント; supplements are 食品, so reduced-rate (outside the food subtree)",
}


def _strip_code(name: str | None, code: str | None) -> str | None:
    """'0001 総合' -> '総合'."""
    if name is None:
        return None
    name = str(name).strip()
    if code and name.startswith(str(code)):
        name = name[len(str(code)):].strip()
    return name


def _relation(c25, n25, c20, n20) -> str:
    if pd.isna(c20):
        return "new"
    if pd.isna(c25):
        return "dropped"
    if c25 != c20:
        return "recoded"
    return "identity" if n25 == n20 else "renamed"


def _pairs(rows, cols) -> list[dict]:
    a, b, c, d = cols
    out = []
    for r in rows[_FIRST_DATA_ROW:]:
        r = list(r) + [None] * (10 - len(r))
        c25, c20 = r[a], r[c]
        if c25 is None and c20 is None:
            continue
        c25 = str(c25).strip() if c25 is not None else None
        c20 = str(c20).strip() if c20 is not None else None
        out.append({
            "code_2025": c25,
            "name_2025": _strip_code(r[b], c25),
            "code_2020": c20,
            "name_2020": _strip_code(r[d], c20),
        })
    return out


def build(codelist: Path = CODELIST) -> tuple[pd.DataFrame, pd.DataFrame]:
    ws = openpyxl.load_workbook(codelist, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))

    items = pd.DataFrame(_pairs(rows, _ITEM_COLS))
    items["relation"] = [
        _relation(r.code_2025, r.name_2025, r.code_2020, r.name_2020) for r in items.itertuples()
    ]
    # Codes below 1000 are aggregates (類・中分類・特殊分類); 4-digit codes from 1000 up are items.
    code = items["code_2025"].fillna(items["code_2020"])
    items["level"] = ["aggregate" if c.isdigit() and int(c) < 1000 else "item" for c in code]
    # Same code, different product: the 2020 item was relabelled. Reviewed by hand.
    items["note"] = items["code_2025"].map(_REVIEWED_RENAMES).fillna("")

    areas = pd.DataFrame(_pairs(rows, _AREA_COLS))
    areas["name_2025"] = [_strip_code(n, c) for n, c in zip(areas["name_2025"], areas["code_2025"])]
    return items, areas


# --- Consumption tax rate flags ---------------------------------------------------
#
# Reduced rate (軽減税率) since 2019-10: food and drink under the Food Labelling Act
# (incl. supplements), excluding alcohol and eat-in dining; plus subscription newspapers.
# Eating-out items are 10% unless delivered/taken out. These rules are drafted from the
# statute and still need checking against SBJ's item-level tax treatment notes for 2019.

ALCOHOL = range(2001, 2100)
EATING_OUT = range(2101, 2200)
FOOD = range(1001, 2000)
REDUCED_OUTSIDE_FOOD = {"2144": "pizza delivery is not eat-in", "6090": "supplements are 食品",
                        "6096": "protein powder is 食品",
                        "2182": "school lunch is reduced-rate", "2183": "school lunch is reduced-rate"}
NEWSPAPERS = {"9204", "9205"}
# Eat-in items often bought as takeout (8%); the CPI item may mix both rates.
MIXED_TAKEOUT = {"2135", "2136", "2161", "2162", "2163", "2164", "2165"}

# Non-taxable (非課税) under the Consumption Tax Act: controls in 2019, never targets.
EXEMPT = {
    "3000": "residential rent", "3016": "residential rent", "3017": "residential rent",
    "3030": "imputed rent (no transaction)", "3180": "insurance premium",
    "6200": "insured medical care", "7351": "administrative fee", "7370": "insurance premium",
    "7390": "insurance premium", "7401": "postage", "8010": "school tuition",
    "8020": "school tuition", "8030": "school tuition", "8040": "school tuition",
    "8060": "school tuition", "8070": "school tuition", "8077": "school tuition",
    "8110": "school textbooks", "9187": "insurance premium", "9901": "administrative fee",
    "9914": "long-term care insurance service", "9921": "licensed childcare",
    "9928": "insurance premium", "9951": "after-school childcare",
}
# Administered or policy-driven prices: dropped from estimation and targets.
ADMINISTERED = {
    "2182": "school lunch: municipal pricing and free-lunch policies",
    "2183": "school lunch: municipal pricing and free-lunch policies",
    "3016": "public housing rent", "3017": "public housing rent",
    "3810": "water tariff", "4610": "sewerage tariff", "6200": "medical fee schedule",
    "7401": "postal tariff", "8020": "free public high-school tuition",
    "9921": "free childcare from 2019-10", "9951": "childcare policy",
}
ADMINISTERED_NAME_PATTERN = "電気代|都市ガス代|プロパンガス|受信料|たばこ"  # tobacco retail prices need MOF approval

# Paired items for the eat-in vs takeout/grocery wedge. 'placebo' pairs share a rate.
WEDGE_PAIRS = {
    "sushi": (["1795", "1798"], ["2121", "2123"]),
    "hamburg_steak": (["1844", "1852"], ["2141"]),
    "gyoza": (["1881", "1853"], ["2134"]),
    "yakitori": (["1843"], ["2172"]),
    "curry": (["1871"], ["2133"]),
    "coffee": (["1923", "1924"], ["2162", "2165"]),
    "udon": (["1031"], ["2101"]),
    "fried_chicken": (["1842"], ["2164"]),
    "beer_placebo": (["2021"], ["2171"]),
}


def _rates(code: str) -> tuple[float, float, float, float, str]:
    """(2019 old, 2019 new, 2027 old, 2027 new, rate_class) for a 2025-base item code."""
    c = int(code)
    if code in NEWSPAPERS:
        return 8, 8, 8, float("nan"), "reduced_newspaper"   # 2027 treatment not legislated
    if c in FOOD or code in REDUCED_OUTSIDE_FOOD:
        return 8, 8, 8, 1, "reduced_food"
    if c in ALCOHOL:
        return 8, 10, 10, 10, "standard_alcohol"
    if c in EATING_OUT:
        return 8, 10, 10, 10, "standard_eating_out"
    if code in EXEMPT:
        return 0, 0, 0, 0, "exempt"
    return 8, 10, 10, 10, "standard"


def build_rate_flags(items: pd.DataFrame) -> pd.DataFrame:
    it = items[(items["level"] == "item") & items["code_2025"].notna()]
    rows = []
    pair_of = {c: (p, side) for p, (grocery, eat_in) in WEDGE_PAIRS.items()
               for side, codes in (("grocery", grocery), ("eat_in", eat_in)) for c in codes}
    for r in it.itertuples():
        o19, n19, o27, n27, cls = _rates(r.code_2025)
        admin = ADMINISTERED.get(r.code_2025, "")
        if not admin and pd.Series([r.name_2025]).str.contains(ADMINISTERED_NAME_PATTERN).iloc[0]:
            admin = "regulated tariff, approved price or energy subsidy"
        pair, side = pair_of.get(r.code_2025, ("", ""))
        rows.append({
            "code_2025": r.code_2025, "name_2025": r.name_2025, "code_2020": r.code_2020,
            "rate_class": cls, "rate_2019_old": o19, "rate_2019_new": n19,
            "rate_2027_old": o27, "rate_2027_new": n27,
            "mixed_takeout": r.code_2025 in MIXED_TAKEOUT,
            "exempt_reason": EXEMPT.get(r.code_2025, ""),
            "administered": admin,
            "reduced_reason": REDUCED_OUTSIDE_FOOD.get(r.code_2025, ""),
            "wedge_pair": pair, "wedge_side": side,
        })
    out = pd.DataFrame(rows)
    rate_cols = [c for c in out if c.startswith("rate_20")]
    out[rate_cols] = out[rate_cols].astype("Int64")
    return out


RATES_OUT = CONCORDANCES / "japan_rate_flags.csv"


def main(argv: list[str]) -> None:
    src = Path(argv[0]) if argv else CODELIST
    items, areas = build(src)
    rates = build_rate_flags(items)
    CONCORDANCES.mkdir(exist_ok=True)
    items.to_csv(OUT, index=False, lineterminator="\n", encoding="utf-8")
    areas.to_csv(AREA_OUT, index=False, lineterminator="\n", encoding="utf-8")
    rates.to_csv(RATES_OUT, index=False, lineterminator="\n", encoding="utf-8")
    print(items.groupby(["level", "relation"]).size().to_string())
    print(rates.groupby("rate_class").size().to_string())
    for p in (OUT, AREA_OUT, RATES_OUT):
        print(f"wrote {p.relative_to(p.parents[1]).as_posix()}")


if __name__ == "__main__":
    main(sys.argv[1:])
