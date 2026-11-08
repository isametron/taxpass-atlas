"""Build the Japan April 2027 forecast and, on request, hash-lock it.

  python -m src.forecast.japan_2027          write the forecast files
  python -m src.forecast.japan_2027 --lock   also write forecasts/LOCK_MANIFEST.json

The OpenTimestamps stamp is a separate, explicit step: see src/forecast/lock.py.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.backtest import cross_episode as ce
from src.backtest.injection import noise_sd, target_codes
from src.config import dlog_tax
from src.forecast import extraction as ex, lock
from src.model import features as ft
from src.paths import CONCORDANCES, CONFIG, DATA_PROCESSED, DATA_RAW, FORECASTS, ROOT

SEED = 20261108
DRAWS = 20000
TAU = dlog_tax(8, 1)
NOISE_FIRST, NOISE_LAST = pd.Period("2015-01", "M"), pd.Period("2026-06", "M")
Z = {"q05": -1.644854, "q25": -0.67449, "q75": 0.67449, "q95": 1.644854}
FRESH_GROUPS = ("生鮮魚介", "生鮮野菜", "生鮮果物")


def fresh_codes() -> set[str]:
    c = pd.read_parquet(DATA_PROCESSED / "japan_cpi2025_classes.parquet").query("class_id == 'cat01'").set_index("code")
    parent, name = c.parent.to_dict(), c.name.str.split(" ", n=1).str[-1].to_dict()
    out = set()
    for code in c.index[c.level == "6"]:
        node = code
        while isinstance(parent.get(node), str) and parent[node]:
            node = parent[node]
            if name.get(node) in FRESH_GROUPS:
                out.add(code)
    return out


def tokyo_noise(lv_n: pd.DataFrame, lv_t: pd.DataFrame, win, cost) -> pd.Series:
    """Item sd, in rho units, of National minus Tokyo extraction at dates with no tax change."""
    lead = (win.base - win.pre_start).n
    rows = {}
    for base in pd.period_range(NOISE_FIRST + lead, NOISE_LAST, freq="M"):
        fake = win.shifted(base)
        rows[str(base)] = ex.net_change(lv_n, fake, cost) - ex.net_change(lv_t, fake, cost)
    return pd.DataFrame(rows).std(axis=1, ddof=1) / abs(TAU)


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    table = ft.training_table()
    model, fm = ce.fit_model(table)
    info = pd.read_parquet(DATA_PROCESSED / "japan_items_2025.parquet").set_index("code")
    lv_n, lv_t = ex.levels(ex.NATIONAL), ex.levels(ex.TOKYO)
    cost, win = ex.cost_series(), ex.window()
    codes = target_codes(lv_n)
    codes = [c for c in codes if c in lv_t.columns]

    se_ext = noise_sd(lv_n[codes], win, cost, TAU, NOISE_FIRST, NOISE_LAST)
    tok_sd = tokyo_noise(lv_n[codes], lv_t[codes], win, cost)
    sticky = ft.stickiness(lv_n[codes], "2025-01", "2026-09")
    retail = ft.load_retail()
    charm = ft.charm_share(retail, "2025-01", "2026-09")
    x = pd.DataFrame({"code_2025": codes})
    x["broad"] = [ft.broad_japan(info.gs_code[c], info.durable_code[c]) for c in codes]
    x["cut"], x["display"] = 1.0, 1.0
    x["tax_size"] = np.log(100 * abs(TAU))
    x["sticky"] = x.code_2025.map(sticky)
    x["charm"] = x.code_2025.map(charm)
    X = fm.matrix(x)
    mu, sd = model.predict(X, None)

    fresh = fresh_codes()
    fc = pd.DataFrame({"code_2025": codes, "name": [info.name[c] for c in codes], "broad": x.broad,
                       "fresh_food": [c in fresh for c in codes],
                       "weight_national": [info.weight_national[c] for c in codes],
                       "rho_mu": mu, "rho_sd": sd})
    for q, z in Z.items():
        fc[f"rho_{q}"] = mu + z * sd
    fc["extraction_se"] = se_ext.reindex(codes).to_numpy()
    fc["tokyo_noise_sd"] = tok_sd.reindex(codes).to_numpy()
    fc = fc.dropna(subset=["extraction_se", "tokyo_noise_sd"]).reset_index(drop=True)

    # Aggregates: draws share the coefficients and the single episode effect.
    keep = np.isin(codes, fc.code_2025)
    draws = model.sample_new_episode(X[keep], DRAWS, SEED)
    w = fc.weight_national.to_numpy()
    total = float(info.weight_national.sum())
    nonfresh = ~fc.fresh_food.to_numpy()
    fresh_weight = float(info.weight_national[list(fresh & set(info.index))].sum())
    flags = pd.read_csv(CONCORDANCES / "japan_rate_flags.csv", dtype={"code_2025": str}).fillna({"administered": ""})
    grocery = flags[(flags.wedge_side == "grocery") & (flags.wedge_pair != "beer_placebo")].code_2025
    in_wedge = fc.code_2025.isin(grocery).to_numpy()
    stats = {
        "headline_cpi_effect_pp": 100 * TAU * (draws * w).sum(axis=1) / total,
        "core_cpi_ex_fresh_effect_pp": 100 * TAU * (draws[:, nonfresh] * w[nonfresh]).sum(axis=1) / (total - fresh_weight),
        "food_basket_rho": (draws * w).sum(axis=1) / w.sum(),
        "wedge_grocery_minus_eat_in_pp": 100 * TAU * draws[:, in_wedge].mean(axis=1),
    }
    agg = pd.DataFrame([{"statistic": k, "mean": v.mean(), **{f"q{int(p * 100):02d}": np.quantile(v, p) for p in (0.05, 0.25, 0.75, 0.95)}}
                        for k, v in stats.items()])

    out_lv = lv_n[[c for c in lv_n.columns if c in set(flags[flags.rate_class == "standard_eating_out"].code_2025)
                   and c not in set(flags[flags.administered != ""].code_2025)]]
    sp_sd = noise_sd(out_lv, win, cost, 1.0, NOISE_FIRST, NOISE_LAST)
    spill = pd.DataFrame({"code_2025": out_lv.columns, "name": [info.name[c] for c in out_lv.columns],
                          "mean_dlog_pct": 0.0, "sd_dlog_pct": 100 * sp_sd.reindex(out_lv.columns).to_numpy()})
    return fc, spill, agg, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true")
    args = ap.parse_args()
    fc, spill, agg, model = build()
    FORECASTS.mkdir(exist_ok=True)
    paths = {"forecast": FORECASTS / "japan_2027_locked_forecast.csv",
             "spillover": FORECASTS / "japan_2027_spillover_forecast.csv",
             "aggregates": FORECASTS / "japan_2027_aggregates.csv"}
    fc.to_csv(paths["forecast"], index=False, lineterminator="\n", encoding="utf-8", float_format="%.6f")
    spill.to_csv(paths["spillover"], index=False, lineterminator="\n", encoding="utf-8", float_format="%.6f")
    agg.to_csv(paths["aggregates"], index=False, lineterminator="\n", encoding="utf-8", float_format="%.6f")
    print(f"{len(fc)} items; tau={model.tau:.3f}; sigma={model.sigma:.3f}")
    print(fc[["rho_mu", "rho_sd", "extraction_se", "tokyo_noise_sd"]].describe().loc[["mean", "min", "max"]].round(3).to_string())
    print(agg.round(3).to_string(index=False))
    if args.lock:
        files = [*paths.values(), CONFIG / "japan_2027_protocol.yaml"]
        # These two change as data is refetched, so they are recorded by value, not verified later.
        snapshot = {"requirements_lock_sha256": lock.sha256_file(ROOT / "requirements.lock"),
                    "input_data_manifest_sha256": lock.sha256_file(DATA_RAW / "MANIFEST.json"),
                    "model": {"tau": model.tau, "sigma": model.sigma}}
        digest = lock.write_manifest(lock.build_manifest(files, snapshot), FORECASTS / "LOCK_MANIFEST.json")
        print("LOCK_MANIFEST.json sha256:", digest)


if __name__ == "__main__":
    main()
