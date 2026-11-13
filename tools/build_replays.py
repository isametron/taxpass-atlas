"""Build replay fixtures so the 2027 scorer can be run end to end on past episodes.

  python -m tools.build_replays            writes tests/fixtures/<episode>_replay/

Each fixture mimics the 2027 lock for a past tax change:
- forecast.csv: leave-one-episode-out model forecast (fit without the episode), with the
  extraction noise and, for Japan, the Tokyo-National noise measured at placebo dates;
- spec.yaml: the episode's window and tax change;
- levels.csv / cost.csv: only the months the scorer needs;
- LOCK_MANIFEST.json: hashes of forecast.csv and spec.yaml, checked by the scorer.

Japan 2019 (standard-rate items, 8% -> 10%) has National and Tokyo data and a CGPI cost
control. Singapore 2024 (8% -> 9%) has National data only and no cost control.
Needs data/processed (run the fetchers and src.panel first).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from src.backtest import cross_episode as ce
from src.config import episode
from src.forecast import lock
from src.model import features as ft
from src.panel import complete_items, japan_2019
from src.passthrough import estimator as est, inference as inf, run as R
from src.paths import DATA_PROCESSED, ROOT
from tools.score_2027 import extract

FIXTURES = ROOT / "tests" / "fixtures"
REPLAYS = {
    "japan_2019": "Japan consumption tax 8% -> 10%, Oct 2019, standard-rate items; forecast fit without Japan 2019",
    "singapore_2024": "Singapore GST 8% -> 9%, Jan 2024; forecast fit without Singapore 2024",
}
JAPAN_COST = "all_commodities"


def _levels(name: str) -> dict[str, pd.DataFrame]:
    if name == "japan_2019":
        return {src: est.wide_log_levels(complete_items(japan_2019(area), name))
                for src, area in (("national", "00000"), ("tokyo", "13A01"))}
    return {"national": est.wide_log_levels(pd.read_parquet(DATA_PROCESSED / f"panel_{name}.parquet"))}


def _cost(name: str) -> pd.DataFrame | None:
    if name != "japan_2019":
        return None
    raw = pd.read_parquet(DATA_PROCESSED / "japan_cgpi.parquet")
    wide = raw[raw.series == JAPAN_COST].pivot(index="month", columns="series", values="index")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    return np.log(wide)


def _placebo_net(lv: pd.DataFrame, win: est.Window, cost, bases) -> pd.DataFrame:
    rows = {}
    for b in bases:
        fake = win.shifted(b)
        rows[str(b)] = extract(lv, fake, cost, fake.event + fake.headline_h)
    return pd.DataFrame(rows)


def build(name: str) -> pd.DataFrame:
    ep = episode(name)
    win = R.window(ep)
    table = ft.training_table()
    test = table[table.episode == name].reset_index(drop=True)
    model, fm = ce.fit_model(table[table.episode != name])
    mu, sd = model.predict(fm.matrix(test), None)
    tau = ep.dlog("standard")                     # exact, from the statutory rates
    assert np.allclose(test.dlog_tax, tau, atol=1e-5), f"{name}: replay assumes one tax change for all items"

    levels, cost = _levels(name), _cost(name)
    items = [i for i in test.item_id if all(i in lv.columns for lv in levels.values())]
    P = lambda s: pd.Period(str(s), "M")  # noqa: E731
    bases = inf.placebo_bases(win, P(ep.raw["history_start"]),
                              tuple((P(a), P(b)) for a, b in ep.raw.get("placebo_exclude", [])))
    nat = _placebo_net(levels["national"][items], win, cost, bases)
    fc = pd.DataFrame({"item": test.item_id, "name": test.item_name, "weight_national": 1.0, "fresh_food": False,
                       "rho_mu": mu, "rho_sd": sd}).set_index("item").loc[items]
    fc["extraction_se"] = nat.std(axis=1, ddof=1) / abs(tau)
    if "tokyo" in levels:
        tok = _placebo_net(levels["tokyo"][items], win, cost, bases)
        fc["tokyo_noise_sd"] = (nat - tok).std(axis=1, ddof=1) / abs(tau)
    else:
        fc["tokyo_noise_sd"] = np.nan
    fc = fc.dropna(subset=["extraction_se"]).reset_index()

    d = FIXTURES / f"{name}_replay"
    d.mkdir(parents=True, exist_ok=True)
    spec = {"name": name, "description": REPLAYS[name], "base_month": str(win.base), "event_month": str(win.event),
            "horizons": list(win.horizons), "headline_h": win.headline_h,
            "pre_window": [str(win.pre_start), str(win.pre_end)],
            "exclude": [[str(a), str(b)] for a, b in win.exclude], "dlog_tax": round(tau, 10),
            "cost_series": JAPAN_COST if cost is not None else None, "placebo_dates": len(bases)}
    (d / "spec.yaml").write_bytes(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True).encode("utf-8"))
    _csv(fc, d / "forecast.csv")
    months = pd.period_range(win.pre_start, win.end, freq="M")
    long = pd.concat([lv.reindex(index=months, columns=fc.item).rename_axis(index="month", columns="item")
                      .stack().rename("log_index").reset_index().assign(area=src)
                      for src, lv in levels.items()], ignore_index=True)
    long["month"] = long.month.astype(str)
    _csv(long[["area", "item", "month", "log_index"]], d / "levels.csv", "%.8f")
    if cost is not None:
        c = cost.reindex(months).iloc[:, 0]
        _csv(pd.DataFrame({"month": months.astype(str), "log_cost": c.to_numpy()}), d / "cost.csv", "%.8f")
    lock.write_manifest(lock.build_manifest([d / "forecast.csv", d / "spec.yaml"], {"replay_of": name}, root=d),
                        d / "LOCK_MANIFEST.json")
    print(f"{name}: {len(fc)} items, {len(bases)} placebo dates -> {d.relative_to(ROOT)}")
    return fc


def _csv(df: pd.DataFrame, path, fmt: str = "%.6f") -> None:
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8", float_format=fmt)


def main() -> None:
    for name in REPLAYS:
        build(name)


if __name__ == "__main__":
    main()
