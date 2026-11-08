"""Backtests of the frozen Tokyo -> National updating rule.

Rule: a normal-normal update. Prior N(m, v) for the National quantity; the Tokyo reading is
a signal with noise variance r, so posterior mean = m + k (s - m), k = v / (v + r), and
posterior variance = v r / (v + r).

(a) Monthly: predict National item m/m changes in 2025-26 from Tokyo, with m, v and r
    estimated only on 2015-2024.
(b) Oct 2019: predict National item rho from a prior fitted without Japan 2019, updated
    with Tokyo's rho. This is the closest analogue to April 2027.

  python -m src.backtest.tokyo_national
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import cross_episode as ce
from src.config import episode
from src.forecast import extraction as ex
from src.model import features as ft, hierarchical as hm
from src.panel import complete_items, japan_2019
from src.passthrough import estimator as est, inference as inf, run as R
from src.paths import REPORTS
from src.scoring import metrics as sc

OUT = REPORTS / "backtests"
TRAIN_END = pd.Period("2024-12", "M")
TEST = (pd.Period("2025-01", "M"), pd.Period("2026-09", "M"))
SD_FLOOR = 1e-4


def update(m, v, s, r):
    """Normal-normal posterior (mean, variance)."""
    k = v / (v + r)
    return m + k * (s - m), v * r / (v + r)


def monthly() -> pd.DataFrame:
    nat, tok = ex.levels(ex.NATIONAL), ex.levels(ex.TOKYO)
    codes = nat.columns.intersection(tok.columns)
    dn, dt = nat[codes].diff(), tok[codes].reindex(nat.index).diff()   # Tokyo is published a month earlier
    train = dn.index <= TRAIN_END
    test = (dn.index >= TEST[0]) & (dn.index <= TEST[1])
    seasonal = dn[train].groupby(dn.index[train].month).mean()
    m = pd.DataFrame(seasonal.reindex(dn.index.month).to_numpy(), index=dn.index, columns=codes)
    v = (dn[train] - m[train]).var(ddof=1).clip(lower=SD_FLOOR ** 2)
    r = (dn[train] - dt[train]).var(ddof=1).clip(lower=SD_FLOOR ** 2)

    obs = dn[test].stack()
    prior = m[test].stack().reindex(obs.index)
    signal = dt[test].stack().reindex(obs.index)
    item = obs.index.get_level_values(1)
    vv, rr = v.reindex(item).to_numpy(), r.reindex(item).to_numpy()
    ok = signal.notna().to_numpy() & prior.notna().to_numpy() & ~np.isnan(vv) & ~np.isnan(rr)
    y, p, s, vv, rr = 100 * obs.to_numpy()[ok], 100 * prior.to_numpy()[ok], 100 * signal.to_numpy()[ok], 1e4 * vv[ok], 1e4 * rr[ok]
    post, pv = update(p, vv, s, rr)
    label = {"test": "National item m/m, 2025-01..2026-09 (percent)"}
    w = np.ones_like(y)
    return pd.DataFrame([
        ce.score_row(label, "Tokyo update (frozen rule)", y, post, np.sqrt(pv), w),
        ce.score_row(label, "seasonal prior only", y, p, np.sqrt(vv), w),
        ce.score_row(label, "Tokyo reading as-is", y, s, np.sqrt(rr), w),
    ])


def tokyo_2019() -> pd.DataFrame:
    """Oct 2019: LOEO prior vs prior updated with Tokyo, scored on National rho."""
    table = ft.training_table()
    nat_items = table[table.episode == "japan_2019"].set_index("item_id")
    ep = episode("japan_2019")
    win = R.window(ep)
    panels = {area: complete_items(japan_2019(area), "japan_2019") for area in ("00000", "13A01")}
    results, placebos = {}, {}
    for area, p in panels.items():
        p = p[p.role != "excluded"]
        lv = est.wide_log_levels(p)
        meta = p.drop_duplicates("item_id").set_index("item_id").reindex(lv.columns)
        cost, group = R.cost_inputs(ep, lv.columns, meta.category)
        ctrl, cat = meta.role == "control", meta.category
        net = est.net_of_controls(est.deviations(lv, win, est.Spec(), cost, group), ctrl, cat)
        results[area] = est.headline(net, win)
        bases = inf.placebo_bases(win, pd.Period(str(ep.raw["history_start"]), "M"))
        placebos[area] = inf.placebo_matrix(lv, win, bases, ctrl, cat, est.Spec(), cost, group)

    items = nat_items.index.intersection(results["13A01"].dropna().index)
    tau = nat_items.dlog_tax[items]
    tokyo_rho = results["13A01"][items] / tau
    gap = (placebos["00000"].loc[items] - placebos["13A01"].loc[items]).div(tau, axis=0)
    r = gap.std(axis=1, ddof=1).clip(lower=hm.SE_FLOOR) ** 2

    model, fm = ce.fit_model(table[table.episode != "japan_2019"])
    test = nat_items.loc[items].reset_index()
    mu, sd = model.predict(fm.matrix(test), None)
    post, pv = update(mu, sd ** 2, tokyo_rho.to_numpy(), r.to_numpy())

    y, se = test.rho.to_numpy(), np.maximum(test.se.to_numpy(), hm.SE_FLOOR)
    w = ce._weights(se)
    label = {"test": "National rho, Oct 2019 (prior excludes Japan 2019)"}
    return pd.DataFrame([
        ce.score_row(label, "Tokyo update (frozen rule)", y, post, np.sqrt(pv + se ** 2), w),
        ce.score_row(label, "prior only", y, mu, np.sqrt(sd ** 2 + se ** 2), w),
        ce.score_row(label, "Tokyo reading as-is", y, tokyo_rho.to_numpy(), np.sqrt(r.to_numpy() + se ** 2), w),
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.concat([monthly(), tokyo_2019()], ignore_index=True)
    res.to_csv(OUT / "tokyo_national.csv", index=False, lineterminator="\n", encoding="utf-8", float_format="%.6f")
    pd.set_option("display.width", 200)
    print(res[["test", "forecast", "n", "crps", "crps_weighted", "mae", "coverage_50", "coverage_90"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
