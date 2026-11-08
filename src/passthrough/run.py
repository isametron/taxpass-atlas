"""Estimate pass-through for every historical episode and write reports/estimates/.

  python -m src.passthrough.run

Files, per episode: <episode>_items.csv, <episode>_categories.csv.
Across episodes: overall.csv, robustness.csv, japan_2019_wedge.csv.
Pooled estimates weight items by placebo precision. They are not CPI-weighted: CPI weights
are not in the panels yet, so they describe a typical item, not the consumer basket.
"""

from __future__ import annotations

import numpy as np
import openpyxl
import pandas as pd

from src.config import Episode, episode
from src.passthrough import estimator as est, inference as inf
from src.paths import CONCORDANCES, DATA_PROCESSED, DATA_RAW, REPORTS

OUT = REPORTS / "estimates"
EPISODES = ("japan_2019", "singapore_2023", "singapore_2024", "india_gst2")
SEED = 20261108
ROBUSTNESS = {
    "baseline": est.Spec(),
    "flat_trend": est.Spec(seasonal=False),
    "no_controls": est.Spec(did=False),
    "no_cost_control": est.Spec(cost=False),
}
INDIA_DURABLE_PREFIXES = ("6.1.01.1", "6.1.01.2", "6.1.01.3", "6.1.01.4", "6.1.01.5", "6.1.03.1", "6.1.04.1")


def window(ep: Episode) -> est.Window:
    P = lambda s: pd.Period(s, "M")
    return est.Window(P(ep.base_month), P(ep.event_month), ep.horizons, ep.headline_h,
                      P(ep.pre_window[0]), P(ep.pre_window[1]),
                      tuple((P(a), P(b)) for a, b in ep.exclude))


def cost_inputs(ep: Episode, items: pd.Index, category: pd.Series):
    """(log cost series by group, item -> group), or (None, None) where no control is used."""
    if ep.country == "JP":
        raw = pd.read_parquet(DATA_PROCESSED / "japan_cgpi.parquet")
        wide = raw.pivot(index="month", columns="series", values="index")[["all_commodities", "beverages_and_foods"]]
        group = pd.Series(["beverages_and_foods" if i[0] in "12" else "all_commodities" for i in items], index=items)
    elif ep.country == "IN":
        raw = pd.read_parquet(DATA_PROCESSED / "india_wpi.parquet")
        wide = raw[raw.comm_code.isin(["1301000000", "1300000000"])].pivot(
            index="month", columns="comm_code", values="index").rename(
            columns={"1301000000": "food_products", "1300000000": "manufactured"})
        group = category.reindex(items).str.startswith("1.").map({True: "food_products", False: "manufactured"})
    else:
        return None, None
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    return np.log(wide), group


def category_names(ep: Episode) -> dict[str, str]:
    if ep.country == "SG":
        cpi = pd.read_parquet(DATA_PROCESSED / "singapore_cpi.parquet")
        return cpi.drop_duplicates("series_no").set_index("series_no")["name"].to_dict()
    if ep.country == "IN":
        ws = openpyxl.load_workbook(DATA_RAW / "india" / "cpi_metadata.xlsx", read_only=True)["SubGroup_code"]
        return {str(c).strip(): str(d).strip() for c, d, *_ in ws.iter_rows(min_row=2, values_only=True) if c}
    return {}


def pooled_row(label: dict, members: pd.Index, y: pd.Series, tau: pd.Series, placebo: pd.DataFrame,
               weights: pd.Series, clusters: pd.Series | None, seed: int) -> dict:
    """Precision-weighted rho for `members`, with placebo and wild intervals; headline CI is their union."""
    yy, tt = y.reindex(members).to_numpy(), tau.reindex(members).to_numpy()
    ww = weights.reindex(members).to_numpy()
    rho = est.ols_through_origin(yy, tt, ww)
    pl = inf.placebo_interval(rho, inf.pooled_placebo_draws(placebo.loc[members], tau, weights))
    cl = clusters.reindex(members).to_numpy() if clusters is not None else np.arange(len(members))
    root = np.sqrt(ww)
    w_lo, w_hi = inf.wild_interval(root * yy, root * tt, cl, seed=seed)
    bounds_lo = [v for v in (pl.low, w_lo) if not np.isnan(v)]
    bounds_hi = [v for v in (pl.high, w_hi) if not np.isnan(v)]
    return {**label, "n_items": int((~np.isnan(yy)).sum()), "rho": rho,
            "rho_unweighted": est.ols_through_origin(yy, tt),
            "rho_median": float(np.nanmedian(yy / tt)) if len(yy) else np.nan,
            "se_placebo": pl.se, "ci_placebo_low": pl.low, "ci_placebo_high": pl.high,
            "ci_wild_low": w_lo, "ci_wild_high": w_hi,
            "ci_low": min(bounds_lo) if bounds_lo else np.nan,
            "ci_high": max(bounds_hi) if bounds_hi else np.nan,
            "p_placebo": pl.p, "n_placebo_dates": pl.n}


def write_csv(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8", float_format="%.6f")


def run_episode(name: str) -> dict:
    ep = episode(name)
    win = window(ep)
    panel = pd.read_parquet(DATA_PROCESSED / f"panel_{name}.parquet")
    panel = panel[panel.role != "excluded"]
    levels = est.wide_log_levels(panel)
    meta = panel.drop_duplicates("item_id").set_index("item_id").reindex(levels.columns)
    is_control, category, tau = meta.role == "control", meta.category, meta.dlog_tax
    treated = meta.index[meta.role == "treated"]
    cost, group = cost_inputs(ep, levels.columns, category)

    dev = est.deviations(levels, win, est.Spec(), cost, group)
    net = est.net_of_controls(dev, is_control, category)
    y = est.headline(net, win)

    excl = tuple((pd.Period(a, "M"), pd.Period(b, "M")) for a, b in ep.raw.get("placebo_exclude", []))
    bases = inf.placebo_bases(win, pd.Period(str(ep.raw["history_start"]), "M"), excl)
    placebo = inf.placebo_matrix(levels, win, bases, is_control, category, est.Spec(), cost, group)
    weights = inf.precision_weights(placebo.loc[treated])

    rows = []
    for item in treated:
        pl = inf.placebo_interval(y[item] / tau[item], placebo.loc[item].to_numpy(), tau[item])
        rows.append({"item_id": item, "item_name": meta.item_name[item], "category": category[item],
                     "dlog_tax": tau[item], **{f"rho_h{h}": net.loc[item, h] / tau[item] for h in win.horizons},
                     "rho": y[item] / tau[item], "se": pl.se, "ci_low": pl.low, "ci_high": pl.high,
                     "p_placebo": pl.p, "n_placebo_dates": pl.n})
    items = pd.DataFrame(rows)
    write_csv(items, OUT / f"{name}_items.csv")

    names = category_names(ep)
    cats = []
    for i, (cat, members) in enumerate(meta.loc[treated].groupby("category").groups.items()):
        cats.append(pooled_row({"category": cat, "category_name": names.get(cat, cat)}, members, y, tau,
                               placebo, weights, None, SEED + i))
    write_csv(pd.DataFrame(cats).sort_values("category"), OUT / f"{name}_categories.csv")

    overall = [pooled_row({"episode": name, "group": "all treated"}, treated, y, tau, placebo, weights, category, SEED)]
    if ep.country == "IN":
        for label, mask in [("food and beverages", treated.str.startswith("1.")),
                            ("durables", np.array([t.startswith(INDIA_DURABLE_PREFIXES) for t in treated]))]:
            overall.append(pooled_row({"episode": name, "group": label}, treated[mask], y, tau, placebo, weights, category, SEED))

    robust = []
    for label, spec in ROBUSTNESS.items():
        d = est.deviations(levels, win, spec, cost, group)
        yy = est.headline(est.net_of_controls(d, is_control, category, spec), win)
        robust.append({"episode": name, "spec": label, "rho": est.ols_through_origin(
            yy[treated].to_numpy(), tau[treated].to_numpy(), weights[treated].to_numpy())})
    robust.append({"episode": name, "spec": "unweighted_items",
                   "rho": est.ols_through_origin(y[treated].to_numpy(), tau[treated].to_numpy())})

    wedge = japan_wedge(levels, win, meta, cost, group, bases) if name == "japan_2019" else None
    return {"overall": overall, "robust": robust, "wedge": wedge, "items": items}


def japan_wedge(levels, win, meta, cost, group, bases) -> pd.DataFrame:
    """Eat-in vs grocery pairs in Oct 2019. Taxed sides report rho; untaxed grocery sides
    report a spillover (percent, relative to food controls outside the pairs)."""
    flags = pd.read_csv(CONCORDANCES / "japan_rate_flags.csv", dtype={"code_2020": str}).dropna(subset=["wedge_pair"])
    flags = flags[flags.code_2020.isin(levels.columns)]
    in_pairs = flags.code_2020[flags.wedge_side == "grocery"]
    is_control = (meta.role == "control") & ~meta.index.isin(in_pairs)
    category, tau = meta.category, meta.dlog_tax
    dev = est.deviations(levels, win, est.Spec(), cost, group)
    y = est.headline(est.net_of_controls(dev, is_control, category), win)
    placebo = inf.placebo_matrix(levels, win, bases, is_control, category, est.Spec(), cost, group)
    rows = []
    for (pair, side), g in flags.groupby(["wedge_pair", "wedge_side"]):
        ids = pd.Index(g.code_2020)
        taxed = (tau[ids] != 0).all()
        if taxed:
            value = est.ols_through_origin(y[ids].to_numpy(), tau[ids].to_numpy())
            draws = inf.pooled_placebo_draws(placebo.loc[ids], tau)
            pl = inf.placebo_interval(value, draws)
        else:
            value = 100 * y[ids].mean()
            pl = inf.placebo_interval(value, 100 * placebo.loc[ids].mean().to_numpy())
        rows.append({"pair": pair, "side": side, "measure": "rho" if taxed else "spillover_pct", "n_items": len(ids),
                     "value": value, "se": pl.se, "ci_low": pl.low, "ci_high": pl.high})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {name: run_episode(name) for name in EPISODES}
    overall = pd.DataFrame([r for x in results.values() for r in x["overall"]])
    write_csv(overall, OUT / "overall.csv")
    write_csv(pd.DataFrame([r for x in results.values() for r in x["robust"]]), OUT / "robustness.csv")
    write_csv(results["japan_2019"]["wedge"], OUT / "japan_2019_wedge.csv")
    cols = ["episode", "group", "n_items", "rho", "rho_unweighted", "rho_median", "ci_low", "ci_high", "p_placebo", "n_placebo_dates"]
    print(overall[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
