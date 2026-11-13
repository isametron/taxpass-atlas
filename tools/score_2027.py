"""Score the locked Japan April 2027 forecast against Tokyo or National CPI.

  python -m tools.score_2027 --month 2027-04 --source tokyo            early read, late April 2027
  python -m tools.score_2027 --month 2027-04 --source national         interim, May 2027
  python -m tools.score_2027 --month 2027-06 --source national --fetch final score, after June National
  python -m tools.score_2027 --replay tests/fixtures/japan_2019_replay --month 2019-12 --source national

Steps, as fixed in config/japan_2027_protocol.yaml:
1. Check the forecast and protocol against forecasts/LOCK_MANIFEST.json, and the src/ + config/
   code tree against the hash recorded at the lock. Stop on any mismatch.
2. With --fetch, refresh the e-Stat CPI and BOJ CGPI pulls first (needs ESTAT_APP_ID in .env).
3. Run the locked extraction on the outcome months published so far.
4. Score items, eating-out spillovers and CPI aggregates; write CSVs and a Markdown report.

The scorer lives outside src/ on purpose: the lock records a hash over src/ and config/, and
keeping new code out of them lets every score prove it ran the exact locked extraction.

A replay directory holds a forecast and data for a past episode, locked the same way, so the
whole scoring path can be run today (see tools/build_replays.py).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest import cross_episode as ce
from src.backtest.tokyo_national import update as tokyo_update
from src.config import dlog_tax, episode
from src.forecast import extraction as ex, lock
from src.passthrough import estimator as est
from src.paths import DATA_PROCESSED, FORECASTS, REPORTS, ROOT
from src.scoring import metrics as sc

Z90 = 1.644854
SOURCES = ("national", "tokyo")
P = lambda s: pd.Period(str(s), "M")  # noqa: E731


class LockError(RuntimeError):
    """The forecast, protocol or code no longer matches the lock."""


class DataError(RuntimeError):
    """The requested month cannot be scored with the data on hand."""


@dataclass
class Case:
    name: str
    win: est.Window
    tau: float
    forecast: pd.DataFrame                 # item, name, weight_national, fresh_food, rho_mu, rho_sd, extraction_se, tokyo_noise_sd
    levels: dict[str, pd.DataFrame]        # source -> months x items, log index
    cost: pd.DataFrame | None              # months x 1 log cost series, or None for no cost control
    provenance: list[str]
    spillover: pd.DataFrame | None = None  # item, name, mean_dlog_pct, sd_dlog_pct
    aggregates: pd.DataFrame | None = None
    total_weight: float | None = None
    fresh_weight: float | None = None


# ---------------------------------------------------------------- lock

def check_lock(manifest: Path, root: Path, code: bool, allow_code_change: bool = False) -> list[str]:
    """Raise LockError unless every locked file (and, if `code`, the code tree) matches."""
    if not manifest.exists():
        raise LockError(f"{manifest} not found")
    problems = lock.verify_manifest(manifest, root=root)
    if problems:
        raise LockError("locked files do not match the manifest: " + "; ".join(problems))
    lines = [f"Lock manifest `{manifest.name}` (sha256 `{lock.sha256_file(manifest)}`): every locked file matches."]
    ots = manifest.with_name(manifest.name + ".ots")
    lines.append(f"Time-stamp receipt `{ots.name}`: {'present' if ots.exists() else 'MISSING'}.")
    if code:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))["code_tree_sha256"]
        if lock.code_tree_sha256(root) == recorded:
            lines.append("Code tree (src/, config/): matches the lock.")
        elif allow_code_change:
            lines.append("**WARNING: code under src/ or config/ differs from the lock.** Scored anyway "
                         "with --allow-code-change; check the extraction against the lock commit.")
        else:
            raise LockError("code under src/ or config/ changed since the lock. Score from the lock commit, "
                            "or pass --allow-code-change to score with a warning in the report.")
    return lines


# ---------------------------------------------------------------- extraction

def window_through(win: est.Window, through: pd.Period) -> est.Window:
    """The headline window cut to outcome months up to `through` (the full window once all are out)."""
    hs = tuple(h for h in win.headline_horizons() if win.event + h <= through)
    if not hs:
        raise DataError(f"{through} is before the event month {win.event}")
    return dataclasses.replace(win, horizons=hs, headline_h=max(hs))


def _group(lv: pd.DataFrame, cost: pd.DataFrame | None) -> pd.Series | None:
    return None if cost is None else pd.Series(cost.columns[0], index=lv.columns)


def extract(lv: pd.DataFrame, win: est.Window, cost: pd.DataFrame | None, through: pd.Period) -> pd.Series:
    """Tax-attributable log change by item over the outcome months published by `through`.

    With every outcome month out this is exactly src.forecast.extraction.net_change.
    """
    w = window_through(win, through)
    lv = lv.loc[:through]
    return est.headline(est.deviations(lv, w, ex.SPEC, cost, _group(lv, cost)), w)


def anticipation(lv: pd.DataFrame, win: est.Window, cost: pd.DataFrame | None) -> pd.Series | None:
    """Cumulative deviation over the three months before the event (Jan-Mar 2027). Never scored."""
    if win.base not in lv.index:
        return None
    pre = dataclasses.replace(win, base=win.base - 3, event=win.base - 2, horizons=(0, 1, 2), headline_h=2)
    lv = lv.loc[:win.base]
    return est.deviations(lv, pre, ex.SPEC, cost, _group(lv, cost))[2]


# ---------------------------------------------------------------- scoring

def _last_month(lv: pd.DataFrame) -> pd.Period:
    return lv.dropna(how="all").index.max()


def _skill(rows: list[dict]) -> list[dict]:
    model = rows[0]
    for r in rows[1:]:
        model[f"skill_vs_{r['forecast'].split('(')[-1].rstrip(')')}"] = sc.skill(model["crps"], r["crps"])
        model[f"skill_weighted_vs_{r['forecast'].split('(')[-1].rstrip(')')}"] = sc.skill(model["crps_weighted"], r["crps_weighted"])
    return rows


def score_aggregates(case: Case, items: pd.DataFrame) -> pd.DataFrame | None:
    """Observed vs forecast food-basket rho and headline / core CPI effects (pp)."""
    if case.aggregates is None or case.total_weight is None:
        return None
    fc, obs = items, items.dropna(subset=["observed_rho"])
    w_all, w_obs = fc.weight_national, obs.weight_national
    nf_all, nf_obs = ~fc.fresh_food.astype(bool), ~obs.fresh_food.astype(bool)
    basket = float(np.average(obs.observed_rho, weights=w_obs))
    core_rho = float(np.average(obs.observed_rho[nf_obs], weights=w_obs[nf_obs]))
    scale_headline = 100 * case.tau * w_all.sum() / case.total_weight
    scale_core = 100 * case.tau * w_all[nf_all].sum() / (case.total_weight - case.fresh_weight)
    # Items with no reading are assumed to move like the observed ones (weights rescaled).
    observed = {"food_basket_rho": (basket, 1.0),
                "headline_cpi_effect_pp": (basket * scale_headline, scale_headline),
                "core_cpi_ex_fresh_effect_pp": (core_rho * scale_core, scale_core)}
    rows = []
    for stat, (y, full) in observed.items():
        f = case.aggregates.set_index("statistic").loc[stat]
        sd = (f.q95 - f.q05) / (2 * Z90)
        crps = float(sc.crps_gaussian(y, f["mean"], sd))
        rows.append({"statistic": stat, "observed": y, "forecast_mean": f["mean"], "q05": f.q05, "q25": f.q25,
                     "q75": f.q75, "q95": f.q95, "in_50": f.q25 <= y <= f.q75, "in_90": f.q05 <= y <= f.q95,
                     "crps_gaussian_approx": crps, "abs_error": abs(y - f["mean"]),
                     "abs_error_rho1": abs(y - full), "abs_error_rho0": abs(y),
                     "weight_share_observed": float(w_obs.sum() / w_all.sum())})
    return pd.DataFrame(rows)


def score_spillover(case: Case, lv: pd.DataFrame, through: pd.Period) -> dict | None:
    if case.spillover is None:
        return None
    sp = case.spillover
    y = 100 * extract(lv.reindex(columns=sp.item), case.win, case.cost, through).reindex(sp.item).to_numpy()
    ok = ~np.isnan(y) & (sp.sd_dlog_pct.to_numpy() > 0)
    if not ok.any():
        return None
    mu, sd =sp.mean_dlog_pct.to_numpy()[ok], sp.sd_dlog_pct.to_numpy()[ok]
    row = ce.score_row({"section": "eating-out spillover, dlog p (pct)"}, "locked forecast", y[ok], mu, sd, np.ones(ok.sum()))
    row["observed_mean"] = float(y[ok].mean())
    return row


def score(case: Case, month, source: str) -> dict:
    """Score `case` with data published by `month` from `source`. Returns tables and report lines."""
    if source not in case.levels:
        raise DataError(f"no {source} data in this case")
    win, month = case.win, P(month)
    lv_all = case.levels[source]
    last = _last_month(lv_all)
    if month > last:
        raise DataError(f"{source} data end at {last}; {month} is not published yet")
    final_month = win.event + win.headline_h
    through = min(month, final_month)
    used = window_through(win, through)
    fc = case.forecast
    lv = lv_all.reindex(columns=fc.item)

    net = extract(lv, win, case.cost, through).reindex(fc.item)
    y = (net / case.tau).to_numpy()
    mu, sd, se = fc.rho_mu.to_numpy(), fc.rho_sd.to_numpy(), fc.extraction_se.to_numpy()
    items = fc.copy()
    items["observed_rho"] = y
    if source == "national":
        noise = se
    else:
        tok = fc.tokyo_noise_sd.to_numpy()
        noise = np.sqrt(se ** 2 + tok ** 2)
        post, pv = tokyo_update(mu, sd ** 2, y, tok ** 2)      # protocol: k = sd^2 / (sd^2 + tokyo_noise_sd^2)
        items["nowcast_national_mean"], items["nowcast_national_sd"] = post, np.sqrt(pv)
    pred_sd = np.sqrt(sd ** 2 + noise ** 2)
    items["pred_sd"] = pred_sd
    ok = ~np.isnan(y) & ~np.isnan(pred_sd)
    items["pit"] = np.where(ok, sc.pit_gaussian(np.nan_to_num(y), mu, pred_sd), np.nan)
    items["crps"] = np.where(ok, sc.crps_gaussian(np.nan_to_num(y), mu, pred_sd), np.nan)
    items["in_50"] = np.where(ok, np.abs(y - mu) <= 0.67449 * pred_sd, np.nan)
    items["in_90"] = np.where(ok, np.abs(y - mu) <= Z90 * pred_sd, np.nan)
    ant = anticipation(lv, win, case.cost)
    if ant is not None:
        items["anticipation_pct"] = 100 * ant.reindex(fc.item).to_numpy()
    if not ok.any():
        raise DataError("no item could be extracted")

    w = ce._weights(se[ok])
    label = {"section": f"items, {source} rho"}
    rows = _skill([ce.score_row(label, "locked forecast", y[ok], mu[ok], pred_sd[ok], w),
                   ce.score_row(label, "full pass-through (rho=1)", y[ok], np.ones(ok.sum()), noise[ok], w),
                   ce.score_row(label, "no pass-through (rho=0)", y[ok], np.zeros(ok.sum()), noise[ok], w)])
    scores = pd.DataFrame(rows)
    agg = spill = None
    if source == "national":
        agg = score_aggregates(case, items)
        spill = score_spillover(case, lv_all, through)
        if spill:
            scores = pd.concat([scores, pd.DataFrame([spill])], ignore_index=True)

    missing_cost = []
    if case.cost is not None:
        need = pd.period_range(win.base + 1, through, freq="M")
        missing_cost = [str(m) for m in need if m not in case.cost.index or np.isnan(case.cost.loc[m].iloc[0])]
    caveats = []
    if through < final_month:
        caveats.append(f"Interim: {len(used.horizons)} of {len(win.headline_horizons())} outcome months are out. "
                       "extraction_se was measured for the full 3-month mean, so interim intervals are approximate.")
    if source == "tokyo":
        caveats.append("Provisional: scored against Tokyo ward-area prices with the locked Tokyo noise added. "
                       "The protocol scores National data; the Tokyo nowcast columns are the protocol's update rule.")
    if missing_cost:
        caveats.append(f"CGPI not yet published for {', '.join(missing_cost)}: those months enter the cost term as "
                       "no surprise. Rescore once CGPI is out.")
    status = "final" if not caveats else "interim"
    return {"case": case, "source": source, "month": month, "through": through, "status": status,
            "months_used": [str(win.event + h) for h in used.horizons], "items": items, "scores": scores,
            "aggregates": agg, "caveats": caveats, "n_items": len(fc), "n_scored": int(ok.sum()),
            "anticipation": None if ant is None else _anticipation_summary(items, case.tau)}


def _anticipation_summary(items: pd.DataFrame, tau: float) -> dict:
    a = items.dropna(subset=["anticipation_pct"])
    m = float(np.average(a.anticipation_pct, weights=a.weight_national))
    return {"weighted_mean_pct": m, "rho_equivalent": m / (100 * tau), "n": len(a)}


# ---------------------------------------------------------------- report

def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.3f}"
        return str(v)
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df[cols].itertuples(index=False)]
    return "\n".join(lines)


def report_markdown(res: dict) -> str:
    case = res["case"]
    out = [f"# Score: {case.name}, {res['source']} data through {res['month']}", "",
           f"**Status: {res['status']}.** Outcome months used: {', '.join(res['months_used'])}.", ""]
    out += [f"- {c}" for c in res["caveats"]] + ([""] if res["caveats"] else [])
    out += ["## Lock", ""] + [f"- {p}" for p in case.provenance] + [""]
    out += ["## Items", "",
            f"Tax change dlog(1+tau) = {case.tau:.5f}. {res['n_scored']} of {res['n_items']} forecast items had a reading.", "",
            _md_table(res["scores"], ["section", "forecast", "n", "crps", "crps_weighted", "mae", "coverage_50", "coverage_90",
                                      "skill_vs_rho=1", "skill_vs_rho=0"]), ""]
    if res["aggregates"] is not None:
        out += ["## Aggregates", "", "Forecast intervals are the locked simulation quantiles; CRPS uses a normal fit to them.", "",
                _md_table(res["aggregates"], ["statistic", "observed", "forecast_mean", "q05", "q95", "in_50", "in_90",
                                              "crps_gaussian_approx", "abs_error", "abs_error_rho1", "abs_error_rho0"]), "",
                "The grocery vs eat-in wedge is not scored: the protocol does not define how its eat-in side is observed.", ""]
    if res["anticipation"]:
        a = res["anticipation"]
        out += ["## Anticipation (reported, never scored)", "",
                f"Weighted mean deviation over the three months before the event: {a['weighted_mean_pct']:.3f}% "
                f"({a['rho_equivalent']:.3f} in rho units, {a['n']} items).", ""]
    return "\n".join(out)


def write(res: dict, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key in ("items", "scores", "aggregates"):
        if res[key] is not None:
            p = out_dir / f"{stem}_{key}.csv"
            res[key].to_csv(p, index=False, lineterminator="\n", encoding="utf-8", float_format="%.6f")
            paths.append(p)
    p = out_dir / f"{stem}.md"
    p.write_bytes((report_markdown(res) + "\n").encode("utf-8"))
    return paths + [p]


# ---------------------------------------------------------------- loading

def load_2027(data_dir: Path = DATA_PROCESSED, event_month: str | None = None, old_rate: float | None = None,
              new_rate: float | None = None, allow_code_change: bool = False) -> Case:
    provenance = check_lock(FORECASTS / "LOCK_MANIFEST.json", ROOT, code=True, allow_code_change=allow_code_change)
    ep = episode("japan_2027")
    old, new = ep.rates["reduced_food"]
    old = old if old_rate is None else old_rate
    new = new if new_rate is None else new_rate
    win = ex.window()
    if event_month is not None and P(event_month) != win.event:
        win = win.shifted(P(event_month) - 1)          # protocol: every window moves with the effective month
        provenance.append(f"Effective month moved to {event_month}: every window shifted with it.")
    tau = dlog_tax(old, new)
    provenance.append(f"Tax change scored: reduced food rate {old:g}% -> {new:g}%, effective {win.event}.")

    fc = pd.read_csv(FORECASTS / "japan_2027_locked_forecast.csv", dtype={"code_2025": str}).rename(columns={"code_2025": "item"})
    sp = pd.read_csv(FORECASTS / "japan_2027_spillover_forecast.csv", dtype={"code_2025": str}).rename(columns={"code_2025": "item"})
    agg = pd.read_csv(FORECASTS / "japan_2027_aggregates.csv")
    cpi = data_dir / "japan_cpi2025.parquet"
    levels = {"national": ex.levels(ex.NATIONAL, cpi), "tokyo": ex.levels(ex.TOKYO, cpi)}
    cost = ex.cost_series(data_dir / "japan_cgpi.parquet")

    from src.forecast.japan_2027 import fresh_codes     # same weights and fresh-food set as the forecast
    info = pd.read_parquet(DATA_PROCESSED / "japan_items_2025.parquet").set_index("code")
    fresh = fresh_codes() & set(info.index)
    return Case("Japan food tax cut, April 2027", win, tau, fc, levels, cost, provenance, sp, agg,
                float(info.weight_national.sum()), float(info.weight_national[list(fresh)].sum()))


def load_replay(d: Path) -> Case:
    provenance = check_lock(d / "LOCK_MANIFEST.json", d, code=False)
    spec = yaml.safe_load((d / "spec.yaml").read_text(encoding="utf-8"))
    win = est.Window(P(spec["base_month"]), P(spec["event_month"]), tuple(spec["horizons"]), int(spec["headline_h"]),
                     P(spec["pre_window"][0]), P(spec["pre_window"][1]),
                     tuple((P(a), P(b)) for a, b in spec.get("exclude", [])))
    fc = pd.read_csv(d / "forecast.csv", dtype={"item": str})
    long = pd.read_csv(d / "levels.csv", dtype={"item": str, "area": str})
    levels = {}
    for area, g in long.groupby("area"):
        wide = g.pivot(index="month", columns="item", values="log_index")
        wide.index = pd.PeriodIndex(wide.index, freq="M")
        levels[area] = wide.reindex(pd.period_range(wide.index.min(), wide.index.max(), freq="M"))
    cost = None
    if spec.get("cost_series"):
        c = pd.read_csv(d / "cost.csv")
        cost = pd.DataFrame({spec["cost_series"]: c.log_cost.to_numpy()}, index=pd.PeriodIndex(c.month, freq="M"))
    provenance.append(f"Replay of {spec['name']}: {spec['description']}")
    return Case(spec["name"], win, float(spec["dlog_tax"]), fc, levels, cost, provenance)


def fetch() -> None:
    for args in (["src.fetch.estat", "cpi2025"], ["src.fetch.boj"]):
        subprocess.run([sys.executable, "-m", *args], check=True, cwd=ROOT)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score the locked Japan 2027 forecast.")
    ap.add_argument("--month", required=True, help="latest month to use, e.g. 2027-04")
    ap.add_argument("--source", choices=SOURCES, required=True)
    ap.add_argument("--replay", type=Path, help="score a replay directory instead of the 2027 forecast")
    ap.add_argument("--data", type=Path, default=DATA_PROCESSED, help="directory with japan_cpi2025 and japan_cgpi parquet")
    ap.add_argument("--fetch", action="store_true", help="refresh the CPI and CGPI pulls first")
    ap.add_argument("--event-month", help="effective month if the enacted date differs from 2027-04")
    ap.add_argument("--old-rate", type=float)
    ap.add_argument("--new-rate", type=float, help="enacted reduced rate if it differs from 1%%")
    ap.add_argument("--allow-code-change", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.replay:
            case = load_replay(args.replay)
            out, stem = args.out or REPORTS / "replays", f"{args.replay.name}_{args.month}_{args.source}"
        else:
            if args.fetch:
                fetch()
            case = load_2027(args.data, args.event_month, args.old_rate, args.new_rate, args.allow_code_change)
            out, stem = args.out or REPORTS / "score_2027", f"score_2027_{args.month}_{args.source}"
        res = score(case, args.month, args.source)
    except (LockError, DataError) as e:
        raise SystemExit(f"not scored: {e}")
    for p in write(res, out, stem):
        print("wrote", p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
    print(res["scores"][["section", "forecast", "n", "crps", "crps_weighted", "coverage_50", "coverage_90"]].round(3).to_string(index=False))
    print("status:", res["status"])


if __name__ == "__main__":
    main()
