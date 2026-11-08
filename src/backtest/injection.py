"""Fake-shock test: can the 2027 extraction formula recover a known pass-through?

A -6.701% tax cut with a known rho per item is added to real item series at past dates.
The exact protocol extraction is then run. Interval coverage uses each item's noise,
measured at placebo dates strictly before the injected dates.

  python -m src.backtest.injection
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import dlog_tax
from src.forecast import extraction as ex
from src.passthrough import estimator as est
from src.paths import CONCORDANCES, REPORTS

OUT = REPORTS / "backtests"
SEED = 20261108
TRUE_MEAN, TRUE_SD = 0.5, 0.3
LAST_NOISE_BASE = pd.Period("2023-12", "M")


def target_codes(lv: pd.DataFrame) -> list[str]:
    f = pd.read_csv(CONCORDANCES / "japan_rate_flags.csv", dtype={"code_2025": str}).fillna({"administered": ""})
    keep = f[(f.rate_class == "reduced_food") & (f.administered == "")].code_2025
    return [c for c in keep if c in lv.columns]


def noise_sd(lv: pd.DataFrame, win: est.Window, cost: pd.DataFrame, tau: float, first: pd.Period, last: pd.Period) -> pd.Series:
    """Item sd of the extraction at dates with no tax change, in rho units."""
    lead = (win.base - win.pre_start).n
    rows = {}
    for base in pd.period_range(first + lead, last, freq="M"):
        rows[str(base)] = ex.net_change(lv, win.shifted(base), cost)
    return pd.DataFrame(rows).std(axis=1, ddof=1) / abs(tau)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    lv = ex.levels(ex.NATIONAL)
    codes = target_codes(lv)
    lv = lv[codes]
    cost, win = ex.cost_series(), ex.window()
    tau = dlog_tax(8, 1)
    rng = np.random.default_rng(SEED)
    truth = pd.Series(rng.normal(TRUE_MEAN, TRUE_SD, len(codes)), index=codes)
    se = noise_sd(lv, win, cost, tau, pd.Period("2015-01", "M"), LAST_NOISE_BASE)

    rows = []
    for base in pd.period_range("2024-03", "2026-06", freq="M"):
        fake = win.shifted(base)
        shocked = lv.copy()
        shocked.loc[fake.event:] = shocked.loc[fake.event:].add(truth * tau, axis=1)
        got = ex.realised_rho(shocked, fake, cost, tau)
        rows.append(pd.DataFrame({"base": str(base), "calendar_month": base.month, "item": codes,
                                  "truth": truth.to_numpy(), "recovered": got.to_numpy(), "se": se.reindex(codes).to_numpy()}))
    long = pd.concat(rows, ignore_index=True).dropna()
    return long, summarize(long)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    def stats(g: pd.DataFrame, label: str) -> dict:
        e = g.recovered - g.truth
        return {"sample": label, "n": len(g), "items": g.item.nunique(), "bias": e.mean(),
                "median_error": e.median(), "rmse": float(np.sqrt((e ** 2).mean())),
                "mae": e.abs().mean(), "median_se": g.se.median(),
                "coverage_50": (e.abs() <= 0.6745 * g.se).mean(), "coverage_90": (e.abs() <= 1.6449 * g.se).mean()}

    out = [stats(long, "all injected dates"), stats(long[long.calendar_month == 3], "March base months (same as 2027)")]
    trimmed = long[long.se <= long.se.quantile(0.75)]
    out.append(stats(trimmed, "lower-noise 75% of items"))
    by_month = long.groupby("base").apply(lambda g: (g.recovered - g.truth).mean(), include_groups=False)
    out.append({"sample": "basket mean error by date: sd across dates", "n": len(by_month), "bias": by_month.mean(),
                "rmse": float(np.sqrt((by_month ** 2).mean())), "mae": by_month.abs().mean()})
    return pd.DataFrame(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, summary = run()
    summary.to_csv(OUT / "injection.csv", index=False, lineterminator="\n", encoding="utf-8", float_format="%.6f")
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
