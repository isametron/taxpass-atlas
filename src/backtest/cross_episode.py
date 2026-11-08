"""Out-of-sample backtests of the pass-through model.

Leave-one-episode-out: predict an episode from the others, as a new episode.
India holdout: predict half of India's categories (seeded split) from the other half.

Observed rho is itself noisy, so the predictive distribution for it adds the item's se^2.
Mean scores over all items are dominated by very noisy ones, so scores are also reported
with the items' precision weights (floored at the median item's precision).

  python -m src.backtest.cross_episode
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model import features as ft, hierarchical as hm
from src.paths import REPORTS
from src.scoring import metrics as sc

OUT = REPORTS / "backtests"
SPLIT_SEED = 20261108
LEVELS = (0.5, 0.9)


def fit_model(train: pd.DataFrame) -> tuple[hm.Fit, ft.FeatureMap]:
    fm = ft.FeatureMap(train)
    return hm.fit(fm.matrix(train), train.rho.to_numpy(), train.se.to_numpy(), train.episode.tolist(), fm.columns), fm


def _weights(se: np.ndarray) -> np.ndarray:
    s = np.maximum(se, hm.SE_FLOOR)
    return 1.0 / np.maximum(s, np.median(s)) ** 2


def score_row(label: dict, forecast: str, y, mu, sd, w) -> dict:
    crps = sc.crps_gaussian(y, mu, sd)
    row = {**label, "forecast": forecast, "n": len(y), "crps": float(crps.mean()),
           "crps_weighted": float(np.average(crps, weights=w)), "mae": sc.mae(y, mu),
           "mae_weighted": float(np.average(np.abs(y - mu), weights=w))}
    for lv in LEVELS:
        lo, hi = sc.central_interval(mu, sd, lv)
        row[f"coverage_{int(lv * 100)}"] = sc.coverage(y, lo, hi)
    return row


def baselines(train: pd.DataFrame, test: pd.DataFrame, se: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {"full pass-through (rho=1)": (np.ones(len(test)), se),
           "no pass-through (rho=0)": (np.zeros(len(test)), se)}
    jp = train[train.episode == "japan_2019"]
    if len(jp):
        by = jp.groupby("broad").rho.agg(["median", "std"])
        overall = jp.rho.median()
        mean = test.broad.map(by["median"]).fillna(overall).to_numpy()
        spread = test.broad.map(by["std"]).fillna(jp.rho.std()).to_numpy()
        out["Japan 2019 category medians"] = (mean, np.sqrt(se ** 2 + spread ** 2))
    return out


def evaluate(train: pd.DataFrame, test: pd.DataFrame, label: dict, known_episode: str | None) -> list[dict]:
    model, fm = fit_model(train)
    mu, sd = model.predict(fm.matrix(test), known_episode)
    y, se = test.rho.to_numpy(), np.maximum(test.se.to_numpy(), hm.SE_FLOOR)
    w = _weights(se)
    rows = [score_row(label, "hierarchical model", y, mu, np.sqrt(sd ** 2 + se ** 2), w)]
    for name, (m, s) in baselines(train, test, se).items():
        rows.append(score_row(label, name, y, m, s, w))
    return rows


def india_holdout_split(table: pd.DataFrame) -> set[str]:
    cats = sorted(table[table.episode == "india_gst2"].category.unique())
    rng = np.random.default_rng(SPLIT_SEED)
    return set(rng.permutation(cats)[: len(cats) // 2].tolist())


def run(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ft.EPISODES:
        train = table[table.episode != target]
        rows += evaluate(train, table[table.episode == target], {"test": f"leave out {target}"}, None)
    held = india_holdout_split(table)
    is_held = (table.episode == "india_gst2") & table.category.isin(held)
    rows += evaluate(table[~is_held], table[is_held], {"test": "India held-out categories"}, "india_gst2")
    out = pd.DataFrame(rows)
    model_key = "hierarchical model"
    for col in ("crps", "crps_weighted"):
        ref = out[out.forecast == model_key].set_index("test")[col]
        out[f"model_skill_vs_this_{col}"] = [1 - ref[t] / v if f != model_key else np.nan
                                             for t, f, v in zip(out.test, out.forecast, out[col])]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = ft.training_table()
    res = run(table)
    res.to_csv(OUT / "cross_episode.csv", index=False, lineterminator="\n", encoding="utf-8", float_format="%.6f")
    pd.set_option("display.width", 220)
    print(res[["test", "forecast", "n", "crps", "crps_weighted", "mae_weighted", "coverage_50", "coverage_90",
               "model_skill_vs_this_crps_weighted"]].round(3).to_string(index=False))
    model, fm = fit_model(table)
    print(f"\nall-data fit: tau={model.tau:.3f} sigma={model.sigma:.3f}")
    for c, (m, s) in model.coefficients().items():
        print(f"  {c:18s} {m:7.3f} ({s:.3f})")
    print("  episode effects:", {k: round(v, 3) for k, v in model.episode_effects().items()})


if __name__ == "__main__":
    main()
