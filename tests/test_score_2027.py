import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.config import dlog_tax
from src.forecast import extraction as ex, lock
from src.passthrough import estimator as est
from src.paths import DATA_PROCESSED, FORECASTS, ROOT
from tools import score_2027 as s27

P = lambda m: pd.Period(m, "M")  # noqa: E731
FIXTURES = ROOT / "tests" / "fixtures"
WIN = est.Window(P("2027-03"), P("2027-04"), (0, 1, 2), 2, P("2025-01"), P("2026-12"))
TAU = -0.06701


def _synthetic(n=30, rho=0.6, seed=0):
    """Seasonal random-walk log levels 2024-01..2027-06 with a tax cut passed through at `rho`."""
    rng = np.random.default_rng(seed)
    months = pd.period_range("2024-01", "2027-06", freq="M")
    season = rng.normal(0, 0.004, (12, n))
    steps = season[(months.month - 1).to_numpy()] + 0.001 + rng.normal(0, 0.0005, (len(months), n))
    lv = pd.DataFrame(np.cumsum(steps, axis=0), index=months, columns=[f"{1000 + i}" for i in range(n)])
    lv.loc[P("2027-04"):] += rho * TAU
    cost = pd.DataFrame({ex.COST_GROUP: np.cumsum(rng.normal(0.001, 0.002, len(months)))}, index=months)
    return lv, cost


def test_full_window_is_the_locked_extraction():
    lv, cost = _synthetic()
    got = s27.extract(lv, WIN, cost, P("2027-06"))
    pd.testing.assert_series_equal(got, ex.net_change(lv, WIN, cost))


def test_extraction_recovers_known_rho():
    lv, cost = _synthetic(rho=0.6)
    rho = s27.extract(lv, WIN, cost, P("2027-06")) / TAU
    assert abs(rho.median() - 0.6) < 0.1


def test_interim_uses_only_published_months():
    lv, cost = _synthetic()
    w = s27.window_through(WIN, P("2027-04"))
    assert w.horizons == (0,) and w.headline_h == 0
    april = s27.extract(lv, WIN, cost, P("2027-04"))
    # Data after April must not matter for an April score.
    later = lv.copy()
    later.loc[P("2027-05"):] += 0.5
    pd.testing.assert_series_equal(april, s27.extract(later, WIN, cost, P("2027-04")))
    with pytest.raises(s27.DataError):
        s27.window_through(WIN, P("2027-03"))


def _case(rho=0.6, tokyo=True):
    lv, cost = _synthetic(rho=rho)
    items = list(lv.columns)
    fc = pd.DataFrame({"item": items, "name": items, "weight_national": 1.0, "fresh_food": False,
                       "rho_mu": 0.5, "rho_sd": 0.3, "extraction_se": 0.1, "tokyo_noise_sd": 0.2})
    levels = {"national": lv.loc[:"2027-05"]}
    if tokyo:
        levels["tokyo"] = lv
    return s27.Case("synthetic", WIN, TAU, fc, levels, cost, ["synthetic"])


def test_unpublished_month_is_refused():
    with pytest.raises(s27.DataError, match="not published"):
        s27.score(_case(), "2027-06", "national")


def test_tokyo_nowcast_follows_protocol_rule():
    res = s27.score(_case(), "2027-06", "tokyo")
    it = res["items"]
    k = 0.3 ** 2 / (0.3 ** 2 + 0.2 ** 2)
    np.testing.assert_allclose(it.nowcast_national_mean, 0.5 + k * (it.observed_rho - 0.5))
    np.testing.assert_allclose(it.nowcast_national_sd, np.sqrt(0.3 ** 2 * 0.2 ** 2 / (0.3 ** 2 + 0.2 ** 2)))
    assert res["status"] == "interim"          # Tokyo scores are always provisional


def test_final_national_score_and_baselines():
    res = s27.score(_case(rho=1.0, tokyo=False), "2027-05", "national")
    assert res["status"] == "interim" and res["months_used"] == ["2027-04", "2027-05"]
    sc = res["scores"].set_index("forecast")
    # Truth is full pass-through, so the rho=1 baseline should beat rho=0.
    assert sc.loc["full pass-through (rho=1)", "crps"] < sc.loc["no pass-through (rho=0)", "crps"]


# ---------------------------------------------------------------- replays

@pytest.mark.parametrize("name, month", [("japan_2019_replay", "2019-12"), ("singapore_2024_replay", "2024-03")])
def test_replay_scores_end_to_end(name, month, tmp_path):
    case = s27.load_replay(FIXTURES / name)
    res = s27.score(case, month, "national")
    assert res["status"] == "final" and res["n_scored"] > 50
    paths = s27.write(res, tmp_path, "r")
    assert {p.name for p in paths} >= {"r.md", "r_items.csv", "r_scores.csv"}
    assert "every locked file matches" in (tmp_path / "r.md").read_text(encoding="utf-8")


def test_japan_replay_passes_through_close_to_one():
    res = s27.score(s27.load_replay(FIXTURES / "japan_2019_replay"), "2019-12", "national")
    assert 0.7 < res["items"].observed_rho.median() < 1.3


def test_tampered_replay_is_refused(tmp_path):
    d = tmp_path / "replay"
    shutil.copytree(FIXTURES / "japan_2019_replay", d)
    f = d / "forecast.csv"
    f.write_bytes(f.read_bytes().replace(b"0.025835", b"0.925835", 1))
    with pytest.raises(s27.LockError, match="hash mismatch"):
        s27.load_replay(d)


def test_scorer_cli_reports_refusal(tmp_path):
    with pytest.raises(SystemExit, match="not scored"):
        s27.main(["--replay", str(FIXTURES / "singapore_2024_replay"), "--month", "2024-03",
                  "--source", "tokyo", "--out", str(tmp_path)])


# ---------------------------------------------------------------- the real 2027 lock

def test_code_tree_still_matches_the_2027_lock():
    """New code belongs outside src/ and config/; otherwise the 2027 scorer will refuse to run."""
    lines = s27.check_lock(FORECASTS / "LOCK_MANIFEST.json", ROOT, code=True)
    assert any("matches the lock" in ln for ln in lines)


def _extend(df: pd.DataFrame, keys: list[str], value: str, until: str) -> pd.DataFrame:
    """Append fake months: each new month repeats the change of the same month a year earlier."""
    df = df[df[value] > 0]
    wide = df.pivot_table(index="month", columns=keys, values=value)
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    lg = np.log(wide).reindex(pd.period_range(wide.index.min(), until, freq="M", name="month"))
    was_missing = lg.isna()
    first = lg.apply(lambda c: c.last_valid_index()).min() + 1      # series end at different months
    for m in pd.period_range(first, until, freq="M"):
        lg.loc[m] = lg.loc[m].fillna(lg.loc[m - 1] + (lg.loc[m - 12] - lg.loc[m - 13]))
    new = np.exp(lg.where(was_missing & (lg.index >= first)[:, None])).stack(keys, future_stack=True)
    new = new.dropna().rename(value).reset_index()
    new["month"] = new.month.astype(str)
    return pd.concat([df, new], ignore_index=True)


@pytest.mark.skipif(not (DATA_PROCESSED / "japan_cpi2025.parquet").exists(), reason="needs data/processed")
def test_2027_path_recovers_injected_pass_through(tmp_path):
    """Real CPI and CGPI, extended with fake 2027 months in which the cut is passed through at 0.6."""
    rho_true = 0.6
    cpi = pd.read_parquet(DATA_PROCESSED / "japan_cpi2025.parquet")
    cpi = cpi[(cpi.tab == "1") & (cpi.month >= "2023-01")]
    cpi = _extend(cpi, ["tab", "item_code", "area"], "value", "2027-06")
    fc = pd.read_csv(FORECASTS / "japan_2027_locked_forecast.csv", dtype={"code_2025": str})
    hit = cpi.item_code.isin(fc.code_2025) & (cpi.month >= "2027-04")
    cpi.loc[hit, "value"] *= np.exp(rho_true * dlog_tax(8, 1))
    cpi.to_parquet(tmp_path / "japan_cpi2025.parquet", index=False)
    cgpi = pd.read_parquet(DATA_PROCESSED / "japan_cgpi.parquet")
    _extend(cgpi, ["series"], "index", "2027-06").to_parquet(tmp_path / "japan_cgpi.parquet", index=False)

    case = s27.load_2027(tmp_path)
    res = s27.score(case, "2027-06", "national")
    assert res["status"] == "final" and res["n_scored"] > 150
    assert abs(res["items"].observed_rho.median() - rho_true) < 0.15
    agg = res["aggregates"].set_index("statistic")
    assert abs(agg.loc["food_basket_rho", "observed"] - rho_true) < 0.15
    assert agg.loc["headline_cpi_effect_pp", "observed"] < 0      # a cut lowers the CPI
    tok = s27.score(case, "2027-04", "tokyo")
    assert tok["status"] == "interim" and tok["items"].nowcast_national_mean.notna().sum() > 150
    assert json.loads((FORECASTS / "LOCK_MANIFEST.json").read_text())["files"]   # lock untouched
    assert lock.verify_manifest(FORECASTS / "LOCK_MANIFEST.json") == []
