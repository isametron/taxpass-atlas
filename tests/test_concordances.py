import pandas as pd
import pytest

from src.paths import CONCORDANCES

JP = CONCORDANCES / "japan_cpi_2020_to_2025.csv"
RATES = CONCORDANCES / "japan_rate_flags.csv"

pytestmark = pytest.mark.skipif(not JP.exists(), reason="run `python -m src.concord.japan` first")


@pytest.fixture(scope="module")
def conc():
    return pd.read_csv(JP, dtype=str)


@pytest.fixture(scope="module")
def rates():
    return pd.read_csv(RATES, dtype={"code_2025": str, "code_2020": str})


def test_every_2025_item_has_rate_flags(conc, rates):
    items_2025 = set(conc.loc[(conc.level == "item") & conc.code_2025.notna(), "code_2025"])
    assert items_2025 == set(rates.code_2025)


def test_every_2027_target_links_to_2020_or_is_flagged_new(conc, rates):
    targets = set(rates.loc[rates.rate_class == "reduced_food", "code_2025"])
    rel = conc.set_index("code_2025").relation
    unlinked = {c for c in targets if rel[c] not in {"identity", "renamed", "new"}}
    assert not unlinked


def test_2027_targets_get_the_statutory_cut(rates):
    food = rates[rates.rate_class == "reduced_food"]
    assert (food.rate_2027_old == 8).all() and (food.rate_2027_new == 1).all()
    assert (food.rate_2019_old == food.rate_2019_new).all()  # food was the 2019 control group


def test_wedge_pairs_have_both_sides(rates):
    pairs = rates[rates.wedge_pair.notna()].groupby("wedge_pair").wedge_side.agg(set)
    assert all(s == {"grocery", "eat_in"} for s in pairs)
    real = rates[rates.wedge_pair.notna() & (rates.wedge_pair != "beer_placebo")]
    eat_in = real[real.wedge_side == "eat_in"]
    assert (eat_in.rate_2027_new == 10).all()


def test_exempt_items_have_no_tax_change(rates):
    ex = rates[rates.rate_class == "exempt"]
    assert (ex[["rate_2019_old", "rate_2019_new", "rate_2027_old", "rate_2027_new"]] == 0).all().all()


INDIA = CONCORDANCES / "india_gst2_slab_map.csv"


def test_india_slab_status_matches_rates():
    d = pd.read_csv(INDIA)
    assert d.item_code.is_unique
    assert (d.loc[d.status == "cut", "rate_new"] < d.loc[d.status == "cut", "rate_old"]).all()
    assert (d.loc[d.status == "hike", "rate_new"] > d.loc[d.status == "hike", "rate_old"]).all()
    same = d.status.isin(["unchanged", "exempt", "administered"])
    assert (d.loc[same, "rate_new"] == d.loc[same, "rate_old"]).all()
    assert set(d.confidence) <= {"high", "medium", "low"}


SG = CONCORDANCES / "singapore_gst_flags.csv"


def test_singapore_flags_cover_controls():
    d = pd.read_csv(SG, dtype=str)
    assert d.series_no.is_unique
    assert {"exempt", "gst_absorbed", "standard"} == set(d.gst_status)
