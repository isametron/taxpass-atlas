import pytest

from src import config

# Log tax changes from the plan, to 5 significant figures.
EXPECTED = {
    ("japan_2019", "standard"): 0.018349,
    ("japan_2019", "reduced"): 0.0,
    ("singapore_2023", "standard"): 0.0093023,
    ("singapore_2024", "standard"): 0.0092166,
    ("japan_2027", "reduced_food"): -0.067011,
}


@pytest.mark.parametrize("key,value", EXPECTED.items())
def test_episode_dlog_tax(key, value):
    ep, rate_class = key
    assert config.episode(ep).dlog(rate_class) == pytest.approx(value, abs=5e-7)


@pytest.mark.parametrize("old,new,value", [(12, 5, -0.064539), (28, 18, -0.081346), (18, 5, -0.116724)])
def test_india_slab_dlog(old, new, value):
    assert config.dlog_tax(old, new) == pytest.approx(value, abs=5e-7)


def test_india_rates_are_per_item():
    with pytest.raises(ValueError):
        config.episode("india_gst2").dlog("standard")


def test_episode_windows_are_ordered():
    for ep in config.episodes().values():
        lo, hi = ep.pre_window
        assert lo < hi < ep.base_month < ep.event_month
        for a, b in ep.exclude:
            assert a <= b


def test_japan_2027_drift_window_ends_before_anticipation_months():
    assert config.episode("japan_2027").pre_window[1] <= "2026-12"
