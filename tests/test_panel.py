import pandas as pd
import pytest

from src import panel
from src.config import episode
from src.paths import DATA_PROCESSED


def _toy(months_by_item):
    return pd.DataFrame([{"item_id": i, "month": m} for i, ms in months_by_item.items() for m in ms])


def test_complete_items_drops_items_missing_pre_or_post_months():
    ep = episode("singapore_2024")
    pre = pd.period_range(*ep.pre_window, freq="M").strftime("%Y-%m").tolist()
    post = [ep.base_month, "2024-01", "2024-02", "2024-03", "2024-04"]
    df = _toy({"full": pre + post, "new_in_2024": post, "no_h3": pre + post[:-1]})
    assert set(panel.complete_items(df, "singapore_2024").item_id) == {"full"}


def test_excluded_months_do_not_count_against_coverage():
    ep = episode("singapore_2024")
    a, b = ep.exclude[0]
    pre = [m for m in pd.period_range(*ep.pre_window, freq="M").strftime("%Y-%m") if not a <= m <= b]
    post = [ep.base_month, "2024-01", "2024-02", "2024-03", "2024-04"]
    assert set(panel.complete_items(_toy({"x": pre + post}), "singapore_2024", 1.0).item_id) == {"x"}


@pytest.mark.parametrize("ep", ["singapore_2023", "singapore_2024"])
def test_built_panel_schema(ep):
    path = DATA_PROCESSED / f"panel_{ep}.parquet"
    if not path.exists():
        pytest.skip("run `python -m src.panel` first")
    df = pd.read_parquet(path)
    assert list(df.columns) == panel.COLUMNS
    assert set(df.role) <= {"treated", "control", "excluded"}
    assert (df.loc[df.role != "treated", "dlog_tax"] == 0).all()
    assert (df.loc[df.role == "treated", "dlog_tax"] > 0).all()
    assert not df.duplicated(["item_id", "month"]).any()
