import pytest

from src.fetch import estat, singstat


def test_estat_time_code():
    assert estat.parse_time("2019001010") == "2019-10"
    assert estat.parse_time("2025000101") == "2025-01"
    with pytest.raises(ValueError):
        estat.parse_time("2019000000")  # annual code


def _estat_page(values, next_key=None):
    result_inf = {"NEXT_KEY": next_key} if next_key else {}
    return {"GET_STATS_DATA": {"STATISTICAL_DATA": {
        "RESULT_INF": result_inf,
        "CLASS_INF": {"CLASS_OBJ": [
            {"@id": "cat01", "@name": "品目", "CLASS": [
                {"@code": "0001", "@name": "総合", "@level": "1"},
                {"@code": "1001", "@name": "うるち米Ａ", "@level": "5", "@parentCode": "0004"},
            ]},
            {"@id": "area", "@name": "地域", "CLASS": {"@code": "00000", "@name": "全国"}},
        ]},
        "DATA_INF": {"VALUE": values},
    }}}


def test_estat_tidy_merges_pages_and_skips_missing():
    v = lambda item, t, x: {"@tab": "1", "@cat01": item, "@area": "00000", "@time": t, "$": x}
    pages = [
        _estat_page([v("0001", "2025000101", "100.1"), v("1001", "2025000101", "-")], next_key="3"),
        _estat_page(v("1001", "2025000202", "98.7")),  # single VALUE comes back as a dict
    ]
    values, classes = estat.tidy(pages)
    assert list(values.month) == ["2025-01", "2025-02"]
    assert list(values.value) == [100.1, 98.7]
    assert set(classes.class_id) == {"cat01", "area"}
    assert classes.loc[classes.code == "1001", "parent"].item() == "0004"


def test_singstat_tidy_marks_leaves():
    raw = {"series": {
        "1": {"name": "All Items", "obs": {"2023 Jan": "100.0"}},
        "1.01": {"name": "Food", "obs": {"2023 Jan": "101.5"}},
        "1.01.1": {"name": "Bread", "obs": {"2023 Jan": "102.0", "2023 Feb": "na"}},
    }}
    df = singstat.tidy(raw)
    assert df.set_index("series_no").is_leaf.to_dict() == {"1": False, "1.01": False, "1.01.1": True}
    assert df.month.unique().tolist() == ["2023-01"]


def test_boj_tidy_maps_series_names():
    from src.fetch import boj
    body = {"RESULTSET": [{"SERIES_CODE": "PRCG20_2200120001",
                           "VALUES": {"SURVEY_DATES": [201909, 201910], "VALUES": [101.2, None]}}]}
    df = boj.tidy(body)
    assert df.to_dict("records") == [{"series": "beverages_and_foods", "series_code": "PRCG20_2200120001",
                                      "month": "2019-09", "index": 101.2}]


def test_wpi_tidy_reshapes_wide_file(tmp_path):
    import pandas as pd
    from src.fetch import wpi
    wide = pd.DataFrame({"COMM_NAME": ["All commodities"], "COMM_CODE": ["1000000000"], "COMM_WT": [100],
                         "INDX082025": [154.1], "INDX092025": [""]})
    path = tmp_path / "w.xlsx"
    wide.to_excel(path, index=False)
    df = wpi.tidy(path)
    assert df[["comm_code", "month", "index"]].to_dict("records") == [
        {"comm_code": "1000000000", "month": "2025-08", "index": 154.1}]
