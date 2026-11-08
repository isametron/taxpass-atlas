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
    body = {"RESULTSET": [{"SERIES_CODE": "PRCG20_1200120001",
                           "VALUES": {"SURVEY_DATES": [201909, 201910], "VALUES": [101.2, None]}}]}
    df = boj.tidy(body)
    assert df.to_dict("records") == [{"series": "beverages_and_foods", "series_code": "PRCG20_1200120001",
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


def test_mospi_fetch_year_pages_and_checks_total():
    from src.fetch import mospi
    pages = {1: ["a", "b"], 2: ["c"]}
    seen = []

    def get(params):
        seen.append(params)
        return {"data": pages[params["page"]], "meta_data": {"totalPages": 2, "totalRecords": 3}}

    assert mospi.fetch_year(2025, get) == ["a", "b", "c"]
    assert [p["page"] for p in seen] == [1, 2]
    assert all(p["year"] == 2025 for p in seen)


def test_mospi_fetch_year_rejects_short_download():
    from src.fetch import mospi
    get = lambda params: {"data": ["a"], "meta_data": {"totalPages": 1, "totalRecords": 2}}
    with pytest.raises(RuntimeError, match="got 1 records"):
        mospi.fetch_year(2025, get)


def test_mospi_extract_token_shapes():
    from src.fetch import mospi
    assert mospi.extract_token({"token": "t"}) == "t"
    assert mospi.extract_token({"data": {"accessToken": "t"}}) == "t"
    assert mospi.extract_token({"data": [], "msg": "nope"}) is None
    assert mospi.extract_token({"msg": "ok", "statusCode": True, "response": "aaa.bbb.ccc"}) == "aaa.bbb.ccc"
    assert mospi.extract_token({"msg": "ok", "response": {"token": "t"}}) == "t"
    assert mospi.extract_token({"msg": "ok", "response": "Login successful"}) is None


def test_mospi_describe_shape_hides_values():
    from src.fetch import mospi
    shape = mospi.describe_shape({"msg": "secret-ish", "response": {"token": "abc.def.ghi", "user": {"id": 1}}})
    assert "abc" not in shape and "secret" not in shape
    assert shape == "{msg: str, response: {token: str, user: dict}}"  # stops two levels down


def test_mospi_session_scopes_legacy_tls_to_mospi_only():
    import ssl
    from src.fetch import mospi
    s = mospi.session()
    adapter = s.get_adapter("https://api.mospi.gov.in/api/users/login")
    assert isinstance(adapter, mospi._LegacyRenegotiationAdapter)
    ctx = adapter.poolmanager.connection_pool_kw["ssl_context"]
    assert ctx.options & ssl.OP_LEGACY_SERVER_CONNECT
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    assert not isinstance(s.get_adapter("https://api.e-stat.go.jp/"), mospi._LegacyRenegotiationAdapter)


def test_mospi_signup_uses_email_as_username(monkeypatch):
    from src.fetch import mospi
    env = {"MOSPI_EMAIL": "a.b@example.com", "MOSPI_PASSWORD": "x", "MOSPI_ORGANIZATION": "BMS College",
           "MOSPI_PURPOSE": "Academic research", "MOSPI_GENDER": "G"}
    monkeypatch.setattr(mospi, "_env", lambda name: env[name])
    monkeypatch.setenv("MOSPI_FIELD_MOBILE", "123")
    body = mospi.signup_body()
    assert body["username"] == body["email"] == "a.b@example.com"
    assert body["gender"] == "G" and body["mobile"] == "123"
    env["MOSPI_PURPOSE"] = "research!"
    with pytest.raises(SystemExit):
        mospi.signup_body()


def test_mospi_extract_token_accepts_opaque_string_only_on_success():
    from src.fetch import mospi
    opaque = "a1B2c3D4e5F6g7H8i9J0k1L2"
    assert mospi.extract_token({"msg": "ok", "statusCode": True, "response": opaque}) == opaque
    assert mospi.extract_token({"msg": "bad", "statusCode": False, "response": opaque}) is None
    assert mospi.extract_token({"msg": "ok", "statusCode": True, "response": "Please verify your email address first"}) is None


def test_mospi_login_failure_detail_hides_response_text():
    from src.fetch import mospi
    detail = mospi.login_failure_detail({"msg": "Invalid password", "statusCode": False, "response": "hidden-value"})
    assert "Invalid password" in detail and "12 chars" in detail and "hidden" not in detail


def test_mospi_page_size_within_api_limit():
    from src.fetch import mospi
    assert mospi.PAGE_SIZE <= 200


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def test_mospi_polite_send_waits_out_rate_limit():
    from src.fetch import mospi
    replies = iter([_Resp(429), _Resp(429), _Resp(200)])
    waits = []
    r = mospi.polite_send(lambda: next(replies), sleep=waits.append)
    assert r.status_code == 200
    assert waits == [mospi.REQUEST_PAUSE, mospi.RATE_LIMIT_WAIT, mospi.RATE_LIMIT_WAIT]


def test_mospi_polite_send_gives_up_after_limit():
    from src.fetch import mospi
    calls = []

    def send():
        calls.append(1)
        return _Resp(429)

    assert mospi.polite_send(send, sleep=lambda s: None).status_code == 429
    assert len(calls) == mospi.MAX_RATE_LIMIT_WAITS + 1
