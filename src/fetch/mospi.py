"""India all-India item-level CPI (base 2012) from the MoSPI API platform.

Needs a free MoSPI API account, with MOSPI_EMAIL and MOSPI_PASSWORD in .env. The token
lasts 15 minutes, which is enough for one full pull. The manual limits unauthenticated
requests to 10 records, so this script refuses to run without a token.

  python -m src.fetch.mospi --signup   create the account from the details in .env (once)
  python -m src.fetch.mospi            fetch 2014-01..2025-12 item indices
  python -m src.fetch.mospi --offline  parse data/raw only

Credentials are read from .env and are never printed or written to data/raw.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

from src.config import sources
from src.fetch.manifest import write_json
from src.paths import DATA_PROCESSED, DATA_RAW, ROOT

RAW_DIR = DATA_RAW / "india"
YEARS = range(2014, 2026)
PAGE_SIZE = 200  # the API rejects larger limits (HTTP 400)
REQUEST_PAUSE = 1.0  # seconds between requests; about 108 unpaced requests triggered HTTP 429
RATE_LIMIT_WAIT = 65  # the 429 message says to retry after 1 minute
MAX_RATE_LIMIT_WAITS = 5
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}


class _LegacyRenegotiationAdapter(HTTPAdapter):
    """api.mospi.gov.in only supports pre-RFC 5746 TLS renegotiation, which OpenSSL 3
    refuses by default (UNSAFE_LEGACY_RENEGOTIATION_DISABLED). Allow it for this host
    alone; certificate and hostname verification stay on."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def session() -> requests.Session:
    s = requests.Session()
    s.mount("https://api.mospi.gov.in/", _LegacyRenegotiationAdapter())
    return s


HTTP = session()


def _env(name: str) -> str:
    load_dotenv(ROOT / ".env")
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set in .env.")
    return value


def _url(endpoint: str) -> str:
    cfg = sources()["india"]
    return cfg["mospi_api"] + cfg["endpoints"][endpoint]


def signup_body() -> dict:
    """Signup payload. The live API wants the email as the username ("In username please
    enter a valid email id"), unlike the manual's example; organization and purpose
    must not contain special characters."""
    email = _env("MOSPI_EMAIL")
    fields = {"organization": _env("MOSPI_ORGANIZATION"), "purpose": _env("MOSPI_PURPOSE")}
    for name, value in fields.items():
        if not re.fullmatch(r"[A-Za-z0-9 ]+", value):
            raise SystemExit(f"{name} may contain only letters, digits and spaces.")
    body = {"username": email, "email": email, "password": _env("MOSPI_PASSWORD"),
            "gender": _env("MOSPI_GENDER"), **fields}
    # The live form has grown beyond the manual: any field the server reports as blank
    # can be added in .env as MOSPI_FIELD_<name>=value without a code change.
    body.update(_extra_fields())
    return body


def _extra_fields() -> dict:
    load_dotenv(ROOT / ".env")
    return {k[len("MOSPI_FIELD_"):].lower(): v.strip() for k, v in os.environ.items()
            if k.upper().startswith("MOSPI_FIELD_") and v.strip()}


def signup() -> None:
    r = HTTP.post(_url("signup"), json=signup_body(), timeout=60)
    print(f"HTTP {r.status_code}: {_message(r)}")
    r.raise_for_status()


def _message(r: requests.Response) -> str:
    """The server's status message only, so a token in the body is never printed."""
    try:
        return str(r.json().get("msg") or r.json().get("message") or "(no message)")
    except ValueError:
        return "(response was not JSON)"


TOKEN_KEYS = ("token", "Token", "accessToken", "access_token", "jwt")


def _looks_like_token(text: str) -> bool:
    """A JWT, or an opaque string that is long and has no spaces (so not a status message)."""
    return (text.count(".") == 2 or len(text) >= 20) and not any(c.isspace() for c in text)


def extract_token(body: dict) -> str | None:
    """Find the token in a login response. The live API returns {msg, response, statusCode},
    with the token either as `response` itself or inside it."""
    if body.get("statusCode") is False:
        return None
    for container in ("response", "data"):
        inner = body.get(container)
        if isinstance(inner, str) and _looks_like_token(inner):
            return inner
        if isinstance(inner, dict):
            for key in TOKEN_KEYS:
                if isinstance(inner.get(key), str) and inner[key]:
                    return inner[key]
    for key in TOKEN_KEYS:
        if isinstance(body.get(key), str) and body[key]:
            return body[key]
    return None


def login_failure_detail(body: dict) -> str:
    """The server's message and flags, plus only the size of `response`, never its text."""
    inner = body.get("response")
    size = f"{len(inner)} chars, spaces: {any(c.isspace() for c in inner)}" if isinstance(inner, str) else type(inner).__name__
    return f"Server said: {body.get('msg')!r}; statusCode: {body.get('statusCode')}; response: {size}."


def describe_shape(obj, depth: int = 0) -> str:
    """Keys and value types only, never values, so a failed lookup can be reported safely."""
    if isinstance(obj, dict) and depth < 2:
        return "{" + ", ".join(f"{k}: {describe_shape(v, depth + 1)}" for k, v in sorted(obj.items())) + "}"
    if isinstance(obj, list):
        return f"list[{len(obj)}]"
    return type(obj).__name__


def _token() -> str:
    # Signup takes the email as `username`; login answers 500 when a field it expects is
    # missing, so send the email under both keys.
    email = _env("MOSPI_EMAIL")
    r = HTTP.post(_url("login"), json={"username": email, "email": email, "password": _env("MOSPI_PASSWORD")},
                  timeout=60)
    if not r.ok:
        raise SystemExit(f"MoSPI login failed: HTTP {r.status_code}: {_message(r)}")
    body = r.json()
    token = extract_token(body)
    if not token:
        raise SystemExit(f"MoSPI login returned no token. {login_failure_detail(body)}")
    return token


def fetch_year(year: int, get) -> list[dict]:
    """All records for one year. `get(params)` returns the decoded JSON body."""
    records: list[dict] = []
    page = 1
    while True:
        body = get({"base_year": 2012, "year": year, "Format": "JSON", "limit": PAGE_SIZE, "page": page})
        records.extend(body["data"])
        meta = body.get("meta_data", {})
        if page >= meta.get("totalPages", 1):
            break
        page += 1
    expected = meta.get("totalRecords")
    if expected is not None and len(records) != expected:
        raise RuntimeError(f"{year}: got {len(records)} records, API reports {expected}")
    return records


def polite_send(send, sleep=time.sleep) -> requests.Response:
    """Pace requests and honour HTTP 429 ("try again after 1 minute") by waiting it out."""
    sleep(REQUEST_PAUSE)
    for _ in range(MAX_RATE_LIMIT_WAITS):
        r = send()
        if r.status_code != 429:
            return r
        print(f"Rate limited by MoSPI; waiting {RATE_LIMIT_WAIT}s as it asks...", flush=True)
        sleep(RATE_LIMIT_WAIT)
    return send()


def fetch_items() -> list[dict]:
    url = _url("item_index")
    token = _token()
    # The manual doesn't say whether the header takes a Bearer prefix; try it first.
    schemes = [f"Bearer {token}", token]

    def get(params: dict) -> dict:
        while True:
            r = polite_send(lambda: HTTP.get(url, params=params, headers={"Authorization": schemes[0]}, timeout=120))
            if r.status_code in (401, 403) and len(schemes) > 1:
                schemes.pop(0)
                continue
            if not r.ok:
                raise SystemExit(f"MoSPI item fetch failed: HTTP {r.status_code}: {_message(r)}")
            return r.json()

    records: list[dict] = []
    for year in YEARS:
        got = fetch_year(year, get)
        print(f"{year}: {len(got)} records")
        records.extend(got)
    write_json(RAW_DIR / "cpi_item_index_2012.json", records, url, {"years": f"{YEARS.start}-{YEARS.stop - 1}"})
    return records


def tidy(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df[df["baseyear"].astype(str) == "2012"].copy()
    df["month"] = [f"{int(y):04d}-{MONTHS[m]:02d}" for y, m in zip(df["year"], df["month"])]
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    out = df.rename(columns={"item": "item_name"})[["item_name", "month", "index", "status"]]
    return out.dropna(subset=["index"]).sort_values(["item_name", "month"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--signup", action="store_true", help="create the MoSPI account, then exit")
    args = ap.parse_args()
    if args.signup:
        signup()
        return
    if args.offline:
        records = json.loads((RAW_DIR / "cpi_item_index_2012.json").read_text(encoding="utf-8"))
    else:
        records = fetch_items()
    df = tidy(records)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED / "india_cpi_items.parquet", index=False)
    print(f"{df.item_name.nunique()} items, {df.month.min()}..{df.month.max()}")


if __name__ == "__main__":
    main()
