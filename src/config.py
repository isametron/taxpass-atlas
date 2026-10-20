"""Load episode and source configuration.

Log tax changes are derived here from statutory rates and nowhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml

from src.paths import CONFIG


def dlog_tax(old_pct: float, new_pct: float) -> float:
    """Log change in the tax-inclusive price factor, ln((1+new)/(1+old))."""
    return math.log((1 + new_pct / 100) / (1 + old_pct / 100))


@dataclass(frozen=True)
class Episode:
    name: str
    country: str
    base_month: str
    event_month: str
    pre_window: tuple[str, str]
    horizons: tuple[int, ...]
    headline_h: int
    direction: str
    tax_inclusive_display_required: bool
    rates: dict[str, tuple[float, float]] | None  # None when rates are per item
    partial_months: tuple[str, ...] = ()
    exclude: tuple[tuple[str, str], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def dlog(self, rate_class: str) -> float:
        if self.rates is None:
            raise ValueError(f"{self.name} has per-item rates; use the slab map")
        old, new = self.rates[rate_class]
        return dlog_tax(old, new)


def _load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _episode(name: str, d: dict[str, Any]) -> Episode:
    rates = d["rates"]
    if rates == "per_item":
        rates = None
    else:
        rates = {k: (float(v[0]), float(v[1])) for k, v in rates.items()}
    return Episode(
        name=name,
        country=d["country"],
        base_month=str(d["base_month"]),
        event_month=str(d["event_month"]),
        pre_window=(str(d["pre_window"][0]), str(d["pre_window"][1])),
        horizons=tuple(d["horizons"]),
        headline_h=int(d["headline_h"]),
        direction=d["direction"],
        tax_inclusive_display_required=bool(d["tax_inclusive_display_required"]),
        rates=rates,
        partial_months=tuple(str(m) for m in d.get("partial_months", [])),
        exclude=tuple((str(a), str(b)) for a, b in d.get("exclude", [])),
        raw=d,
    )


@lru_cache(maxsize=None)
def episodes() -> dict[str, Episode]:
    return {name: _episode(name, d) for name, d in _load_yaml("episodes.yaml").items()}


def episode(name: str) -> Episode:
    return episodes()[name]


@lru_cache(maxsize=None)
def sources() -> dict[str, Any]:
    return _load_yaml("sources.yaml")
