"""Repository paths. Everything is resolved relative to the repo root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
CONCORDANCES = ROOT / "concordances"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FORECASTS = ROOT / "forecasts"
REPORTS = ROOT / "reports"
PAPER = ROOT / "paper"
