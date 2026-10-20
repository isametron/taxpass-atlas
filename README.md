# TaxPass Atlas

TaxPass Atlas measures how much of a consumption tax change reaches shop prices. It estimates item-level pass-through for India's GST 2.0 cut (2025), Japan's 2019 hike and Singapore's 2023–24 hikes. On 31 Oct 2026 it locks a time-stamped forecast for Japan's food tax cut on 1 April 2027 (8% → 1%, conditional on enactment).

Work in progress. See `concordances/README.md` for what is verified.

## Setup
```bash
uv venv --python 3.12 .venv
uv pip sync requirements.lock --python .venv/Scripts/python.exe
cp .env.example .env   # add ESTAT_APP_ID, MOSPI_EMAIL, MOSPI_PASSWORD
```

## Data
```bash
python -m src.fetch.singstat          # Singapore CPI (no key)
python -m src.fetch.estat discover    # list e-Stat CPI tables
python -m src.fetch.estat cpi2025     # Japan 2025-base CPI, national + Tokyo
python -m src.fetch.mospi             # India item CPI (needs MoSPI account)
python -m src.fetch.boj               # Japan CGPI (producer prices, no key)
python -m src.fetch.wpi               # India WPI, base 2011-12 (no key)
python -m src.concord.japan           # Japan concordance + rate flags
python -m src.concord.singapore       # Singapore GST flags
python -m src.panel                   # estimation panels, one per episode
```
Raw downloads go to `data/raw/` and are git-ignored. Their URLs and SHA-256 hashes are recorded in `data/raw/MANIFEST.json`, which is committed.

## Tests
```bash
python -m pytest
```
