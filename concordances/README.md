# Concordances

| File | Built by | Source | Status |
|---|---|---|---|
| `japan_cpi_2020_to_2025.csv` | `python -m src.concord.japan` | e-Stat code list `2025DBcode.xlsx` (2020 and 2025 codes side by side) | Generated. Checked 2026-11-08 against the 2025-01..2026-09 overlap: continuing items have median month-on-month correlation 0.996; the 3 notable renames link cleanly (see `note`). |
| `japan_cpi_area_codes.csv` | same | same | Generated |
| `japan_rate_flags.csv` | same (rules in `src/concord/japan.py`) | Consumption Tax Act reduced-rate rules | Matches e-Stat's item hierarchy (test `test_rate_flags_match_estat_hierarchy`); Oct 2019 Sep→Oct median change: standard +1.85% vs statutory +1.835%, reduced food 0.00% |
| `singapore_gst_flags.csv` | `python -m src.concord.singapore` | IRAS exempt-supply rules; government GST absorption | Draft: 3 items marked `review` |
| `india_gst2_slab_map.csv` | edited by hand (first draft from MoSPI item codes) | New rates: notifications 9/2025-CT(Rate) (goods slabs), 10/2025-CT(Rate) (nil goods), 15/2025-CT(Rate) (services, amends 11/2017), 2/2025-Compensation Cess (Rate), all effective 2025-09-22. Old rates: GSTC-56 Annexure I (PIB 03.09.2025, PRID 2163555). | Verified 2026-10-20: all 299 rows checked. `note` cites the entry, e.g. `N9 I/48` = 9/2025 Schedule I S.No. 48, `N10/67` = 10/2025 S.No. 67, `A-I:1902` = Council old→new entry for that HSN, `N15 7(i)` = 11/2017 S.No. 7(i) as amended by 15/2025. Price- or variant-dependent items (apparel, footwear, made-ups, TVs, cars, etc.) are `mixed`. `low`-confidence and `mixed` rows are left out of ρ estimation. Tobacco keeps its old rates and cess through 2025 (the move to 40% was deferred); its 8 rows are set to `low` so they are not used as controls, because prices may have moved early ahead of the 40% rate. |

## Conventions
- Rates are statutory percentages. `src/config.py:dlog_tax` turns them into log changes.
- `status`/`rate_class` values:
  - `exempt` and `gst_absorbed`: no statutory tax change. They are controls in Singapore and India. In Japan 2019 only reduced-rate items are controls, because exempt prices carry non-deductible input tax and some were revised for the hike.
  - `administered`: dropped from both estimation and targets.
  - `mixed`: the item's rate depends on price or variant. These are dropped.
- Japan 2027: newspapers (`reduced_newspaper`) have `rate_2027_new` empty because their treatment is not legislated yet. They are reported separately and never scored.
