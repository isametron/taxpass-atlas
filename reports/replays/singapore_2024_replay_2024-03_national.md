# Score: singapore_2024, national data through 2024-03

**Status: final.** Outcome months used: 2024-01, 2024-02, 2024-03.

## Lock

- Lock manifest `LOCK_MANIFEST.json` (sha256 `4fb8ab63a8dd28fc4ff6e55d8d01e4cbd50ab758f07b9cf779e40cdc8d735967`): every locked file matches.
- Time-stamp receipt `LOCK_MANIFEST.json.ots`: MISSING.
- Replay of singapore_2024: Singapore GST 8% -> 9%, Jan 2024; forecast fit without Singapore 2024

## Items

Tax change dlog(1+tau) = 0.00922. 96 of 96 forecast items had a reading.

| section | forecast | n | crps | crps_weighted | mae | coverage_50 | coverage_90 | skill_vs_rho=1 | skill_vs_rho=0 |
|---|---|---|---|---|---|---|---|---|---|
| items, national rho | locked forecast | 96 | 2.149 | 1.244 | 2.936 | 0.417 | 0.792 | 0.015 | -0.070 |
| items, national rho | full pass-through (rho=1) | 96 | 2.183 | 1.291 | 2.958 | 0.375 | 0.740 |  |  |
| items, national rho | no pass-through (rho=0) | 96 | 2.010 | 1.181 | 2.716 | 0.375 | 0.771 |  |  |

## Anticipation (reported, never scored)

Weighted mean deviation over the three months before the event: -0.539% (-0.584 in rho units, 96 items).

