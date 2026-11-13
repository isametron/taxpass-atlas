# Score: japan_2019, national data through 2019-12

**Status: final.** Outcome months used: 2019-10, 2019-11, 2019-12.

## Lock

- Lock manifest `LOCK_MANIFEST.json` (sha256 `5c4e49d49c4fae4f4444999d4972939a7497304f064f1271307bbc51ddb139a6`): every locked file matches.
- Time-stamp receipt `LOCK_MANIFEST.json.ots`: MISSING.
- Replay of japan_2019: Japan consumption tax 8% -> 10%, Oct 2019, standard-rate items; forecast fit without Japan 2019

## Items

Tax change dlog(1+tau) = 0.01835. 316 of 316 forecast items had a reading.

| section | forecast | n | crps | crps_weighted | mae | coverage_50 | coverage_90 | skill_vs_rho=1 | skill_vs_rho=0 |
|---|---|---|---|---|---|---|---|---|---|
| items, national rho | locked forecast | 316 | 0.629 | 0.468 | 0.841 | 0.690 | 0.991 | -0.201 | 0.334 |
| items, national rho | full pass-through (rho=1) | 316 | 0.523 | 0.331 | 0.678 | 0.402 | 0.687 |  |  |
| items, national rho | no pass-through (rho=0) | 316 | 0.944 | 0.902 | 1.139 | 0.222 | 0.386 |  |  |

## Anticipation (reported, never scored)

Weighted mean deviation over the three months before the event: -0.063% (-0.034 in rho units, 316 items).

