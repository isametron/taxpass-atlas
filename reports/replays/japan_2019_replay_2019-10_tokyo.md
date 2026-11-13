# Score: japan_2019, tokyo data through 2019-10

**Status: interim.** Outcome months used: 2019-10.

- Interim: 1 of 3 outcome months are out. extraction_se was measured for the full 3-month mean, so interim intervals are approximate.
- Provisional: scored against Tokyo ward-area prices with the locked Tokyo noise added. The protocol scores National data; the Tokyo nowcast columns are the protocol's update rule.

## Lock

- Lock manifest `LOCK_MANIFEST.json` (sha256 `5c4e49d49c4fae4f4444999d4972939a7497304f064f1271307bbc51ddb139a6`): every locked file matches.
- Time-stamp receipt `LOCK_MANIFEST.json.ots`: MISSING.
- Replay of japan_2019: Japan consumption tax 8% -> 10%, Oct 2019, standard-rate items; forecast fit without Japan 2019

## Items

Tax change dlog(1+tau) = 0.01835. 316 of 316 forecast items had a reading.

| section | forecast | n | crps | crps_weighted | mae | coverage_50 | coverage_90 | skill_vs_rho=1 | skill_vs_rho=0 |
|---|---|---|---|---|---|---|---|---|---|
| items, tokyo rho | locked forecast | 316 | 0.882 | 0.628 | 1.161 | 0.665 | 0.930 | -0.117 | 0.210 |
| items, tokyo rho | full pass-through (rho=1) | 316 | 0.790 | 0.513 | 1.012 | 0.532 | 0.769 |  |  |
| items, tokyo rho | no pass-through (rho=0) | 316 | 1.118 | 0.966 | 1.412 | 0.310 | 0.576 |  |  |

## Anticipation (reported, never scored)

Weighted mean deviation over the three months before the event: -0.160% (-0.087 in rho units, 316 items).

