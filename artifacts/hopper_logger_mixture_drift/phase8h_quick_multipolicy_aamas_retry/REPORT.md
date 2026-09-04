# Phase 8H-Q — Quick Action-Wise Multi-Policy AAMAS Envelope Gate

This is a quick exploratory gate. Neural outputs are **approximate AAMAS upper backups**, not certified bounds.

## Direct answers

1. Action-level minimum beats balanced pooled AAMAS on all three primary metrics: **True**.
2. Action-level minimum beats state-level minimum on all three primary metrics: **False**. The algebraic potential inequality is checked separately.
3. Candidate-set and envelope effects are separated by `pooled_aamas_native` versus `pooled_aamas_union`, then `pooled_aamas_union` versus `action_level_min` in `method_metrics.csv`.
4. Maximum pooled-composition prediction MAE: **0.79988211**.
5. The source-wise envelope is composition invariant by construction and passed explicit duplication/composition checks.
6. Mean fraction of states whose candidate actions select more than one minimizing source: **71.861%**.
7. The independent-latents control is reported separately; any retained gain there is descriptive evidence of ordinary coverage rather than hidden-confounding recovery.
8. Promotion to confidence correction, full potential training, or short-budget SAC requires scientific review of these metrics; this code applies no automatic success threshold.

## Primary metric snapshot

| method | do MAE | ranking disagreement | mean regret | phi MAE |
|---|---:|---:|---:|---:|
| action-level minimum | 2.3723346 | 0.9004329 | 0.80837928 | 3.0398432 |
| state-level minimum | 2.4923018 | 0.9047619 | 0.80836937 | 3.1806472 |
| balanced pooled union | 3.1832864 | 0.90909091 | 0.83940766 | 4.3581651 |

## Limits

The unit of replication is the model seed. Anchors are fixed, the negative control has one seed, no finite-sample confidence correction is used, and no online policy or long-horizon return is evaluated.
