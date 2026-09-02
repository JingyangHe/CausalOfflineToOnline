# Phase 8D-PRIC Report

This is an exploratory algorithm-development experiment because aggregate results on the
pre-existing test split had been viewed before Phase 8D. Confirmatory claims require fresh
anchor and policy seeds.

## Scope

The deployable method is a binary-latent, reward-only proof of concept. It does not establish
continuous-confounder recovery, transition-model validity, or end-to-end SAC improvement.

## Reproduction

Phase 8C-FD was reproduced before training: V0 collapsed in 35/35
primary cells and V6 collapsed in 0/35 cells.

## Main method table

| Method | Collapse | Val NLL | Do MAE | Rank error | Regret | Reward separation | Behavior error |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled_mlp | — | -1.61488 | 0.019127 | 0.623763 | 0.00389176 | — | — |
| V0_random_init_mechanism | 1 | -0.960455 | 0.0186133 | 0.629773 | 0.00389849 | 0.00788306 | 0.193362 |
| V1_explicit_collapsed | — | -0.88808 | 0.0208428 | 0.620989 | 0.0038417 | — | — |
| public_residual_init_nll_best | 0 | -1.37131 | 0.0135472 | 0.448451 | 0.00229353 | 0.0636266 | 0.0826672 |
| public_residual_init_uniform_candidate_ensemble | — | -1.22179 | 0.0168403 | 0.545354 | 0.00301179 | — | — |
| public_residual_init_intervention_calibrated | — | — | 0.0127865 | 0.452173 | 0.00232432 | — | — |
| source_shuffle_initialization | 0 | -1.04408 | 0.0136391 | 0.453537 | 0.00234042 | 0.0640597 | 0.318149 |
| no_staged_training | 0 | -1.33696 | 0.0125865 | 0.442071 | 0.00228926 | 0.0511918 | 0.0857631 |
| V6_oracle_initialized_joint | 0 | -1.3367 | 0.0112291 | 0.311974 | 0.00130919 | 0.0495886 | 0.00433781 |
| V7_oracle_u_aware | — | — | 0.0111627 | 0.230143 | 0.0007858 | — | — |

## Calibration table

| Calibration B | Do MAE | Rank error | Regret | Gap to Oracle-best | Candidate entropy |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0168403 | 0.545354 | 0.00301179 | 0.0050062 | 2.30259 |
| 8 | 0.0138867 | 0.460855 | 0.00238844 | 0.00205264 | 1.84394 |
| 16 | 0.0132544 | 0.45473 | 0.00234147 | 0.00142028 | 1.64165 |
| 32 | 0.0131926 | 0.451581 | 0.00231805 | 0.0013585 | 1.42451 |
| 64 | 0.012955 | 0.451447 | 0.0023157 | 0.00112095 | 1.21867 |
| 128 | 0.0127865 | 0.452173 | 0.00232432 | 0.000952419 | 0.945642 |

## Direct answers

1. Public initialization collapse is 0 versus V0 1;
   its observed collapse direction is lower.
2. By do MAE it is closer to V6 than V0.
3. Source-shuffle do MAE is 0.0136391 versus public initialization 0.0135472; no mechanism claim follows if these are similar.
4. No-staging do MAE is 0.0125865 versus staged 0.0135472; no staging claim follows if these are similar.
5. `candidate_selection_metrics.csv` reports the post-hoc Oracle rank of every selected candidate.
6. Observational-NLL choice and intervention-calibrated choice are reported separately; agreement is not assumed.
7. The first tested positive budget with lower mean do MAE than B=0 is 8.
8. The three calibration curves report do MAE, rank error, and regret without a success threshold.
9. Public, source-shuffle, no-staging, B=0, and B>0 rows separate initialization from intervention selection.
10. This reward-only evidence alone does not authorize extension claims for transition learning or SAC.

## Interpretation boundary

No numerical success threshold was imposed. Inspect `summary.json`, `seed_metrics.csv`, and
`calibration_budget_metrics.csv` to determine whether public initialization reduced collapse,
whether intervention calibration improved causal selection, and whether the source-shuffle and
no-staging controls retain the effect. Oracle V6/V7 remain diagnostic ceilings only.
