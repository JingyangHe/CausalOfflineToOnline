# Phase 8A-NC-LH: Long-Horizon Causal Consequence Audit

## Estimand and design

This audit estimates a **fixed-policy finite-horizon intervention value**. The first commanded
action is fixed by the Phase 8A anchor table. Thereafter the
hidden-blind continuation is `0.5*(pi_500k([o,-1],deterministic=True)+pi_500k([o,+1],deterministic=True))`. Future hidden variables are iid balanced and
integrated exactly at H=5 or with common-random-number antithetic Monte Carlo at H=20/50.

Gamma is 0.99 and its recorded source is `explicit_cli`. The primary
cross-horizon population contains 24 anchors that have enough
TimeLimit steps for the maximum requested horizon. Per-horizon eligible counts are
{'1': 24, '5': 24}.

## Integrity and numerical validation

All 35 hard checks passed. H=1 reproduced Phase 8A-NC with maximum absolute
difference 1.86517e-14. H=5 used all 16
equiprobable future-U sequences. Inputs were unchanged by SHA256 before and after analysis.

## Balanced observational-do return error

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 24 | 4.93432e-17 | [2.46716e-17, 8.34209e-17] |
| 0 | 5 | logger12_balanced | 24 | 2.71388e-16 | [1.46796e-16, 4.68761e-16] |
| 0.3 | 1 | logger12_balanced | 24 | 0.00188493 | [0.00150264, 0.00224789] |
| 0.3 | 5 | logger12_balanced | 24 | 0.00971464 | [0.00729698, 0.011561] |

## Initial-U effect and heavy-mixture drift

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | none | 24 | 6.16791e-18 | [0, 1.85037e-17] |
| 0 | 5 | none | 24 | 4.93432e-17 | [0, 1.23358e-16] |
| 0.3 | 1 | none | 24 | 0.00972368 | [0.00803152, 0.0113085] |
| 0.3 | 5 | none | 24 | 0.0495445 | [0.037944, 0.0614016] |

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | none | 24 | 2.62136e-17 | [4.54883e-18, 5.87493e-17] |
| 0 | 5 | none | 24 | 2.59052e-16 | [1.4803e-16, 4.13558e-16] |
| 0.3 | 1 | none | 24 | 0.000977371 | [0.000789701, 0.00119502] |
| 0.3 | 5 | none | 24 | 0.00503722 | [0.00373971, 0.00661598] |

## Do decision scale

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | none | 24 | 0.00700736 | [0.00564525, 0.00840895] |
| 0 | 5 | none | 24 | 0.0373947 | [0.0270124, 0.0506608] |
| 0.3 | 1 | none | 24 | 0.00586708 | [0.00452522, 0.00690685] |
| 0.3 | 5 | none | 24 | 0.0298286 | [0.0221172, 0.0393304] |

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | none | 24 | 0.00337427 | [0.00280731, 0.0040271] |
| 0 | 5 | none | 24 | 0.0157247 | [0.0116709, 0.0196309] |
| 0.3 | 1 | none | 24 | 0.00279332 | [0.00190146, 0.00338393] |
| 0.3 | 5 | none | 24 | 0.0141311 | [0.0105903, 0.0176408] |

## Ranking disagreement

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 24 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 24 | 0 | [0, 0] |
| 0.3 | 1 | logger12_balanced | 24 | 0.0833333 | [0, 0.166667] |
| 0.3 | 5 | logger12_balanced | 24 | 0 | [0, 0] |

## Best-case decision regret under balanced observational selection

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger1_heavy | 24 | 0 | [0, 0] |
| 0 | 1 | logger12_balanced | 24 | 0 | [0, 0] |
| 0 | 1 | logger2_heavy | 24 | 0 | [0, 0] |
| 0 | 5 | logger1_heavy | 24 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 24 | 0 | [0, 0] |
| 0 | 5 | logger2_heavy | 24 | 0 | [0, 0] |
| 0.3 | 1 | logger1_heavy | 24 | 2.7318e-06 | [0, 6.83267e-06] |
| 0.3 | 1 | logger12_balanced | 24 | 2.7318e-06 | [0, 6.34029e-06] |
| 0.3 | 1 | logger2_heavy | 24 | 2.7318e-06 | [0, 5.52997e-06] |
| 0.3 | 5 | logger1_heavy | 24 | 0 | [0, 0] |
| 0.3 | 5 | logger12_balanced | 24 | 0 | [0, 0] |
| 0.3 | 5 | logger2_heavy | 24 | 0 | [0, 0] |

## Worst-case tie regret

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger1_heavy | 24 | 0 | [0, 0] |
| 0 | 1 | logger12_balanced | 24 | 0 | [0, 0] |
| 0 | 1 | logger2_heavy | 24 | 0 | [0, 0] |
| 0 | 5 | logger1_heavy | 24 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 24 | 0 | [0, 0] |
| 0 | 5 | logger2_heavy | 24 | 0 | [0, 0] |
| 0.3 | 1 | logger1_heavy | 24 | 2.7318e-06 | [0, 6.89903e-06] |
| 0.3 | 1 | logger12_balanced | 24 | 2.7318e-06 | [0, 6.89271e-06] |
| 0.3 | 1 | logger2_heavy | 24 | 2.7318e-06 | [0, 6.83267e-06] |
| 0.3 | 5 | logger1_heavy | 24 | 0 | [0, 0] |
| 0.3 | 5 | logger12_balanced | 24 | 0 | [0, 0] |
| 0.3 | 5 | logger2_heavy | 24 | 0 | [0, 0] |

## Survival, termination, and future clipping audits

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 24 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 24 | 0 | [0, 0] |
| 0.3 | 1 | logger12_balanced | 24 | 0 | [0, 0] |
| 0.3 | 5 | logger12_balanced | 24 | 0 | [0, 0] |

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 24 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 24 | 0 | [0, 0] |
| 0.3 | 1 | logger12_balanced | 24 | 0 | [0, 0] |
| 0.3 | 5 | logger12_balanced | 24 | 0 | [0, 0] |

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 24 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 24 | 0 | [0, 0] |
| 0.3 | 1 | logger12_balanced | 24 | 0 | [0, 0] |
| 0.3 | 5 | logger12_balanced | 24 | 0.0932075 | [0.0414556, 0.149289] |

## Horizon amplification

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 24 | undefined | undefined |
| 0 | 5 | logger12_balanced | 24 | undefined | undefined |
| 0.3 | 1 | logger12_balanced | 24 | 1 | [1, 1] |
| 0.3 | 5 | logger12_balanced | 24 | 5.15384 | [3.51176, 7.19565] |

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 24 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 24 | 0 | [0, 0] |
| 0.3 | 1 | logger12_balanced | 24 | 0 | [0, 0] |
| 0.3 | 5 | logger12_balanced | 24 | -2.7318e-06 | [-7.01912e-06, 0] |

## Monte Carlo integration audit

| kappa | H | R | mean branch SE | R/2-to-R error change | H5 MC-exact |
|---:|---:|---:|---:|---:|---:|

## Negative controls and subsets

The kappa=0 initial-U equality, independent-latent equality to do, and base-action equality to do
all passed at numerical tolerance. The **initial-step strict-unclipped subset** contains
11 selected anchors, including
11 in the primary common-horizon population.
The any-initial-clipping comparison is descriptive and future clipping is reported without filtering.

## Supported claims

The audited tables support statements about measured initial-selection consequences for the named
fixed-policy finite-horizon intervention value, within the selected anchors and fixed continuation.

## Unsupported claims

These results do not support unrestricted control claims, full historical-logger trajectory-return
claims, cross-policy-seed generalization, or a pure causal interpretation of clipping-subset
differences.

## Interpretation limits

The intervals describe anchor-level variation for one fixed 500k behavior-policy checkpoint; they
are not cross-policy-seed significance statements. Initial clipping subsets are descriptive, not a
causal clipping comparison. Future clipping remains part of the closed-loop Hopper outcome and is
never used to filter trajectories. No scientific-success threshold was applied. The numerical
tables support only the measured fixed-policy finite-horizon consequences under this continuation,
not unrestricted control, a full historical-logger trajectory return, or generalization across policy
seeds.
