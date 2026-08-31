# Phase 8B-RS-O — Oracle Reward-Confounding Audit

No neural network or learned prediction was used. The statistical unit is
`anchor_id` (n=512); uncertainty intervals are
anchor-bootstrap percentile intervals with 2000 repetitions.

## Primary table

| lambda | kappa | balanced plus bias | balanced minus bias | heavy drift | base bias | independent bias | do shift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 4.1915256e-16 | 4.33680869e-17 | 1.28586378e-16 | 1.97324795e-17 | 8.8817842e-16 | 0 |
| 0.05 | 0 | 0.03 | -0.03 | 0.0155555556 | 1.97324795e-17 | 8.8817842e-16 | 0 |
| 0.1 | 0 | 0.06 | -0.06 | 0.0311111111 | 1.97324795e-17 | 8.8817842e-16 | 0 |
| 0.2 | 0 | 0.12 | -0.12 | 0.0622222222 | 1.97324795e-17 | 8.8817842e-16 | 0 |
| 0 | 0.3 | -0.00256926451 | 0.00195272884 | -0.00133221123 | 1.08420217e-17 | 1.77635684e-15 | 0 |
| 0.05 | 0.3 | 0.0274307355 | -0.0280472712 | 0.0142233443 | 1.08420217e-17 | 1.77635684e-15 | 0 |
| 0.1 | 0.3 | 0.0574307355 | -0.0580472712 | 0.0297788999 | 1.08420217e-17 | 1.77635684e-15 | 0 |
| 0.2 | 0.3 | 0.117430735 | -0.118047271 | 0.060890011 | 1.08420217e-17 | 1.77635684e-15 | 0 |

`heavy drift` is the plus-action logger1-heavy minus logger2-heavy contrast.
`independent bias` is the maximum absolute observational-do bias over all anchors,
actions, and primary mixtures. `do shift` is the maximum absolute lambda-induced
change in the do mean.

## Direct answers

- Q1: Yes. The confounded observational-do reward bias grows exactly linearly with lambda.
- Q2: Yes. P(S,A) is unchanged; the added bias is exactly lambda E[U_env|S,A].
- Q3: Yes. The balanced source mixture retains slopes -0.6 and +0.6.
- Q4: No. The direct term cancels under the symmetric do(U_env) average.
- Q5: Yes. Independent latents remove the direct and total observational-do bias.
- Q6: Yes. The base action has zero direct bias for every anchor and lambda.
- Q7: Yes. Total bias equals physical bias plus direct U-to-reward bias within tolerance.

## Numerical audit

- Maximum reward-definition residual: 4.163e-16
- Maximum population-table recomputation residual: 1.776e-15
- Maximum bias-decomposition residual: 5.274e-16
- Maximum lambda-induced do shift: 0.000e+00
- Maximum P(S,A) mass change: 0.000e+00

## Theory versus empirical lambda slopes

| kappa | family | action | empirical | theoretical | abs. error | intercept | R² | max anchor residual |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.0 | balanced_bias | minus | -0.6 | -0.6 | 6.661e-16 | 1.27192026e-16 | 1 | 3.816e-16 |
| 0.0 | balanced_bias | base | -1.04143493e-31 | 0 | 1.041e-31 | 1.97324795e-17 | 1 | 0.000e+00 |
| 0.0 | balanced_bias | plus | 0.6 | 0.6 | 7.772e-16 | 3.36150355e-16 | 1 | 3.816e-16 |
| 0.0 | heavy_drift | minus | -0.311111111 | -0.311111111 | 6.661e-16 | 6.35960131e-17 | 1 | 4.163e-16 |
| 0.0 | heavy_drift | base | 0 | 0 | 0.000e+00 | 0 | 1 | 0.000e+00 |
| 0.0 | heavy_drift | plus | 0.311111111 | 0.311111111 | 6.106e-16 | 1.95330612e-16 | 1 | 4.996e-16 |
| 0.3 | balanced_bias | minus | -0.6 | -0.6 | 7.772e-16 | 0.00195272884 | 1 | 3.886e-16 |
| 0.3 | balanced_bias | base | -4.86675404e-32 | 0 | 4.867e-32 | 1.08420217e-17 | 1 | 0.000e+00 |
| 0.3 | balanced_bias | plus | 0.6 | 0.6 | 5.551e-16 | -0.00256926451 | 1 | 3.886e-16 |
| 0.3 | heavy_drift | minus | -0.311111111 | -0.311111111 | 6.661e-16 | 0.00101252606 | 1 | 4.163e-16 |
| 0.3 | heavy_drift | base | 0 | 0 | 0.000e+00 | 0 | 1 | 0.000e+00 |
| 0.3 | heavy_drift | plus | 0.311111111 | 0.311111111 | 6.661e-16 | -0.00133221123 | 1 | 4.996e-16 |

## Physical/direct/total decomposition

The following is the balanced confounded mixture at kappa=0.3 and lambda=0.20.

| action | physical | direct | total |
|---|---:|---:|---:|
| minus | 0.00195272884 | -0.12 | -0.118047271 |
| base | 1.08420217e-17 | 0 | 1.08420217e-17 |
| plus | -0.00256926451 | 0.12 | 0.117430735 |

## Population signal versus previous neural error

| kappa | lambda | balanced/error | heavy/error | previous error |
|---:|---:|---:|---:|---:|
| 0.0 | 0.00 | 0 | 0 | 0.0260947 |
| 0.0 | 0.05 | 1.14966 | 0.59612 | 0.0260947 |
| 0.0 | 0.10 | 2.29932 | 1.19224 | 0.0260947 |
| 0.0 | 0.20 | 4.59864 | 2.38448 | 0.0260947 |
| 0.3 | 0.00 | 0 | 0 | 0.0245549 |
| 0.3 | 0.05 | 1.22175 | 0.6335 | 0.0245549 |
| 0.3 | 0.10 | 2.4435 | 1.267 | 0.0245549 |
| 0.3 | 0.20 | 4.887 | 2.534 | 0.0245549 |

At lambda=0.05 the balanced signal already reaches the previous neural-error
scale; by lambda=0.10 and 0.20 both the balanced signal and, eventually, the
heavy contrast are at or above that scale. This is a scale comparison only.

## Secondary ranking and regret

| kappa | lambda | mixture | disagreement | true regret |
|---:|---:|---|---:|---:|
| 0.0 | 0.20 | logger12_balanced | 0.9375 | 0.0059545 |
| 0.0 | 0.20 | logger1_heavy | 0.9375 | 0.0059545 |
| 0.0 | 0.20 | logger2_heavy | 0.9375 | 0.0059545 |
| 0.3 | 0.20 | logger12_balanced | 0.933594 | 0.00547334 |
| 0.3 | 0.20 | logger1_heavy | 0.933594 | 0.00547334 |
| 0.3 | 0.20 | logger2_heavy | 0.933594 | 0.00547334 |

These decision metrics are secondary: absence of a ranking flip would not
invalidate the exact reward-confounding result.

## Supported conclusions

- Direct reward confounding is nonzero after source balancing and grows with the locked theoretical slopes.
- The direct channel changes observational reward but neither P(S,A) nor the do-action mean.
- Physical and direct reward bias are exactly additive at the audited numerical tolerance.
- Independent latents, base action, lambda=0, and kappa=0 behave as the specified controls.

## Evidence boundary

This audit identifies the exact population signal and its decomposition. It does
not establish that a learned reward model can recover the signal, that action
ranking must flip, or that any downstream policy improves.
