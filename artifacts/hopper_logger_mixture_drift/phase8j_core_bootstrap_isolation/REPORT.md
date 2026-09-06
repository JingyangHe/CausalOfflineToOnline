# 发散第一次出现在：B. 真实 transition bootstrap。

# Phase 8J-BI-Q Core Bootstrap Isolation Diagnostic

本实验是单 seed、从头训练的稳定性定位，不是算法改进、策略价值评估或显著性检验。

| Variant | Late instability | Pred. SD 0→200 | |P99| growth | Gradient growth | Final/best val |
|---|---|---:|---:|---:|---:|
| reward_only | False | 5.2121e-08 → 0.77069 | 8.8468e+07 | 0.0062616 | 1.1277 |
| real_transition_bootstrap | True | 5.2121e-08 → 34.526 | 3.8797e+09 | 6.839 | 2498.3 |
| full_aamas_frozen | True | 5.2121e-08 → 224.49 | 3.1282e+10 | 102.99 | 2338 |

## Mechanism labels

- `CORE_BOOTSTRAP_VALUE_FITTING_INSTABILITY`
- `CORE_BOOTSTRAP_INSTABILITY_AMPLIFIED_BY_AAMAS_BACKUP`

## Interpretation

`reward_only` isolates ordinary fixed-label regression. `real_transition_bootstrap` adds only logged next-state bootstrapping with a target frozen for five inner epochs. `full_aamas_frozen` adds the existing complete pooled-union AAMAS backup. All three arms share initialization, rows, minibatch order, optimizer, normalization, network, and 200-epoch budget.

A late-growth label requires monotone growth in at least three of six independently recorded health curves over epochs 150/160/170/180/190/200. It does not use an abs(V)<1,000,000 threshold.
Case D is reported only when the real-transition arm has at least one monotone late growth signal and the full-backup arm grows faster in prediction SD, |P99|, and gradient norm over epochs 150→200.

## Boundaries

- model seed n=1; anchors and rows are repeated measurements, not independent runs.
- This phase does not run SAC or inspect online return.
- No gradient clipping, clamp, sigmoid, Huber loss, normalization change, LR change, early stopping, or checkpoint selection was introduced.
- If no arm reproduces late monotone growth, the required conclusion is `ROOT_CAUSE_LAYER_NOT_IDENTIFIED`.
