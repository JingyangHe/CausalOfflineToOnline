# Phase 8J-BF-Q Bellman Backup Forensics

This is a read-only, single-seed mechanism diagnosis. It does not retrain a potential, select a checkpoint, tune a hyperparameter, run SAC, establish a global Lipschitz constant, or prove a causal mechanism.

Checkpoint protocol: `coarse_existing_milestones`; temporal resolution is 50 epochs. This supports mechanism classification but cannot localize onset within a checkpoint interval.

## Mechanism labels

- `CORE_BOOTSTRAP_OR_VALUE_SCALE_INSTABILITY`
- `DENSITY_WEIGHTING_CONTRIBUTES_TO_INSTABILITY`
- `MODEL_EXTRAPOLATION_AMPLIFIED_BY_MAX`

Labels use only the direction and ordering of continuous diagnostic growth curves; no magnitude threshold declares a root cause. Multiple labels may coexist.

## Decision tree evidence

### matched_moving_target

Final-minus-initial standard-deviation growth:

- D0_observed_only: 728.54885
- D1_model_next_no_max: 761.83286
- D2_model_next_candidate_mean: 750.38213
- D3_original_candidate_max: 772.2987

Contribution standard-deviation growth:

- reward_contribution: 0
- observed_continuation_contribution: 661.76167
- road_continuation_contribution: 737.86284
- largest continuous growth: road_continuation_contribution

Selected observed-density weight: 0.046180196 -> 0.23784286.
Latest selected/all positive Bellman error means: 248.52014 / 208.22631.
Latest backup/error Spearman correlation: 0.13161715.

### frozen_outer_target

Final-minus-initial standard-deviation growth:

- D0_observed_only: 209.7285
- D1_model_next_no_max: 213.57168
- D2_model_next_candidate_mean: 213.57744
- D3_original_candidate_max: 212.82838

Contribution standard-deviation growth:

- reward_contribution: 0
- observed_continuation_contribution: 223.04149
- road_continuation_contribution: 238.27941
- largest continuous growth: road_continuation_contribution

Selected observed-density weight: 0.046180196 -> 0.21553995.
Latest selected/all positive Bellman error means: 48.887954 / 38.988364.
Latest backup/error Spearman correlation: 0.14192763.

## Interpretation constraints

D0 uses the real logged reward and real logged next state. D1 uses the fixed logged action with the model-next AAMAS backup. D2 averages the same 28 exact candidate backups. D3 takes their maximum. The simulator audit exactly balances binary U and is strictly post-hoc.

The potential is a single unclamped critic, so twin-critic disagreement is not applicable and is recorded explicitly rather than invented. Distance metrics remain continuous descriptions; no OOD threshold is used.

## Next step

Use the reported branch, counterfactual-backup, simulator-error, spatial-value, and amplification curves to choose the next diagnostic. Do not change optimization or pass any checkpoint to SAC within this phase.
