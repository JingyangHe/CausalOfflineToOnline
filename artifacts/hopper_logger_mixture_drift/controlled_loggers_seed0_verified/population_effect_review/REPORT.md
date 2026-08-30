# Phase 8A-R — Hopper Logger-Mixture Population Effect Review

## Input artifact integrity

The verified Phase 8A input contains 2048 anchors and kappa values 0.0, 0.1, 0.2, and 0.3.
All 84 upstream hard invariants were true. Input hashes were recorded before analysis and matched
after analysis. Population effects were recomputed from public NPZ rows, aligned sample weights,
hidden audit labels used only for action/U mechanism auditing, and the raw two-point do oracle.

## Comparison definitions

`PRIMARY_FIXED_STATE_ACTION_MIXTURES` are logger1_heavy, logger12_midpoint, and logger2_heavy.
Their exact `(anchor_id, commanded_action bytes)` probability masses match, with conditional action
mass `(minus, base, plus) = (0.45, 0.10, 0.45)` at every anchor.

`SECONDARY_LOGGER_AND_ACTION_MIXTURE_SHIFT` contains balanced and logger3_heavy. These change the
base-action mass and cannot support the fixed-P(S,A) primary interpretation.

## Main anchor-level summary

| kappa | reward U effect | delta U effect | conf reward drift | ind reward drift | conf delta drift | ind delta drift | heavy ranking diff | strict flip | drift/action gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 2.0744402e-17 | 0 | 2.3491047e-18 | 4.8066296e-18 | 0 | 1.1096918e-16 | 0 | 0 | 3.3993159e-14 |
| 0.10 | 0.0039153429 | 0.26298005 | 0.0020190968 | 2.511735e-17 | 0.13599706 | 9.80449e-17 | 0.0087890625 | 0.0087890625 | 0.41472415 |
| 0.20 | 0.0077826082 | 0.52196706 | 0.0040116878 | 1.8323017e-17 | 0.26915612 | 9.6168386e-17 | 0.01171875 | 0.011230469 | 0.85362235 |
| 0.30 | 0.011599254 | 0.77617177 | 0.0059901097 | 2.1503343e-17 | 0.40003361 | 9.1101702e-17 | 0.97167969 | 0.97167969 | 1.3172698 |

All displayed values are descriptive. Full mean, SD, median, P10/P25/P75/P90, maximum, and paired
anchor-bootstrap intervals are in `aggregate_tables.csv`.

## U posterior mechanism audit

Weighted empirical posteriors matched the analytic complementary-logger values: confounded plus
uses `(8/9, 1/2, 1/9)` across logger1-heavy/midpoint/logger2-heavy; confounded minus reverses this;
base and every independent-latent cell equal 1/2.

## Negative controls and mechanism identity

The kappa=0, independent-latent, and base-action controls are recorded in `hard_checks.json`.
The reward and physical-delta heavy contrasts were checked anchor by anchor against the signed
`(7/9) * U-effect` identity. The midpoint was checked against the do response in the complementary
DGP. No scientific-effect threshold was applied.

## do-error, decision scale, and ranking

Mixture/action/condition-specific do-errors and tie-aware top-action-set comparisons are in the CSV.
The drift/action-gap quantity is a ratio of aggregate means; pointwise division by near-zero action
gaps was not performed. Strict flips require disjoint top-action sets under the two heavy mixtures.

| kappa | conf do-error L1 | conf do-error midpoint | conf do-error L2 | ind do-error max(primary) | secondary reward range conf | secondary reward range ind |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.5664914e-16 | 4.7343495e-17 | 1.5668528e-16 | 1.6058841e-16 | 1.1140177e-16 | 1.1378702e-16 |
| 0.10 | 0.0010095484 | 5.0704522e-17 | 0.0010095484 | 1.3953682e-16 | 0.0020190968 | 1.3259793e-16 |
| 0.20 | 0.0020058439 | 2.8550657e-17 | 0.0020058439 | 1.4369293e-16 | 0.0040116878 | 1.2873094e-16 |
| 0.30 | 0.0029950549 | 1.6263033e-17 | 0.0029950549 | 1.3787438e-16 | 0.0059901097 | 1.317667e-16 |

## Secondary logger-and-action mixture shift

The final two columns above summarize the original four-mixture range and are explicitly secondary:
balanced and logger3-heavy change the base-action mass, so these values cannot establish drift under
fixed P(S,A). Per-mixture secondary do-errors remain available in `aggregate_tables.csv`.

## Existing Phase 8A cross-check

Comparable legacy metrics passed: `True`. Old value, recomputed value, and absolute
difference are stored in `summary.json`. Items with different definitions are labeled
`NOT_COMPARABLE_DUE_TO_DIFFERENT_METRIC_DEFINITION` rather than coerced.

## Statistical scope and unsupported conclusions

The unit is `anchor_id` (N=2048). Intervals use 2000
paired cluster-bootstrap repetitions with seed 0. They describe anchor-level uncertainty only.
There is one behavior-policy training seed, so this review does not establish cross-policy-seed
generalization or population significance. Transition rows are not treated as independent samples.
The midpoint result is a mechanism-positive-control property of this complementary construction;
it does not show that general source balancing always removes confounding, nor that it is generally
ineffective. This stage does not train or evaluate a pooled neural world model.

## Hard checks

- `verified_phase8a_root_required`: True
- `all_expected_anchors_present`: True
- `all_four_kappas_present`: True
- `all_84_phase8a_invariants_true`: True
- `both_phase8a_completion_markers_present`: True
- `all_kappa_public_anchor_sets_match`: True
- `weight_arrays_align_with_public_rows`: True
- `weight_arrays_sum_to_one`: True
- `midpoint_is_average_of_heavy_weights`: True
- `primary_mixtures_preserve_exact_state_action_mass`: True
- `exact_action_key_mapping`: True
- `do_oracle_raw_keys_complete_and_unique`: True
- `public_schema_has_no_hidden_leakage`: True
- `all_input_arrays_finite`: True
- `metrics_use_anchor_level_units`: True
- `kappa_0p00:primary_state_action_mass_preserved`: True
- `kappa_0p00:do_raw_summary_agreement`: True
- `kappa_0p00:do_oracle_mixture_and_condition_independent`: True
- `kappa_0p00:u_posterior_matches_analytic_values`: True
- `kappa_0p00:kappa_zero_negative_control`: True
- `kappa_0p00:independent_population_equals_do`: True
- `kappa_0p00:base_action_is_primary_mixture_invariant_and_equals_do`: True
- `kappa_0p00:midpoint_population_equals_do_in_complementary_dgp`: True
- `kappa_0p00:reward_drift_identity`: True
- `kappa_0p00:delta_drift_identity`: True
- `kappa_0p00:public_schema_has_no_hidden_leakage`: True
- `kappa_0p00:all_arrays_finite`: True
- `kappa_0p10:primary_state_action_mass_preserved`: True
- `kappa_0p10:do_raw_summary_agreement`: True
- `kappa_0p10:do_oracle_mixture_and_condition_independent`: True
- `kappa_0p10:u_posterior_matches_analytic_values`: True
- `kappa_0p10:kappa_zero_negative_control`: True
- `kappa_0p10:independent_population_equals_do`: True
- `kappa_0p10:base_action_is_primary_mixture_invariant_and_equals_do`: True
- `kappa_0p10:midpoint_population_equals_do_in_complementary_dgp`: True
- `kappa_0p10:reward_drift_identity`: True
- `kappa_0p10:delta_drift_identity`: True
- `kappa_0p10:public_schema_has_no_hidden_leakage`: True
- `kappa_0p10:all_arrays_finite`: True
- `kappa_0p20:primary_state_action_mass_preserved`: True
- `kappa_0p20:do_raw_summary_agreement`: True
- `kappa_0p20:do_oracle_mixture_and_condition_independent`: True
- `kappa_0p20:u_posterior_matches_analytic_values`: True
- `kappa_0p20:kappa_zero_negative_control`: True
- `kappa_0p20:independent_population_equals_do`: True
- `kappa_0p20:base_action_is_primary_mixture_invariant_and_equals_do`: True
- `kappa_0p20:midpoint_population_equals_do_in_complementary_dgp`: True
- `kappa_0p20:reward_drift_identity`: True
- `kappa_0p20:delta_drift_identity`: True
- `kappa_0p20:public_schema_has_no_hidden_leakage`: True
- `kappa_0p20:all_arrays_finite`: True
- `kappa_0p30:primary_state_action_mass_preserved`: True
- `kappa_0p30:do_raw_summary_agreement`: True
- `kappa_0p30:do_oracle_mixture_and_condition_independent`: True
- `kappa_0p30:u_posterior_matches_analytic_values`: True
- `kappa_0p30:kappa_zero_negative_control`: True
- `kappa_0p30:independent_population_equals_do`: True
- `kappa_0p30:base_action_is_primary_mixture_invariant_and_equals_do`: True
- `kappa_0p30:midpoint_population_equals_do_in_complementary_dgp`: True
- `kappa_0p30:reward_drift_identity`: True
- `kappa_0p30:delta_drift_identity`: True
- `kappa_0p30:public_schema_has_no_hidden_leakage`: True
- `kappa_0p30:all_arrays_finite`: True
- `existing_summary_crosscheck_where_comparable`: True
- `all_recomputed_arrays_finite`: True
- `aggregate_outputs_have_no_nan_inf`: True
- `input_artifact_hashes_unchanged`: True

## Manual decision required

Review the continuous effect sizes, bootstrap intervals, ranking changes, and drift/action-gap scale.
The program intentionally does not emit a supported/not-supported scientific verdict.
