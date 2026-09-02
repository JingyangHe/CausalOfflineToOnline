# Phase 8C-FD — Oracle-Scaffolded Mechanism Failure Decomposition

`ORACLE_INFORMATION_USED_FOR_DIAGNOSIS_ONLY = True`  
`ORACLE_VARIANTS_ARE_NOT_DEPLOYABLE = True`

## Existing Phase 8C reproduction

The read-only audit reproduced 2048 anchors, 309 held-out anchors, 5 seeds, 7 frozen doses, 35/35 collapsed mechanism models, and pooled-minus-mechanism do-MAE AUC improvement 0.000500155822269.

## Variant definitions

V0 reuses the formal random-initialized model. V1 is an explicit capacity-matched collapsed likelihood. V2 fixes the manifest prior/behavior and uses public gradient training. V3 uses the same initialization and exact generalized EM. V4 freezes an Oracle-U-supervised reward decoder and learns behavior from public observations. V5 is the untrained Oracle-compatible plugin. V6 starts exactly at V5 and then uses only the public observational objective. V7 reuses the isolated Oracle U-aware ceiling.

## Primary descriptive table

The unit summarized here is the model seed; anchors are repeated test cases and are not treated as independent replicates.

| Variant | Obs val NLL | Do MAE | Rank error | Regret | Mode separation | Behavior error | Collapse rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_random_init | -0.96159593 | 0.018613284 | 0.62977346 | 0.0038984942 | 0.007883056 | 0.1933623 | 1 |
| collapsed_constrained_reference | -0.88833298 | 0.020842822 | 0.62098937 | 0.0038417011 | 0 | 0.153158 | 1 |
| true_behavior_fixed | -0.96427936 | 0.016885142 | 0.55275081 | 0.0031472922 | 0.039637541 | 6.6231827e-09 | 0 |
| true_behavior_fixed_em | -0.94568829 | 0.016748344 | 0.6 | 0.0036862548 | 0.029505344 | 6.6231827e-09 | 0 |
| oracle_reward_fixed_learn_behavior | -1.0934007 | 0.012181075 | 0.25048544 | 0.00094350603 | 0.054884894 | 0.056821898 | 0 |
| oracle_compatible_plugin | -1.0924612 | 0.012181075 | 0.25048544 | 0.00094350603 | 0.054884894 | 6.6231827e-09 | 0 |
| oracle_initialized_joint | -1.362125 | 0.011229091 | 0.31197411 | 0.0013091915 | 0.049588603 | 0.0043378087 | 0 |
| oracle_u_aware_ceiling | -1.1380771 | 0.011162692 | 0.23014332 | 0.00078580013 | 0.054828615 | NA | 0 |

## Failure-cause reading matrix

The artifact reports the facts required to assess behavior learning, optimization/initialization, observational-objective underdetermination, reward approximation, and model-class mismatch. It deliberately does not convert these facts into a unique cause with an arbitrary threshold. V2 versus V3 isolates optimizer form; V4 isolates behavior learning; V5 tests representational existence; V6 records whether observational training preserves or destroys an Oracle-compatible point; V7 bounds reward approximation.

| Candidate explanation | Direct evidence to inspect |
|---|---|
| Behavior-learning bottleneck | V2/V3 mode separation and do metrics; V4 aligned behavior-table error |
| Optimization/initialization | V3−V2 and V6−V0 paired curves |
| Objective underdetermination | V5/V6 versus V1 exact NLL gaps alongside causal gaps |
| Reward approximation | V5 versus V7 do error |
| Model-class mismatch | V5 joint NLL and do error; alpha-profile endpoints |

## Collapse trajectories

V6 mean reward-mode separation changes from 0.054904653 at update 0 to 0.056571975 at the final fixed checkpoint. The synchronized NLL/do/rank/regret trajectory is stored in `collapse_trajectories.csv`; no post-hoc metric selected a checkpoint.

## Objective profiles

For the reward-separation profile, mean primary validation NLL is -1.1231437 at alpha=0 and -1.1000832 at alpha=1. For the behavior-separation profile it is -1.3705075 and -1.440135, respectively. Alpha was never selected on test outcomes.

## Direct diagnostic answers (descriptive, not thresholded)

1. Model-class existence: V5 do MAE is 0.012181075, versus V1 0.020842822 and V7 0.011162692.
2. Reward learning with true behavior: V2 mode separation is 0.039637541; its do MAE is 0.016885142.
3. Behavior learning with fixed Oracle reward: V4 aligned behavior MAE is 0.056821898.
4. EM versus gradient: primary V3−V2 do-MAE difference is -0.00013679786.
5. Oracle initialization stability: the V6 separation trajectory is reported above; its best checkpoint is selected only by validation NLL.
6. NLL during V6 training: inspect the exact update-wise `validation_observational_nll` values in `collapse_trajectories.csv` together with the separation values.
7. Noncollapsed versus collapsed likelihood: mean V5−V1 validation-NLL gap is -0.20412821; mean V6-best−V1 gap is -0.47379206.
8. Selection conflict: observational-best/do-best agreement is 0.371429; the corresponding ranking/regret agreements are stored exactly in `objective_comparison.csv`.
9. Cause attribution: the code does not force a unique label. The five competing explanations must be judged jointly from the effect sizes above, seed variation, and profiles.

## Supported and unsupported conclusions

Only post-hoc comparisons supported by the saved seed-level metrics and trajectories are admissible. Oracle scaffold variants are diagnostic and cannot be presented as deployable methods. No test U or do outcome selected a checkpoint, seed, dose, architecture, or stopping point. Similar NLL values do not establish causal equivalence, and a noncollapsed V6 state does not establish identification of the true U without the aligned recovery diagnostics.
