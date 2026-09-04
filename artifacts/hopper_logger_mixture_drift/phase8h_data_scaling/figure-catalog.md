# Figure catalog

## do_mae_vs_data.png

Purpose: Whether numerical do-Bellman error decreases with source data.

Data: confounded model seeds 0–2; error bars are seed SD where shown.

Interpretation checklist: Compare slopes and seed-SD bars; a decline alone does not prove better decisions.

## mean_regret_vs_data.png

Purpose: Whether average decision loss decreases.

Data: confounded model seeds 0–2; error bars are seed SD where shown.

Interpretation checklist: Check Action-min against pooled and Source 3 across the full grid.

## median_regret_vs_data.png

Purpose: Whether typical decision loss decreases.

Data: confounded model seeds 0–2; error bars are seed SD where shown.

Interpretation checklist: Contrast the median pattern with mean and tail behavior.

## tail_regret_vs_data.png

Purpose: Whether P90 and CVaR90 tail risk improves.

Data: confounded model seeds 0–2; error bars are seed SD where shown.

Interpretation checklist: Tail improvement is central to the low-regret hypothesis.

## underestimation_vs_data.png

Purpose: Whether finite-sample underestimation shrinks.

Data: confounded model seeds 0–2; error bars are seed SD where shown.

Interpretation checklist: A decline supports a variance mechanism only if regret also improves.

## actionmin_vs_source3_gap.png

Purpose: Whether Action-min crosses the fixed Source-3 baseline.

Data: confounded model seeds 0–2; error bars are seed SD where shown.

Interpretation checklist: Negative values favor Action-min.

## actionmin_gain_over_pooled.png

Purpose: Whether the multi-source advantage grows with data.

Data: confounded model seeds 0–2; error bars are seed SD where shown.

Interpretation checklist: Positive values favor Action-min; a flat gap indicates generic scaling.

## data_vs_compute_control.png

Purpose: Separate additional data from additional updates.

Data: confounded model seeds 0–2; error bars are seed SD where shown.

Interpretation checklist: Compare n128 with n32 at 4000 updates for seed 0.

## actionmin_per_seed_scaling.png

Purpose: Show the Action-min trajectory for every model seed.

Data: confounded model seeds 0–2; error bars are seed SD where shown.

Interpretation checklist: Check direction consistency rather than treating anchors as independent replicates.
