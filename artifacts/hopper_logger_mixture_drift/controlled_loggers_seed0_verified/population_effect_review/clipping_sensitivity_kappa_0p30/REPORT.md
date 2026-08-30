# Phase 8A-C — Applied-Action Clipping Sensitivity Audit

This is a read-only, descriptive audit of the verified Phase 8A artifact at kappa=0.3.

## Scope and validity

All clipping labels were reconstructed from preclip actions and checked against saved applied
actions. The canonical execution clipping fraction is 0.181722.
There are 831 strict-unclipped anchors out of
2048 analyzed anchors.

The tables report facts only. No effect-retention threshold was used and no paper-success verdict
was selected. Clean and clipped anchors may occupy different state regions, so their descriptive
difference is not a pure causal effect of clipping. Results use one behavior checkpoint seed and do
not establish cross-policy-seed population significance.

## Reading the bundle

`decision_metrics.csv` compares all anchors, strict-unclipped anchors, and anchors with any
clipping. `action_specific_metrics.csv` compares all, pair-unclipped, and pair-clipped samples.
`aggregate_tables.csv` is their common source. Exact prevalence and weighted clipping probabilities
are in `summary.json`; canonical and anchor masks are in the NPZ tables.

## Mechanism boundary

The evidence can establish whether drift and ranking flips remain observable on the strict clean
subset. It cannot establish that clipping has no influence, that clipping is the unique cause, or
that full-versus-clean differences are causal. Manual scientific review is required.
