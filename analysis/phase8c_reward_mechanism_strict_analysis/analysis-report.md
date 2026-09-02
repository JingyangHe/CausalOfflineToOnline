# Phase 8C-RM strict scientific analysis

## Locked question and design

The primary question is whether `mechanism_separated` improves held-out interventional reward prediction and three-action decisions relative to observational baselines. The primary setting was fixed to κ=0.3, `confounded`, and `logger12_balanced`. The analysis covers 2,048 fixed anchors, 309 held-out test anchors, five model seeds, seven manually frozen λ doses, nine methods, and 3,780 trained models. Lower is better for every primary endpoint.

The inferential unit is the model seed. Primary comparisons use the normalized area under the seven-point λ curve per seed, paired by seed. Exact two-sided sign-flip tests enumerate all 2^5 sign assignments; Holm correction covers 24 method-by-metric contrasts. Anchor observations are repeated evaluation cases and are not counted as independent model replicates.

## Main findings

1. **Only a small reward-MAE gain over pooled fitting.** Relative to pooled MLP, the mechanism model's mean AUC improvement in do-reward MAE was 0.000500156 (95% seed-level CI [0.000296817, 0.000703495]), or 1.98% of pooled error. The source-balanced pooled model is numerically identical in the primary balanced mixture, so its repeated contrast is a design identity rather than independent confirmation.

2. **No joint decision improvement.** The pooled-minus-mechanism AUC improvement was only 0.00449191 for top-action disagreement and 4.30387e-05 for mean regret; both CIs crossed zero. At λ=0.1, mechanism disagreement reached 90.3% and regret 0.00626557, versus 28.5% and 0.000878422 for Oracle. Source-shuffle differed by 0.00149838 in ranking AUC, and removing the behavior mechanism differed by 2.54826e-05 in regret AUC. These negative controls are effectively indistinguishable from the intended mechanism.

3. **Large Oracle and simpler-baseline gaps.** The mechanism model's do-MAE AUC was 2.36× the U-aware Oracle error, ranking-disagreement AUC was 2.83× Oracle, and regret AUC was 5.84× Oracle. Per-source models also had lower do-MAE than mechanism by 0.00283641 AUC.

4. **The controlled confounding signal is strong but not removed.** At κ=0.3, the confounded-minus-independent ranking-error AUC was 0.474094 for pooled MLP and 0.472951 for mechanism separation. The similar gaps show that the mechanism model did not isolate the direct U→R pathway. κ=0 also permits direct reward confounding because logger action remains associated with U; therefore κ difference-in-differences is secondary, not the defining test.

5. **Observational fit does not select causal accuracy reliably.** Among deployable methods, the observationally best and do-MAE-best methods agreed in only 9.5% of 420 matched scenario-seed cells. Mechanism separation was observationally best in 0.0% and do-MAE-best in 1.0%.

6. **Composition stability is not mechanism recovery.** The mechanism model's cross-mixture prediction MAE was 0.0117783 [0.0109709, 0.0125858]. `source_balanced_pooled_mlp` is exactly invariant (0) by construction because it always trains on the same balanced rows, so zero drift is a control identity rather than evidence of identification.

7. **Latent recovery failed.** In the primary setting, the mechanism model collapsed its binary latent in 100.0% of 35 seed-dose cells; its mean reward-mode separation was 0.00836733. Together with the source-shuffle/no-behavior results, this rules out a strong claim that the learned latent recovered true U.

8. **Statistical resolution is limited.** With only five seeds, an exact two-sided sign-flip test cannot attain p<0.05 (minimum possible p=0.0625); all 24 Holm-adjusted primary p-values equal 1. Conclusions therefore rely on effect magnitudes, confidence intervals, dose consistency, and negative controls rather than significance declarations.

## Exact primary contrast table

| Metric | Comparator − mechanism AUC | 95% CI | Hedges g_z | exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| mae / pooled_mlp | 0.000500156 | [0.000296817, 0.000703495] | 2.44331 | 0.0625 | 1 |
| mae / source_balanced_pooled_mlp | 0.000500156 | [0.000296817, 0.000703495] | 2.44331 | 0.0625 | 1 |
| mae / oracle_u_aware | -0.0142958 | [-0.0154157, -0.0131759] | -12.6802 | 0.0625 | 1 |
| mae / source_shuffle | -6.68729e-05 | [-0.000565872, 0.000432126] | -0.13312 | 0.75 | 1 |
| mae / no_behavior | -4.3076e-05 | [-0.000518083, 0.000431931] | -0.0900801 | 0.875 | 1 |
| top_set_disagreement / pooled_mlp | 0.00449191 | [-0.0110858, 0.0200696] | 0.286432 | 0.5 | 1 |
| top_set_disagreement / source_balanced_pooled_mlp | 0.00449191 | [-0.0110858, 0.0200696] | 0.286432 | 0.5 | 1 |
| top_set_disagreement / oracle_u_aware | -0.513068 | [-0.52625, -0.499885] | -38.6608 | 0.0625 | 1 |
| top_set_disagreement / source_shuffle | 0.00149838 | [-0.00177204, 0.00476881] | 0.455106 | 0.3125 | 1 |
| top_set_disagreement / no_behavior | 0.00189968 | [-0.0081577, 0.0119571] | 0.187624 | 0.625 | 1 |
| mean_regret / pooled_mlp | 4.30387e-05 | [-9.82646e-05, 0.000184342] | 0.302553 | 0.5 | 1 |
| mean_regret / source_balanced_pooled_mlp | 4.30387e-05 | [-9.82646e-05, 0.000184342] | 0.302553 | 0.5 | 1 |
| mean_regret / oracle_u_aware | -0.00422329 | [-0.00428534, -0.00416125] | -67.6109 | 0.0625 | 1 |
| mean_regret / source_shuffle | 1.07786e-05 | [-3.27351e-05, 5.42924e-05] | 0.246054 | 0.75 | 1 |
| mean_regret / no_behavior | 2.54826e-05 | [-7.22505e-05, 0.000123216] | 0.258998 | 0.5625 | 1 |

## Interpretation boundary

This is a controlled one-step, three-action reward-mechanism diagnostic. It does not establish long-horizon policy-value improvement. The anchors are fixed rather than sampled as independent environments, and the five-seed uncertainty intervals are wide. The source-dependent and per-source models are observational baselines without causal guarantees. The Oracle model uses hidden U and is only a ceiling.

## Decision

**Do not promote the current Phase 8C mechanism model as successful causal identification.** The defensible result is negative: it yields a small generic do-MAE improvement over pooled fitting but does not retain that advantage in ranking or regret, does not beat simpler deployable baselines, is indistinguishable from structural negative controls, and collapses the intended latent. The experiment nevertheless succeeds scientifically by demonstrating a strong observational-to-interventional failure under direct U→R confounding and a large recoverable Oracle gap.
