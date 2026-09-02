# Figure catalog

## Figure 1 — Primary causal and decision curves

- File: `figures/figure-01-primary-causal-decision-curves.pdf`
- Purpose: test whether mechanism separation improves do-reward accuracy, action ranking, and regret across the frozen dose grid.
- Data: primary κ=0.3/confounded/logger12_balanced cells; n=5 model seeds; ribbons are seed-level t 95% CIs.
- Notice: whether improvement is joint across all three endpoints and consistent across λ, rather than isolated to one dose.
- Implication: only a joint improvement supports advancement of the mechanism model.
- Caveat: λ points are evenly spaced visually although the numeric grid is nonuniform; labels show exact values.

## Figure 2 — Mechanism advantage heatmap

- File: `figures/figure-02-mechanism-advantage-heatmap.pdf`
- Purpose: localize pooled-minus-mechanism do-MAE differences across κ, condition, and λ.
- Notice: positive cells favor mechanism separation; similar advantages across conditions suggest generic capacity/optimization rather than confounding-specific recovery.
- Implication: direct U→R confounding can exist at both κ levels because logger action remains associated with U, so κ alone is not a causal on/off switch.
- Caveat: cells show seed means without inferential markers; consult the statistical appendix.

## Figure 3 — Observational fit versus causal error

- File: `figures/figure-03-observational-fit-vs-causal-error.pdf`
- Purpose: assess whether observational validation likelihood selects causally accurate models.
- Notice: points with strong observational fit but high do-MAE demonstrate objective mismatch.
- Implication: observational likelihood alone is insufficient for causal model selection when rankings disagree.
- Caveat: pooled points across λ share anchors and are not independent.

## Figure 4 — Source-composition stability

- File: `figures/figure-04-source-composition-stability.pdf`
- Purpose: compare prediction drift when logger mixture changes while the controlled support is fixed.
- Notice: lower cross-mixture MAE means greater invariance; bars are not causal accuracy.
- Implication: mechanism separation should be preferred only if stability does not come at the cost of do accuracy.
- Caveat: summaries average seven λ doses and three mixture pairs.

## Figure 5 — Latent identification diagnostics

- File: `figures/figure-05-latent-identification-diagnostics.pdf`
- Purpose: test whether the latent mechanism remains active and separates reward modes.
- Notice: high collapse with weak mode separation rules out a strong latent-recovery claim.
- Implication: predictive gains under collapse must be attributed cautiously to architecture/regularization rather than recovered U.
- Caveat: latent labels are exchangeable and these diagnostics do not prove semantic alignment with true U.

## Figure 6 — Confounded-minus-independent performance gap

- File: `figures/figure-06-confounded-minus-independent-gap.pdf`
- Purpose: isolate the observational confounding penalty at κ=0.3 while holding λ, anchors, mixtures, and seeds fixed.
- Notice: large positive gaps at nonzero λ show that direct U→R confounding changes learned action preferences; mechanism and pooled curves remain close while Oracle is much less affected.
- Implication: the DGP creates a real identification challenge, but the current mechanism architecture does not solve it.
- Caveat: ribbons are seed-level t 95% CIs with n=5; points across λ reuse anchors and seeds.
