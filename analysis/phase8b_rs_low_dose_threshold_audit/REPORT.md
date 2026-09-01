# Phase 8B-RS-T — Exact Low-Dose Decision-Threshold Audit

This read-only audit analyzed 2,048 Phase 8A-NC anchors and recovered all 78 held-out test anchors. It used the exact affine identity $R_{obs}(\lambda)=b+c\lambda$; no neural model was trained or evaluated.

## Validity

All input hashes were unchanged. Oracle reconstruction and the existing $\lambda\in\{0,0.05,0.10,0.20\}$ ranking/regret results agreed within the registered `atol=rtol=1e-07` tolerance. Independent-latent slopes and the base-action slope were zero; balanced-confounded slopes were exactly `[-0.6, 0, +0.6]` within tolerance. Infinite thresholds are right-censored cases with no boundary for $\lambda\geq0$.

For `logger12_balanced/confounded`, the finite positive-regret threshold median was 0.00506324 at `kappa=0` and 0.00943415 at `kappa=0.3`. The corresponding censored fractions were 8.30% and 8.45%. Thus the old first positive dose `lambda=0.05` lies well above the typical exact decision boundary and is suitable as a strong positive control, not as a fine transition probe.

## Interpretation boundary

The tables describe exact one-step reward-ranking sensitivity. They do not establish long-horizon policy value or neural-model performance. Clipping subsets are descriptive sensitivity groups, not randomized causal strata. Crossing-point ties use the project tolerance; immediate one-sided decisions use `numpy.nextafter` so no arbitrary epsilon is introduced.

## Dose-grid rule

The proposed quantiles use only train and validation anchors; the held-out test split is reported only for audit comparison. `lambda=0.05` is retained as a strong positive control.

Final neural dose grid must be manually frozen before mechanism-model training and may use only train/validation threshold information.
