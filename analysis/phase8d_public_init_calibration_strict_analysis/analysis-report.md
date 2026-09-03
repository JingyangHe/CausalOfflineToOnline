# Phase 8D-PRIC Strict Analysis

## Analysis gate and estimand

All 25 formal hard checks passed. The experiment contains 2,048 anchors, 309 held-out
test anchors, five model seeds, seven pre-frozen reward doses, two κ values, and two
latent conditions. The primary estimand is the equal-weight mean over the seven frozen
doses after averaging within each model seed. Normalized trapezoidal AUC is reported as
a sensitivity estimand. The inferential unit is the model seed (n=5); the 20 calibration
replicates are nested calibration randomness, not 20 independent training runs.

This remains an exploratory algorithm-development experiment because aggregate results
on the existing test split were viewed before Phase 8D.

## Main findings

1. **Public residual initialization prevents the observed collapse.** V0 collapsed in
   35/35 primary seed-dose cells, while public initialization, source shuffle, and
   no-staged training each collapsed in 0/35. Across both κ values and both conditions,
   V0 collapsed in 140/140 cells and all three public initializations in 0/140.
2. **Non-collapse is useful but not equivalent to recovering the true mechanism.** Public
   initialization reduced primary Do MAE from 0.0186133 to 0.0135472,
   closing 68.6% of the aggregate V0-to-V6 gap. It closed
   57.1% of the ranking gap and
   62.0% of the regret gap. Its behavior-table MAE
   (0.0826672) remained far above V6
   (0.00433781).
3. **The gain is mainly consistent with outcome clustering, not source recovery.** Source
   shuffle retained nearly the same Do MAE (0.0136391) and decision metrics
   despite behavior-table MAE worsening from 0.0826672 to
   0.318149. Therefore the experiment does not support a claim
   that multi-source behavior identification caused the predictive gain.
4. **Staged optimization is not shown to be necessary.** No-staged training also had zero
   collapse and achieved Do MAE 0.0125865, versus
   0.0135472 for staged NLL selection. Its ranking and regret were also
   slightly lower. With n=5 seeds, paired exact tests have limited resolution, but there
   is no positive evidence that staging is required.
5. **Calibration solves part of candidate selection, not the whole decision problem.**
   Relative to the uniform B=0 ensemble, B=128 reduced Do MAE by
   24.1%, ranking error by
   17.1%, and regret by
   22.8%. Relative to the observational-NLL-best
   public model, however, B=128 improved Do MAE by
   5.6% while ranking error and regret
   were slightly worse (0.452173 vs 0.448451, and
   0.00232432 vs 0.00229353).
6. **The candidate bank contains a near-V6 model, but calibration does not fully isolate it.**
   Post-hoc Oracle-best candidate Do MAE was
   0.0118341, versus V6
   0.0112291. B=128 closed
   44.4% of the
   NLL-best-to-Oracle-best candidate selection gap and left mean gap
   0.000952419.
7. **Benefits are dose dependent and the λ=0 negative control is unfavorable.** At λ=0,
   public initialization had Do MAE 0.0163226
   versus pooled MLP 0.0111047; ranking error and regret were also
   worse. The large aggregate gain is concentrated at stronger doses, especially λ=0.05
   and 0.1. This limits claims of a generally safe initialization.

## Primary method table

Values are seed-level equal-grid means over the seven frozen doses. Lower is better for
NLL and all error metrics. Oracle variants V6/V7 are diagnostic ceilings, not deployable.

| Method | Collapse | Val NLL | Do MAE | Rank error | Regret | Reward separation | Behavior MAE |
|---|---|---|---|---|---|---|---|
| Pooled MLP | — | -1.61488 | 0.019127 | 0.623763 | 0.00389176 | — | — |
| V0 random init | 1 | -0.960455 | 0.0186133 | 0.629773 | 0.00389849 | 0.00788306 | 0.193362 |
| Public residual + NLL | 0 | -1.37131 | 0.0135472 | 0.448451 | 0.00229353 | 0.0636266 | 0.0826672 |
| Source shuffle | 0 | -1.04408 | 0.0136391 | 0.453537 | 0.00234042 | 0.0640597 | 0.318149 |
| No staged training | 0 | -1.33696 | 0.0125865 | 0.442071 | 0.00228926 | 0.0511918 | 0.0857631 |
| V6 oracle init | 0 | -1.3367 | 0.0112291 | 0.311974 | 0.00130919 | 0.0495886 | 0.00433781 |
| V7 U-aware | — | — | 0.0111627 | 0.230143 | 0.0007858 | — | — |

## Calibration table

Each model seed is first averaged over 20 nested calibration replicates and then over the
seven frozen doses. Candidate entropy at B=0 equals log(10), as expected for ten candidates
per within-seed bank.

| B | Do MAE | Rank error | Regret | Gap to Oracle-best | Candidate entropy |
|---|---|---|---|---|---|
| 0 | 0.0168403 | 0.545354 | 0.00301179 | 0.0050062 | 2.30259 |
| 8 | 0.0138867 | 0.460855 | 0.00238844 | 0.00205264 | 1.84394 |
| 16 | 0.0132544 | 0.45473 | 0.00234147 | 0.00142028 | 1.64165 |
| 32 | 0.0131926 | 0.451581 | 0.00231805 | 0.0013585 | 1.42451 |
| 64 | 0.012955 | 0.451447 | 0.0023157 | 0.00112095 | 1.21867 |
| 128 | 0.0127865 | 0.452173 | 0.00232432 | 0.000952419 | 0.945642 |

The first positive tested budget, B=8, improves all three metrics relative to the uniform
B=0 ensemble. Relative to the stronger NLL-best single-model baseline, B=16 is the first
tested budget with lower aggregate Do MAE; no tested budget beats it on aggregate ranking
or regret.

## Statistical interpretation

All main comparisons are paired by model seed. Exact two-sided sign-flip tests are the
primary inferential check because n=5 is too small to rely on normal approximations.
Their smallest attainable non-zero two-sided p-value is 0.0625, so no five-seed contrast
can cross 0.05 even when all seeds agree. Paired t statistics, Shapiro-Wilk diagnostics,
standardized paired effects, descriptive t-based 95% CIs, and Holm corrections are retained
in `paired-method-contrasts.csv` and `calibration-contrasts.csv`; they are not used to
override the exact-test resolution limit.

## Interpretation of controls

- **λ=0:** public residual initialization degrades the no-direct-confounding control; this
  is evidence against unconditional safety.
- **Independent latents:** a single shared binary Z is misspecified and its apparent
  non-collapse cannot be described as recovery of the true U.
- **Source shuffle:** predictive gains survive while behavior recovery fails, separating
  outcome-mode clustering from source-mechanism recovery.
- **No staged training:** gains survive without staged freezing, so staged optimization
  is not established as a necessary ingredient.
- **κ=0.3:** the public method remains better than V0 on aggregate, but source shuffle is
  again similar and no-staged training is at least competitive.

## Direct answers

1. Public residual initialization eliminated measured collapse in this experiment.
2. It moved aggregate causal metrics toward V6, but did not reach V6 and harmed λ=0.
3. Source shuffle did not remove the gain; source recovery is therefore not the supported driver.
4. Staged training was not necessary in the observed data.
5. A near-V6 public candidate existed in the bank post hoc.
6. Observational NLL remained unreliable for causal selection: pooled MLP had the best
   validation NLL but poor high-dose causal performance, and NLL-best public selection
   retained a measurable gap to the Oracle-best candidate.
7. B=8 begins improving over the uniform candidate ensemble; B=16 begins improving Do MAE
   over the NLL-best public model.
8. Calibration improved Do MAE, ranking, and regret versus B=0, but only Do MAE versus NLL-best.
9. The supported mechanism is public outcome-residual initialization plus intervention-based
   candidate reweighting for Do-MAE; source-specific recovery and staged optimization are
   not supported as necessary causes.
10. These results do not yet justify transition-model or SAC expansion. Fresh anchor/policy
    seeds, a repaired λ=0 safety behavior, and an explicitly decision-aware calibration
    study are needed first.

## Claim boundary

This is a binary-latent, reward-only proof of concept. It neither identifies a continuous
confounder nor validates transition learning or end-to-end policy improvement. Oracle
components were used only after training for diagnosis. Because only five model seeds are
available and the test aggregate influenced algorithm development history, effect sizes
and consistency are more informative here than confirmatory significance claims.
