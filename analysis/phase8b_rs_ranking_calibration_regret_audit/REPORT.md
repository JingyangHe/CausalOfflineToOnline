# Phase 8B-RS Ranking–Calibration–Regret Audit

## Scope and evidence

This is a read-only **one-step reward** audit. `R_do`, `R_obs`, and `R_nn` are not Q-values. All results use the 78 held-out test anchors and three saved neural seeds. Anchor is the decision unit; seed variation is reported descriptively and seed×anchor rows are not treated as independent experimental replications.

## Compact primary table

| lambda | obs→do rank error | neural→obs rank error | neural→do rank error | obs regret | neural regret | neural calibration MAE | neural gap error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0128 | 0.3554 | 0.3476 | 0.0000 | 0.0014 | 0.0240 | 0.0058 |
| 0.05 | 0.9487 | 0.0356 | 0.9316 | 0.0057 | 0.0055 | 0.0334 | 0.0241 |
| 0.10 | 0.9487 | 0.0157 | 0.9409 | 0.0057 | 0.0056 | 0.0452 | 0.0364 |
| 0.20 | 0.9487 | 0.0057 | 0.9444 | 0.0057 | 0.0056 | 0.0688 | 0.0604 |

Rank error is exact top-set disagreement under the existing project tolerance (`atol=1e-07`, `rtol=1e-07`). Regret uses the best member of a tied top set.

## Direct answers

**Q1. Is observational ranking usually correct?** **No** under the literal majority criterion. At λ=0.20, observational-vs-do top-set disagreement is 0.9487; the complete dose response is in the table.

**Q2. How much additional ranking error comes from the neural model?** At λ=0.20, neural-vs-do disagreement is 0.9444, while neural-vs-observational disagreement is 0.0057; the net neural-minus-observational do-error is -0.0043. Type B isolates neural-created errors and Type C isolates inherited observational errors in `failure_type_metrics.csv`.

**Q3. Common calibration or action-dependent distortion?** Mean |common shift| is 0.0342; mean max-action centered distortion is 0.0323. Their full distributions, rather than a thresholded label, are retained in `anchor_action_metrics.npz`. On mean magnitude, **common calibration shift** is larger.

**Q4. Is the top-second gap systematically distorted?** At λ=0.20, the mean absolute neural gap error is 0.0604, and the signed mean indicates **缩小** (-0.0481). Matched-top over/underconfidence fractions are in `gap_metrics.csv`.

**Q5. Where does decision regret come from?** At λ=0.20, mean observational regret is 0.0057, mean neural regret is 0.0056, and their difference is -0.0000. By mean positive contribution, **observational confounding** is larger. The signed additional-regret distribution separates neural harm from accidental correction.

**Q6. Does increasing λ move from calibration error to decision error?** From λ=0.00 to λ=0.20, observational rank error changes 0.0128→0.9487, and neural rank error changes 0.3476→0.9444. This is one-step evidence only.

**Q7. Which prior is it closer to?** Relative to a uniform-random action baseline, the saved model is closer to **B：存在直接误导 one-step 决策的风险**. This does not establish SAC transfer, positive or negative.

## Ranking-correct versus ranking-wrong anchors

No arbitrary 'large error' threshold was introduced. Ranking correctness is discrete; calibration, gap error, centered bias, and regret remain continuous and are shown by distributions and scatter plots. Thus categories I/II and III/IV are represented as continua rather than forced binary counts.

## Negative controls

`independent_latents` is analyzed with the identical pipeline. Any neural ranking, gap, or regret error there is ordinary approximation error, not direct reward confounding. Base-action action-level errors and λ=0 are retained in `calibration_metrics.csv`.

## Hard checks

All checks passed: **True**. Maximum identity residual `b_nn-(b_obs+e_nn)` = 0.000e+00. Input hashes were unchanged.

## Boundary

The audit describes three discrete candidate actions at fixed states and one-step rewards. It cannot by itself prove online SAC negative transfer or long-horizon policy value.
