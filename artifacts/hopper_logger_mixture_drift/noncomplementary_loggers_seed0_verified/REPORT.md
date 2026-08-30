# Phase 8A-NC - Non-Complementary Logger Population Audit

This artifact uses exact probability-mass enumeration and only reuses fixed Phase 8A do-oracle outcomes. All hard checks passed and all hashed inputs were unchanged.

Logger 1 uses plus probabilities 0.9/0.1 for u=+1/-1; Logger 2 uses 0.7/0.3. Both therefore have plus/minus marginals 0.5/0.5, have the same confounding direction, and are not complementary. Primary mixtures preserve P(S,A)=(0.45,0.10,0.45) exactly.

## Analytic and empirical U posteriors

| mixture | minus | base | plus |
|---|---:|---:|---:|
| logger1_heavy | 0.122222222 | 0.5 | 0.877777778 |
| logger12_balanced | 0.2 | 0.5 | 0.8 |
| logger2_heavy | 0.277777778 | 0.5 | 0.722222222 |
| all_sources_equal | 0.2 | 0.5 | 0.8 |

The exact weighted empirical posteriors match these analytic values. In particular, logger12_balanced and all_sources_equal retain posteriors 0.2/0.8 for minus/plus, rather than the do(action) value 0.5.

## Main results

| kappa | metric | mean [anchor-bootstrap 95% interval] |
|---:|---|---:|
| 0.0 | |U reward effect| | 2.07444e-17 [6.03178e-18, 4.4202e-17] |
| 0.0 | heavy reward drift | 1.37947e-16 [1.32417e-16, 1.44127e-16] |
| 0.0 | balanced reward do-error | 1.85814e-16 [1.79217e-16, 1.9402e-16] |
| 0.0 | equal-source reward do-error | 7.41233e-17 [6.78711e-17, 8.17132e-17] |
| 0.0 | heavy delta drift | 8.42767e-17 [8.04102e-17, 8.83054e-17] |
| 0.0 | balanced delta do-error | 1.09605e-16 [1.05439e-16, 1.13772e-16] |
| 0.0 | heavy ranking disagreement | 0 [0, 0] |
| 0.0 | heavy strict flip | 0 [0, 0] |
| 0.0 | drift / action gap | 6.78693e-14 [6.51974e-14, 7.0503e-14] |
| 0.0 | balanced error / action gap | 5.93968e-14 [5.65906e-14, 6.2764e-14] |
| 0.1 | |U reward effect| | 0.00391534 [0.00381968, 0.00401449] |
| 0.1 | heavy reward drift | 0.000403819 [0.000394024, 0.000413973] |
| 0.1 | balanced reward do-error | 0.000778794 [0.000759903, 0.000798376] |
| 0.1 | equal-source reward do-error | 0.000778794 [0.000759903, 0.000798376] |
| 0.1 | heavy delta drift | 0.0271994 [0.0270019, 0.0273915] |
| 0.1 | balanced delta do-error | 0.052456 [0.0520751, 0.0528265] |
| 0.1 | heavy ranking disagreement | 0.00195312 [0.000488281, 0.00390625] |
| 0.1 | heavy strict flip | 0.00195312 [0.000488281, 0.00390625] |
| 0.1 | drift / action gap | 0.0829448 [0.0825022, 0.0834419] |
| 0.1 | balanced error / action gap | 0.159965 [0.159057, 0.160946] |
| 0.2 | |U reward effect| | 0.00778261 [0.00759298, 0.00797535] |
| 0.2 | heavy reward drift | 0.000802338 [0.000782603, 0.000822692] |
| 0.2 | balanced reward do-error | 0.00154737 [0.00150931, 0.00158662] |
| 0.2 | equal-source reward do-error | 0.00154737 [0.00150931, 0.00158662] |
| 0.2 | heavy delta drift | 0.0538312 [0.0534298, 0.0542049] |
| 0.2 | balanced delta do-error | 0.103817 [0.103043, 0.104538] |
| 0.2 | heavy ranking disagreement | 0.00146484 [0, 0.00341797] |
| 0.2 | heavy strict flip | 0.000976562 [0, 0.00244141] |
| 0.2 | drift / action gap | 0.170724 [0.169657, 0.171877] |
| 0.2 | balanced error / action gap | 0.329254 [0.327131, 0.33144] |
| 0.3 | |U reward effect| | 0.0115993 [0.0113169, 0.0118856] |
| 0.3 | heavy reward drift | 0.00119802 [0.00116863, 0.00122755] |
| 0.3 | balanced reward do-error | 0.00231047 [0.00225379, 0.00236742] |
| 0.3 | equal-source reward do-error | 0.00231047 [0.00225379, 0.00236742] |
| 0.3 | heavy delta drift | 0.0800067 [0.0794044, 0.0805988] |
| 0.3 | balanced delta do-error | 0.154299 [0.153137, 0.15544] |
| 0.3 | heavy ranking disagreement | 0.00537109 [0.00244141, 0.00878906] |
| 0.3 | heavy strict flip | 0.00537109 [0.00244141, 0.00878906] |
| 0.3 | drift / action gap | 0.263454 [0.261143, 0.265867] |
| 0.3 | balanced error / action gap | 0.50809 [0.503537, 0.512702] |

Each entry is an anchor-level mean and percentile bootstrap interval. Ratio rows are ratios of aggregate means and separately retain their numerator and denominator in aggregate_tables.csv.

## Kappa 0.3 clipping subsets

| subset | heavy reward drift | balanced reward do-error | heavy strict flip | balanced-vs-do disagreement |
|---|---:|---:|---:|---:|
| all | 0.00119802 [0.00116863, 0.00122755] | 0.00231047 [0.00225379, 0.00236742] | 0.00537109 [0.00244141, 0.00878906] | 0.015625 [0.0102539, 0.0209961] |
| strict_unclipped | 0.00110587 [0.00106255, 0.00114801] | 0.00213275 [0.00204921, 0.00221402] | 0.00240674 [0, 0.00601685] | 0.00120337 [0, 0.00361011] |
| any_clipping | 0.00126094 [0.00122137, 0.00130068] | 0.00243182 [0.0023555, 0.00250845] | 0.00739523 [0.00328677, 0.0131471] | 0.0254725 [0.0172555, 0.0345111] |

Subset differences are descriptive and are not identified as the causal effect of clipping.

## Mechanism and negative controls

- kappa_0p00: maximum over all four reward/delta identities = 1.77636e-15
- kappa_0p10: maximum over all four reward/delta identities = 1.86521e-15
- kappa_0p20: maximum over all four reward/delta identities = 2.0428e-15
- kappa_0p30: maximum over all four reward/delta identities = 1.86526e-15

Kappa=0, independent-latents, and base-action controls all passed. The independent condition has P(u_env=+1|action)=0.5 and zero weighted latent correlation for every mixture; its population responses equal do(action).

## Interpretation boundary

Supported: equal action marginals do not imply equal hidden-U composition; balancing same-direction non-complementary loggers does not cut the U-to-A and U-to-outcome path in this fixed DGP. The numerical tables quantify whether the remaining bias changes one-step action ranking at each kappa.

Not supported: a claim about arbitrary equal-source sampling, general hidden-confounding resolution, cross-behavior-policy-seed significance, or a causal clipping effect. The script deliberately leaves the overall scientific verdict manual.

Anchor ID is the statistical unit. Bootstrap intervals describe anchor variation for one fixed behavior checkpoint seed; support rows are not treated as independent repeats.
