# Analysis report

## Question

Does changing logger composition alter the population observational response while fixed anchors,
exact commanded-action mass, and the two-point do response remain unchanged?

## Evidence

| kappa | reward U effect | delta U effect | conf reward drift | ind reward drift | conf delta drift | ind delta drift | heavy ranking diff | strict flip | drift/action gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 2.0744402e-17 | 0 | 2.3491047e-18 | 4.8066296e-18 | 0 | 1.1096918e-16 | 0 | 0 | 3.3993159e-14 |
| 0.10 | 0.0039153429 | 0.26298005 | 0.0020190968 | 2.511735e-17 | 0.13599706 | 9.80449e-17 | 0.0087890625 | 0.0087890625 | 0.41472415 |
| 0.20 | 0.0077826082 | 0.52196706 | 0.0040116878 | 1.8323017e-17 | 0.26915612 | 9.6168386e-17 | 0.01171875 | 0.011230469 | 0.85362235 |
| 0.30 | 0.011599254 | 0.77617177 | 0.0059901097 | 2.1503343e-17 | 0.40003361 | 9.1101702e-17 | 0.97167969 | 0.97167969 | 1.3172698 |

The exact mechanism identities and all negative controls are listed in `hard_checks.json`. Detailed
effect distributions and intervals are in `aggregate_tables.csv`.

## Boundary

This is a deterministic population review over one behavior-policy seed. It supports no p-value or
cross-seed generalization claim and applies no automatic scientific-success threshold.
