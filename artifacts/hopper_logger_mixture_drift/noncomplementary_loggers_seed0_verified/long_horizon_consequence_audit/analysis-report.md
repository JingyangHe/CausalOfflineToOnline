# Phase 8A-NC-LH: Long-Horizon Causal Consequence Audit

## Estimand and design

This audit estimates a **fixed-policy finite-horizon intervention value**. The first commanded
action is fixed by the Phase 8A anchor table. Thereafter the
hidden-blind continuation is `0.5*(pi_500k([o,-1],deterministic=True)+pi_500k([o,+1],deterministic=True))`. Future hidden variables are iid balanced and
integrated exactly at H=5 or with common-random-number antithetic Monte Carlo at H=20/50.

Gamma is 0.99 and its recorded source is `explicit_cli`. The primary
cross-horizon population contains 1994 anchors that have enough
TimeLimit steps for the maximum requested horizon. Per-horizon eligible counts are
{'1': 2048, '5': 2047, '20': 2026, '50': 1994}.

## Integrity and numerical validation

All 35 hard checks passed. H=1 reproduced Phase 8A-NC with maximum absolute
difference 2.22933e-13. H=5 used all 16
equiprobable future-U sequences. Inputs were unchanged by SHA256 before and after analysis.

## Balanced observational-do return error

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 1994 | 7.64647e-17 | [7.25477e-17, 8.0493e-17] |
| 0 | 5 | logger12_balanced | 1994 | 3.10165e-16 | [2.90269e-16, 3.30955e-16] |
| 0 | 20 | logger12_balanced | 1994 | 1.27317e-15 | [1.20338e-15, 1.3437e-15] |
| 0 | 50 | logger12_balanced | 1994 | 2.23826e-15 | [2.07893e-15, 2.40861e-15] |
| 0.1 | 1 | logger12_balanced | 1994 | 0.000775835 | [0.000756438, 0.00079522] |
| 0.1 | 5 | logger12_balanced | 1994 | 0.00405032 | [0.00387173, 0.00425667] |
| 0.1 | 20 | logger12_balanced | 1994 | 0.00636313 | [0.00588112, 0.00685585] |
| 0.1 | 50 | logger12_balanced | 1994 | 0.0138375 | [0.0119446, 0.0159535] |
| 0.2 | 1 | logger12_balanced | 1994 | 0.00154121 | [0.00150273, 0.00157915] |
| 0.2 | 5 | logger12_balanced | 1994 | 0.00794827 | [0.0075954, 0.00835569] |
| 0.2 | 20 | logger12_balanced | 1994 | 0.0123167 | [0.011449, 0.0132495] |
| 0.2 | 50 | logger12_balanced | 1994 | 0.0241618 | [0.021474, 0.0271201] |
| 0.3 | 1 | logger12_balanced | 1994 | 0.00230131 | [0.00224472, 0.00236247] |
| 0.3 | 5 | logger12_balanced | 1994 | 0.0118419 | [0.0112845, 0.0125373] |
| 0.3 | 20 | logger12_balanced | 1994 | 0.0180478 | [0.0167539, 0.0194107] |
| 0.3 | 50 | logger12_balanced | 1994 | 0.0355356 | [0.0314049, 0.04021] |

## Initial-U effect and heavy-mixture drift

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | none | 1994 | 1.74458e-18 | [3.71188e-19, 3.60052e-18] |
| 0 | 5 | none | 1994 | 5.33026e-17 | [3.05859e-17, 8.27007e-17] |
| 0 | 20 | none | 1994 | 3.96132e-16 | [3.11189e-16, 4.87592e-16] |
| 0 | 50 | none | 1994 | 1.83159e-15 | [1.56546e-15, 2.11191e-15] |
| 0.1 | 1 | none | 1994 | 0.00390083 | [0.00380057, 0.00400089] |
| 0.1 | 5 | none | 1994 | 0.0202852 | [0.0194549, 0.0211404] |
| 0.1 | 20 | none | 1994 | 0.0319967 | [0.0295579, 0.0344411] |
| 0.1 | 50 | none | 1994 | 0.067394 | [0.0590701, 0.0769351] |
| 0.2 | 1 | none | 1994 | 0.00775236 | [0.00756117, 0.00794054] |
| 0.2 | 5 | none | 1994 | 0.040093 | [0.0383088, 0.0421237] |
| 0.2 | 20 | none | 1994 | 0.0619763 | [0.0574792, 0.0667318] |
| 0.2 | 50 | none | 1994 | 0.122899 | [0.108982, 0.13849] |
| 0.3 | 1 | none | 1994 | 0.0115537 | [0.0112801, 0.0118431] |
| 0.3 | 5 | none | 1994 | 0.0595639 | [0.0566695, 0.0631791] |
| 0.3 | 20 | none | 1994 | 0.0905034 | [0.084102, 0.097349] |
| 0.3 | 50 | none | 1994 | 0.177692 | [0.156325, 0.20093] |

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | none | 1994 | 4.59902e-17 | [4.25381e-17, 4.9387e-17] |
| 0 | 5 | none | 1994 | 2.08459e-16 | [1.92127e-16, 2.24346e-16] |
| 0 | 20 | none | 1994 | 8.12456e-16 | [7.50386e-16, 8.74816e-16] |
| 0 | 50 | none | 1994 | 1.28995e-15 | [1.1768e-15, 1.40814e-15] |
| 0.1 | 1 | none | 1994 | 0.000402285 | [0.000392, 0.000412153] |
| 0.1 | 5 | none | 1994 | 0.00210017 | [0.00201054, 0.00220753] |
| 0.1 | 20 | none | 1994 | 0.0032994 | [0.00305549, 0.00357139] |
| 0.1 | 50 | none | 1994 | 0.00717498 | [0.00620211, 0.00834489] |
| 0.2 | 1 | none | 1994 | 0.000799147 | [0.000777995, 0.000819639] |
| 0.2 | 5 | none | 1994 | 0.00412133 | [0.00393616, 0.00431632] |
| 0.2 | 20 | none | 1994 | 0.00638642 | [0.00591883, 0.00685411] |
| 0.2 | 50 | none | 1994 | 0.0125283 | [0.0110565, 0.0140779] |
| 0.3 | 1 | none | 1994 | 0.00119327 | [0.00116306, 0.00122336] |
| 0.3 | 5 | none | 1994 | 0.00614024 | [0.005845, 0.00649309] |
| 0.3 | 20 | none | 1994 | 0.00935813 | [0.00870026, 0.0100708] |
| 0.3 | 50 | none | 1994 | 0.0184258 | [0.0162948, 0.0208378] |

## Do decision scale

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | none | 1994 | 0.00784724 | [0.00763652, 0.00804556] |
| 0 | 5 | none | 1994 | 0.0404589 | [0.0389347, 0.0420447] |
| 0 | 20 | none | 1994 | 0.0662144 | [0.0602459, 0.0727801] |
| 0 | 50 | none | 1994 | 0.141662 | [0.124681, 0.16019] |
| 0.1 | 1 | none | 1994 | 0.00779697 | [0.00759759, 0.00800927] |
| 0.1 | 5 | none | 1994 | 0.0400898 | [0.0384933, 0.0418776] |
| 0.1 | 20 | none | 1994 | 0.0632359 | [0.0584212, 0.0681594] |
| 0.1 | 50 | none | 1994 | 0.124163 | [0.10789, 0.140025] |
| 0.2 | 1 | none | 1994 | 0.00765729 | [0.00746039, 0.00786028] |
| 0.2 | 5 | none | 1994 | 0.0391543 | [0.0375188, 0.0412169] |
| 0.2 | 20 | none | 1994 | 0.0609715 | [0.0567629, 0.0663008] |
| 0.2 | 50 | none | 1994 | 0.120118 | [0.106409, 0.134717] |
| 0.3 | 1 | none | 1994 | 0.00751195 | [0.00731908, 0.00771498] |
| 0.3 | 5 | none | 1994 | 0.0378607 | [0.0361493, 0.0398148] |
| 0.3 | 20 | none | 1994 | 0.0594263 | [0.0551372, 0.0637692] |
| 0.3 | 50 | none | 1994 | 0.122047 | [0.106866, 0.139522] |

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | none | 1994 | 0.00385055 | [0.00375672, 0.00395565] |
| 0 | 5 | none | 1994 | 0.020211 | [0.0193691, 0.0210641] |
| 0 | 20 | none | 1994 | 0.0317924 | [0.0293056, 0.034559] |
| 0 | 50 | none | 1994 | 0.0708387 | [0.0599717, 0.0826937] |
| 0.1 | 1 | none | 1994 | 0.00378215 | [0.00368391, 0.00388619] |
| 0.1 | 5 | none | 1994 | 0.0200339 | [0.01913, 0.0211201] |
| 0.1 | 20 | none | 1994 | 0.031673 | [0.0291088, 0.0343748] |
| 0.1 | 50 | none | 1994 | 0.0619616 | [0.0537662, 0.0712403] |
| 0.2 | 1 | none | 1994 | 0.00364201 | [0.00353941, 0.00374326] |
| 0.2 | 5 | none | 1994 | 0.018992 | [0.0181579, 0.020027] |
| 0.2 | 20 | none | 1994 | 0.0305768 | [0.0283663, 0.0328903] |
| 0.2 | 50 | none | 1994 | 0.0607735 | [0.0534961, 0.0684369] |
| 0.3 | 1 | none | 1994 | 0.0035733 | [0.00347941, 0.00367732] |
| 0.3 | 5 | none | 1994 | 0.0181342 | [0.0172577, 0.019195] |
| 0.3 | 20 | none | 1994 | 0.0301622 | [0.0278619, 0.0328605] |
| 0.3 | 50 | none | 1994 | 0.0645671 | [0.0540843, 0.0776489] |

## Ranking disagreement

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 20 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 50 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.1 | 1 | logger12_balanced | 1994 | 0.00300903 | [0.00100301, 0.00551655] |
| 0.1 | 5 | logger12_balanced | 1994 | 0.0145436 | [0.00952859, 0.0200602] |
| 0.1 | 20 | logger12_balanced | 1994 | 0.0145436 | [0.00952859, 0.0200602] |
| 0.1 | 50 | logger12_balanced | 1994 | 0.0260782 | [0.0190572, 0.0336008] |
| 0.2 | 1 | logger12_balanced | 1994 | 0.00150451 | [0, 0.00351053] |
| 0.2 | 5 | logger12_balanced | 1994 | 0.0100301 | [0.00601805, 0.0145436] |
| 0.2 | 20 | logger12_balanced | 1994 | 0.0160481 | [0.0105316, 0.0220662] |
| 0.2 | 50 | logger12_balanced | 1994 | 0.0270812 | [0.0200602, 0.0346038] |
| 0.3 | 1 | logger12_balanced | 1994 | 0.0160481 | [0.0110331, 0.0215647] |
| 0.3 | 5 | logger12_balanced | 1994 | 0.0225677 | [0.0165496, 0.0290873] |
| 0.3 | 20 | logger12_balanced | 1994 | 0.0190572 | [0.0135406, 0.0250752] |
| 0.3 | 50 | logger12_balanced | 1994 | 0.050652 | [0.0411234, 0.0601805] |

## Best-case decision regret under balanced observational selection

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger1_heavy | 1994 | 0 | [0, 0] |
| 0 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 1 | logger2_heavy | 1994 | 0 | [0, 0] |
| 0 | 5 | logger1_heavy | 1994 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 5 | logger2_heavy | 1994 | 0 | [0, 0] |
| 0 | 20 | logger1_heavy | 1994 | 0 | [0, 0] |
| 0 | 20 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 20 | logger2_heavy | 1994 | 0 | [0, 0] |
| 0 | 50 | logger1_heavy | 1994 | 0 | [0, 0] |
| 0 | 50 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 50 | logger2_heavy | 1994 | 0 | [0, 0] |
| 0.1 | 1 | logger1_heavy | 1994 | 1.28733e-06 | [4.28699e-07, 2.34495e-06] |
| 0.1 | 1 | logger12_balanced | 1994 | 7.33478e-07 | [1.30833e-07, 1.61666e-06] |
| 0.1 | 1 | logger2_heavy | 1994 | 3.93223e-07 | [8.73866e-08, 8.19639e-07] |
| 0.1 | 5 | logger1_heavy | 1994 | 1.35314e-05 | [7.91909e-06, 1.98717e-05] |
| 0.1 | 5 | logger12_balanced | 1994 | 1.10123e-05 | [5.99163e-06, 1.67344e-05] |
| 0.1 | 5 | logger2_heavy | 1994 | 7.60648e-06 | [3.70068e-06, 1.21886e-05] |
| 0.1 | 20 | logger1_heavy | 1994 | 2.91904e-05 | [1.24708e-05, 5.27115e-05] |
| 0.1 | 20 | logger12_balanced | 1994 | 2.39198e-05 | [8.29751e-06, 4.62438e-05] |
| 0.1 | 20 | logger2_heavy | 1994 | 1.99342e-05 | [5.42306e-06, 4.21786e-05] |
| 0.1 | 50 | logger1_heavy | 1994 | 7.99973e-05 | [4.50102e-05, 0.000124582] |
| 0.1 | 50 | logger12_balanced | 1994 | 5.18855e-05 | [2.77192e-05, 8.17113e-05] |
| 0.1 | 50 | logger2_heavy | 1994 | 4.14211e-05 | [2.01068e-05, 6.98793e-05] |
| 0.2 | 1 | logger1_heavy | 1994 | 8.39367e-07 | [0, 2.25447e-06] |
| 0.2 | 1 | logger12_balanced | 1994 | 8.39367e-07 | [0, 2.25447e-06] |
| 0.2 | 1 | logger2_heavy | 1994 | 1.85035e-08 | [0, 5.55105e-08] |
| 0.2 | 5 | logger1_heavy | 1994 | 1.61824e-05 | [8.47206e-06, 2.51903e-05] |
| 0.2 | 5 | logger12_balanced | 1994 | 9.80937e-06 | [4.42893e-06, 1.64288e-05] |
| 0.2 | 5 | logger2_heavy | 1994 | 6.98188e-06 | [2.4887e-06, 1.25469e-05] |
| 0.2 | 20 | logger1_heavy | 1994 | 3.853e-05 | [1.51307e-05, 7.11285e-05] |
| 0.2 | 20 | logger12_balanced | 1994 | 3.38691e-05 | [1.24884e-05, 6.35138e-05] |
| 0.2 | 20 | logger2_heavy | 1994 | 2.89638e-05 | [8.16049e-06, 5.80493e-05] |
| 0.2 | 50 | logger1_heavy | 1994 | 0.00042663 | [7.62911e-05, 0.00105274] |
| 0.2 | 50 | logger12_balanced | 1994 | 0.000388111 | [4.68988e-05, 0.00100552] |
| 0.2 | 50 | logger2_heavy | 1994 | 9.86707e-05 | [3.06783e-05, 0.000193406] |
| 0.3 | 1 | logger1_heavy | 1994 | 6.28662e-06 | [3.83e-06, 9.18675e-06] |
| 0.3 | 1 | logger12_balanced | 1994 | 4.6941e-06 | [2.84123e-06, 6.88812e-06] |
| 0.3 | 1 | logger2_heavy | 1994 | 2.94383e-06 | [1.64454e-06, 4.40634e-06] |
| 0.3 | 5 | logger1_heavy | 1994 | 4.22884e-05 | [2.62746e-05, 6.05497e-05] |
| 0.3 | 5 | logger12_balanced | 1994 | 3.41221e-05 | [1.96224e-05, 5.22612e-05] |
| 0.3 | 5 | logger2_heavy | 1994 | 2.74306e-05 | [1.65974e-05, 3.96716e-05] |
| 0.3 | 20 | logger1_heavy | 1994 | 9.48831e-05 | [4.25197e-05, 0.000160228] |
| 0.3 | 20 | logger12_balanced | 1994 | 8.34864e-05 | [3.47788e-05, 0.000150884] |
| 0.3 | 20 | logger2_heavy | 1994 | 7.98446e-05 | [3.15791e-05, 0.000145012] |
| 0.3 | 50 | logger1_heavy | 1994 | 0.0012403 | [0.000435554, 0.00247362] |
| 0.3 | 50 | logger12_balanced | 1994 | 0.00122045 | [0.000405599, 0.00241395] |
| 0.3 | 50 | logger2_heavy | 1994 | 0.0010633 | [0.000268995, 0.0022899] |

## Worst-case tie regret

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger1_heavy | 1994 | 0 | [0, 0] |
| 0 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 1 | logger2_heavy | 1994 | 0 | [0, 0] |
| 0 | 5 | logger1_heavy | 1994 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 5 | logger2_heavy | 1994 | 0 | [0, 0] |
| 0 | 20 | logger1_heavy | 1994 | 0 | [0, 0] |
| 0 | 20 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 20 | logger2_heavy | 1994 | 0 | [0, 0] |
| 0 | 50 | logger1_heavy | 1994 | 0 | [0, 0] |
| 0 | 50 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 50 | logger2_heavy | 1994 | 0 | [0, 0] |
| 0.1 | 1 | logger1_heavy | 1994 | 1.28733e-06 | [4.37072e-07, 2.43794e-06] |
| 0.1 | 1 | logger12_balanced | 1994 | 7.33478e-07 | [1.58005e-07, 1.58885e-06] |
| 0.1 | 1 | logger2_heavy | 1994 | 3.93223e-07 | [7.069e-08, 7.86959e-07] |
| 0.1 | 5 | logger1_heavy | 1994 | 1.35314e-05 | [7.76554e-06, 2.01439e-05] |
| 0.1 | 5 | logger12_balanced | 1994 | 1.10123e-05 | [5.9631e-06, 1.67067e-05] |
| 0.1 | 5 | logger2_heavy | 1994 | 7.6647e-06 | [3.90205e-06, 1.20244e-05] |
| 0.1 | 20 | logger1_heavy | 1994 | 2.93399e-05 | [1.26069e-05, 5.23848e-05] |
| 0.1 | 20 | logger12_balanced | 1994 | 2.39805e-05 | [8.83603e-06, 4.73825e-05] |
| 0.1 | 20 | logger2_heavy | 1994 | 1.99342e-05 | [5.92596e-06, 4.2628e-05] |
| 0.1 | 50 | logger1_heavy | 1994 | 7.99973e-05 | [4.63914e-05, 0.000123025] |
| 0.1 | 50 | logger12_balanced | 1994 | 5.18855e-05 | [2.82416e-05, 8.1169e-05] |
| 0.1 | 50 | logger2_heavy | 1994 | 4.15789e-05 | [1.8386e-05, 7.03271e-05] |
| 0.2 | 1 | logger1_heavy | 1994 | 8.48296e-07 | [8.92929e-09, 2.36456e-06] |
| 0.2 | 1 | logger12_balanced | 1994 | 8.39367e-07 | [0, 2.19369e-06] |
| 0.2 | 1 | logger2_heavy | 1994 | 1.85035e-08 | [0, 5.55105e-08] |
| 0.2 | 5 | logger1_heavy | 1994 | 1.66367e-05 | [8.7838e-06, 2.62218e-05] |
| 0.2 | 5 | logger12_balanced | 1994 | 9.80937e-06 | [4.45311e-06, 1.62004e-05] |
| 0.2 | 5 | logger2_heavy | 1994 | 7.17823e-06 | [2.66509e-06, 1.29039e-05] |
| 0.2 | 20 | logger1_heavy | 1994 | 3.853e-05 | [1.44499e-05, 6.89781e-05] |
| 0.2 | 20 | logger12_balanced | 1994 | 3.38691e-05 | [1.16146e-05, 6.27362e-05] |
| 0.2 | 20 | logger2_heavy | 1994 | 2.89638e-05 | [8.38865e-06, 5.90424e-05] |
| 0.2 | 50 | logger1_heavy | 1994 | 0.00042663 | [7.75343e-05, 0.00104436] |
| 0.2 | 50 | logger12_balanced | 1994 | 0.000389343 | [4.58695e-05, 0.000997301] |
| 0.2 | 50 | logger2_heavy | 1994 | 0.000100629 | [3.19969e-05, 0.00019626] |
| 0.3 | 1 | logger1_heavy | 1994 | 6.28662e-06 | [3.83501e-06, 8.94653e-06] |
| 0.3 | 1 | logger12_balanced | 1994 | 4.6941e-06 | [2.85088e-06, 6.78218e-06] |
| 0.3 | 1 | logger2_heavy | 1994 | 2.94383e-06 | [1.74469e-06, 4.30269e-06] |
| 0.3 | 5 | logger1_heavy | 1994 | 4.22884e-05 | [2.614e-05, 5.97006e-05] |
| 0.3 | 5 | logger12_balanced | 1994 | 3.41221e-05 | [1.96549e-05, 5.22179e-05] |
| 0.3 | 5 | logger2_heavy | 1994 | 2.74306e-05 | [1.64938e-05, 4.00511e-05] |
| 0.3 | 20 | logger1_heavy | 1994 | 9.48831e-05 | [4.18554e-05, 0.000159503] |
| 0.3 | 20 | logger12_balanced | 1994 | 8.34864e-05 | [3.26915e-05, 0.000151409] |
| 0.3 | 20 | logger2_heavy | 1994 | 7.98446e-05 | [2.96596e-05, 0.00014935] |
| 0.3 | 50 | logger1_heavy | 1994 | 0.0012403 | [0.000403231, 0.00245521] |
| 0.3 | 50 | logger12_balanced | 1994 | 0.00122073 | [0.000408742, 0.00253667] |
| 0.3 | 50 | logger2_heavy | 1994 | 0.0010633 | [0.000276161, 0.00227714] |

## Survival, termination, and future clipping audits

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 20 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 50 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.1 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.1 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.1 | 20 | logger12_balanced | 1994 | 6.26881e-06 | [0, 1.88064e-05] |
| 0.1 | 50 | logger12_balanced | 1994 | 0.000108137 | [5.17177e-05, 0.000178661] |
| 0.2 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.2 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.2 | 20 | logger12_balanced | 1994 | 1.41048e-05 | [4.7016e-06, 2.66424e-05] |
| 0.2 | 50 | logger12_balanced | 1994 | 0.000186497 | [0.000105003, 0.00028527] |
| 0.3 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.3 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.3 | 20 | logger12_balanced | 1994 | 2.97768e-05 | [7.83601e-06, 5.64193e-05] |
| 0.3 | 50 | logger12_balanced | 1994 | 0.000242916 | [0.000153547, 0.000344784] |

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 20 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 50 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.1 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.1 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.1 | 20 | logger12_balanced | 1994 | 6.26881e-06 | [0, 1.72392e-05] |
| 0.1 | 50 | logger12_balanced | 1994 | 0.000108137 | [5.32849e-05, 0.000177094] |
| 0.2 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.2 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.2 | 20 | logger12_balanced | 1994 | 1.41048e-05 | [3.1344e-06, 2.82096e-05] |
| 0.2 | 50 | logger12_balanced | 1994 | 0.000186497 | [0.00010657, 0.000286837] |
| 0.3 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.3 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.3 | 20 | logger12_balanced | 1994 | 2.97768e-05 | [9.40321e-06, 5.48521e-05] |
| 0.3 | 50 | logger12_balanced | 1994 | 0.000242916 | [0.000155153, 0.000344784] |

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 20 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 50 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.1 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.1 | 5 | logger12_balanced | 1994 | 0.0429199 | [0.0380646, 0.0478326] |
| 0.1 | 20 | logger12_balanced | 1994 | 0.0876938 | [0.0817382, 0.0943016] |
| 0.1 | 50 | logger12_balanced | 1994 | 0.129965 | [0.125574, 0.134332] |
| 0.2 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.2 | 5 | logger12_balanced | 1994 | 0.0756098 | [0.0694424, 0.0820035] |
| 0.2 | 20 | logger12_balanced | 1994 | 0.124205 | [0.116894, 0.131187] |
| 0.2 | 50 | logger12_balanced | 1994 | 0.178588 | [0.173763, 0.183428] |
| 0.3 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.3 | 5 | logger12_balanced | 1994 | 0.120837 | [0.113568, 0.128028] |
| 0.3 | 20 | logger12_balanced | 1994 | 0.156563 | [0.149253, 0.164334] |
| 0.3 | 50 | logger12_balanced | 1994 | 0.217752 | [0.21232, 0.2229] |

## Horizon amplification

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 1994 | undefined | undefined |
| 0 | 5 | logger12_balanced | 1994 | undefined | undefined |
| 0 | 20 | logger12_balanced | 1994 | undefined | undefined |
| 0 | 50 | logger12_balanced | 1994 | undefined | undefined |
| 0.1 | 1 | logger12_balanced | 1994 | 1 | [1, 1] |
| 0.1 | 5 | logger12_balanced | 1994 | 5.2206 | [4.96116, 5.52404] |
| 0.1 | 20 | logger12_balanced | 1994 | 8.20165 | [7.5513, 8.93217] |
| 0.1 | 50 | logger12_balanced | 1994 | 17.8356 | [15.3491, 20.5827] |
| 0.2 | 1 | logger12_balanced | 1994 | 1 | [1, 1] |
| 0.2 | 5 | logger12_balanced | 1994 | 5.15716 | [4.89594, 5.45163] |
| 0.2 | 20 | logger12_balanced | 1994 | 7.99154 | [7.32697, 8.71267] |
| 0.2 | 50 | logger12_balanced | 1994 | 15.6771 | [13.8319, 17.7602] |
| 0.3 | 1 | logger12_balanced | 1994 | 1 | [1, 1] |
| 0.3 | 5 | logger12_balanced | 1994 | 5.14571 | [4.87091, 5.49615] |
| 0.3 | 20 | logger12_balanced | 1994 | 7.8424 | [7.22223, 8.50603] |
| 0.3 | 50 | logger12_balanced | 1994 | 15.4414 | [13.4724, 17.5578] |

| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |
|---:|---:|:---|---:|---:|---:|
| 0 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 5 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 20 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0 | 50 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.1 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.1 | 5 | logger12_balanced | 1994 | 1.02788e-05 | [5.09457e-06, 1.5912e-05] |
| 0.1 | 20 | logger12_balanced | 1994 | 2.31863e-05 | [7.91551e-06, 4.62772e-05] |
| 0.1 | 50 | logger12_balanced | 1994 | 5.1152e-05 | [2.73069e-05, 7.98904e-05] |
| 0.2 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.2 | 5 | logger12_balanced | 1994 | 8.97e-06 | [3.31104e-06, 1.5618e-05] |
| 0.2 | 20 | logger12_balanced | 1994 | 3.30298e-05 | [1.0781e-05, 6.40371e-05] |
| 0.2 | 50 | logger12_balanced | 1994 | 0.000387272 | [4.19929e-05, 0.00105394] |
| 0.3 | 1 | logger12_balanced | 1994 | 0 | [0, 0] |
| 0.3 | 5 | logger12_balanced | 1994 | 2.9428e-05 | [1.56169e-05, 4.73161e-05] |
| 0.3 | 20 | logger12_balanced | 1994 | 7.87923e-05 | [2.77596e-05, 0.000145542] |
| 0.3 | 50 | logger12_balanced | 1994 | 0.00121575 | [0.000380364, 0.00232708] |

## Monte Carlo integration audit

| kappa | H | R | mean branch SE | R/2-to-R error change | H5 MC-exact |
|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 32 | 3.01659e-15 | 2.9695e-18 | 5.57368e-15 |
| 0 | 50 | 32 | 6.37074e-15 | 5.46389e-17 | 5.57368e-15 |
| 0.1 | 20 | 32 | 0.00303859 | 6.36325e-06 | 0.000145059 |
| 0.1 | 50 | 32 | 0.0280295 | 0.0013695 | 0.000145059 |
| 0.2 | 20 | 32 | 0.00577544 | 2.71595e-05 | 0.000334268 |
| 0.2 | 50 | 32 | 0.047551 | 0.00205706 | 0.000334268 |
| 0.3 | 20 | 32 | 0.00916216 | 7.62589e-05 | 0.000600429 |
| 0.3 | 50 | 32 | 0.0707181 | 0.00185405 | 0.000600429 |

## Negative controls and subsets

The kappa=0 initial-U equality, independent-latent equality to do, and base-action equality to do
all passed at numerical tolerance. The **initial-step strict-unclipped subset** contains
831 selected anchors, including
811 in the primary common-horizon population.
The any-initial-clipping comparison is descriptive and future clipping is reported without filtering.

## Supported claims

The audited tables support statements about measured initial-selection consequences for the named
fixed-policy finite-horizon intervention value, within the selected anchors and fixed continuation.

## Unsupported claims

These results do not support unrestricted control claims, full historical-logger trajectory-return
claims, cross-policy-seed generalization, or a pure causal interpretation of clipping-subset
differences.

## Interpretation limits

The intervals describe anchor-level variation for one fixed 500k behavior-policy checkpoint; they
are not cross-policy-seed significance statements. Initial clipping subsets are descriptive, not a
causal clipping comparison. Future clipping remains part of the closed-loop Hopper outcome and is
never used to filter trajectories. No scientific-success threshold was applied. The numerical
tables support only the measured fixed-policy finite-horizon consequences under this continuation,
not unrestricted control, a full historical-logger trajectory return, or generalization across policy
seeds.
