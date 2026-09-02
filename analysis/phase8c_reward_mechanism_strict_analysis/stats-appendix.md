# Statistical appendix

## Analysis units and procedures

- Model-training replication unit: seed (n=5; seeds 0–4).
- Held-out evaluation set: 309 fixed anchors × 3 controlled actions.
- Primary summary: normalized trapezoidal AUC across the seven frozen λ values.
- Paired uncertainty: t-based 95% CI on seed-paired AUC differences.
- Test: exact two-sided sign-flip randomization test over all 32 assignments.
- Effect size: small-sample-corrected paired standardized mean difference (Hedges g_z).
- Multiplicity: Holm correction across 24 primary method-by-metric contrasts.
- Shapiro–Wilk is recorded diagnostically, but n=5 is too small to validate normality reliably.

## All primary contrasts

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

## Confounding difference-in-differences

Positive values mean the confounded-minus-independent penalty increased from κ=0 to κ=0.3. Holm correction covers all 27 method-by-metric interactions.

| Method | Metric | DiD | 95% CI | exact p | Holm p |
|---|---|---:|---:|---:|---:|
| pooled_mlp | mae | -0.0012987 | [-0.00185637, -0.000741031] | 0.0625 | 1 |
| pooled_mlp | top_set_disagreement | -0.120045 | [-0.136687, -0.103403] | 0.0625 | 1 |
| pooled_mlp | mean_regret | -0.00094844 | [-0.000983729, -0.000913152] | 0.0625 | 1 |
| source_balanced_pooled_mlp | mae | -0.0012987 | [-0.00185637, -0.000741031] | 0.0625 | 1 |
| source_balanced_pooled_mlp | top_set_disagreement | -0.120045 | [-0.136687, -0.103403] | 0.0625 | 1 |
| source_balanced_pooled_mlp | mean_regret | -0.00094844 | [-0.000983729, -0.000913152] | 0.0625 | 1 |
| source_conditioned_decoder | mae | -0.00128232 | [-0.0019458, -0.000618835] | 0.0625 | 1 |
| source_conditioned_decoder | top_set_disagreement | -0.113984 | [-0.13242, -0.0955473] | 0.0625 | 1 |
| source_conditioned_decoder | mean_regret | -0.000931029 | [-0.00104502, -0.000817035] | 0.0625 | 1 |
| per_source_models | mae | -0.000920524 | [-0.00123879, -0.000602255] | 0.0625 | 1 |
| per_source_models | top_set_disagreement | -0.101738 | [-0.111784, -0.0916914] | 0.0625 | 1 |
| per_source_models | mean_regret | -0.000907936 | [-0.000960273, -0.000855599] | 0.0625 | 1 |
| mechanism_separated | mae | -0.00137328 | [-0.00188125, -0.00086532] | 0.0625 | 1 |
| mechanism_separated | top_set_disagreement | -0.129916 | [-0.162453, -0.0973791] | 0.0625 | 1 |
| mechanism_separated | mean_regret | -0.00101951 | [-0.00118011, -0.000858917] | 0.0625 | 1 |
| oracle_u_aware | mae | -0.00068948 | [-0.00165621, 0.000277246] | 0.1875 | 1 |
| oracle_u_aware | top_set_disagreement | -0.0105243 | [-0.0447316, 0.0236831] | 0.4375 | 1 |
| oracle_u_aware | mean_regret | -2.16091e-05 | [-0.000124834, 8.16156e-05] | 0.5625 | 1 |
| source_shuffle | mae | -0.00112047 | [-0.00157082, -0.00067011] | 0.0625 | 1 |
| source_shuffle | top_set_disagreement | -0.112068 | [-0.139556, -0.0845797] | 0.0625 | 1 |
| source_shuffle | mean_regret | -0.000948358 | [-0.00108093, -0.000815787] | 0.0625 | 1 |
| no_behavior | mae | -0.00122543 | [-0.00187915, -0.000571711] | 0.0625 | 1 |
| no_behavior | top_set_disagreement | -0.104353 | [-0.127688, -0.0810179] | 0.0625 | 1 |
| no_behavior | mean_regret | -0.000867357 | [-0.00101716, -0.000717552] | 0.0625 | 1 |
| source_dependent_reward | mae | -0.00177594 | [-0.00314795, -0.000403936] | 0.0625 | 1 |
| source_dependent_reward | top_set_disagreement | -0.116298 | [-0.139254, -0.0933411] | 0.0625 | 1 |
| source_dependent_reward | mean_regret | -0.000942503 | [-0.00106282, -0.000822185] | 0.0625 | 1 |

## Confounded-minus-independent gaps

Positive values mean worse AUC under confounded logging than independent latents at the same κ.

| κ | Method | Metric | Gap | 95% CI | exact p | Holm p |
|---:|---|---|---:|---:|---:|---:|
| 0.0 | pooled_mlp | mae | 0.0128277 | [0.0120662, 0.0135893] | 0.0625 | 1 |
| 0.0 | pooled_mlp | top_set_disagreement | 0.594139 | [0.577692, 0.610586] | 0.0625 | 1 |
| 0.0 | pooled_mlp | mean_regret | 0.00495675 | [0.00484915, 0.00506434] | 0.0625 | 1 |
| 0.0 | source_balanced_pooled_mlp | mae | 0.0128277 | [0.0120662, 0.0135893] | 0.0625 | 1 |
| 0.0 | source_balanced_pooled_mlp | top_set_disagreement | 0.594139 | [0.577692, 0.610586] | 0.0625 | 1 |
| 0.0 | source_balanced_pooled_mlp | mean_regret | 0.00495675 | [0.00484915, 0.00506434] | 0.0625 | 1 |
| 0.0 | source_conditioned_decoder | mae | 0.0127286 | [0.0115866, 0.0138705] | 0.0625 | 1 |
| 0.0 | source_conditioned_decoder | top_set_disagreement | 0.625693 | [0.586037, 0.665349] | 0.0625 | 1 |
| 0.0 | source_conditioned_decoder | mean_regret | 0.00505967 | [0.00485456, 0.00526478] | 0.0625 | 1 |
| 0.0 | per_source_models | mae | 0.0107597 | [0.0094031, 0.0121162] | 0.0625 | 1 |
| 0.0 | per_source_models | top_set_disagreement | 0.595883 | [0.5759, 0.615867] | 0.0625 | 1 |
| 0.0 | per_source_models | mean_regret | 0.00491052 | [0.0047864, 0.00503463] | 0.0625 | 1 |
| 0.0 | mechanism_separated | mae | 0.0124951 | [0.0119616, 0.0130286] | 0.0625 | 1 |
| 0.0 | mechanism_separated | top_set_disagreement | 0.602867 | [0.572654, 0.633081] | 0.0625 | 1 |
| 0.0 | mechanism_separated | mean_regret | 0.00499758 | [0.00483823, 0.00515693] | 0.0625 | 1 |
| 0.0 | oracle_u_aware | mae | 0.000550303 | [-0.000587903, 0.00168851] | 0.25 | 1 |
| 0.0 | oracle_u_aware | top_set_disagreement | 0.0347087 | [-0.00568072, 0.0750982] | 0.125 | 1 |
| 0.0 | oracle_u_aware | mean_regret | 0.000131942 | [2.47807e-06, 0.000261406] | 0.0625 | 1 |
| 0.0 | source_shuffle | mae | 0.0122963 | [0.0115408, 0.0130517] | 0.0625 | 1 |
| 0.0 | source_shuffle | top_set_disagreement | 0.600909 | [0.568007, 0.633812] | 0.0625 | 1 |
| 0.0 | source_shuffle | mean_regret | 0.00501221 | [0.00484194, 0.00518248] | 0.0625 | 1 |
| 0.0 | no_behavior | mae | 0.0127122 | [0.011545, 0.0138794] | 0.0625 | 1 |
| 0.0 | no_behavior | top_set_disagreement | 0.608285 | [0.571551, 0.645019] | 0.0625 | 1 |
| 0.0 | no_behavior | mean_regret | 0.0050249 | [0.00483528, 0.00521452] | 0.0625 | 1 |
| 0.0 | source_dependent_reward | mae | 0.0126656 | [0.0115101, 0.0138211] | 0.0625 | 1 |
| 0.0 | source_dependent_reward | top_set_disagreement | 0.622524 | [0.584021, 0.661027] | 0.0625 | 1 |
| 0.0 | source_dependent_reward | mean_regret | 0.0050742 | [0.00488964, 0.00525877] | 0.0625 | 1 |
| 0.3 | pooled_mlp | mae | 0.011529 | [0.011102, 0.0119561] | 0.0625 | 1 |
| 0.3 | pooled_mlp | top_set_disagreement | 0.474094 | [0.449132, 0.499056] | 0.0625 | 1 |
| 0.3 | pooled_mlp | mean_regret | 0.00400831 | [0.00388433, 0.00413229] | 0.0625 | 1 |
| 0.3 | source_balanced_pooled_mlp | mae | 0.011529 | [0.011102, 0.0119561] | 0.0625 | 1 |
| 0.3 | source_balanced_pooled_mlp | top_set_disagreement | 0.474094 | [0.449132, 0.499056] | 0.0625 | 1 |
| 0.3 | source_balanced_pooled_mlp | mean_regret | 0.00400831 | [0.00388433, 0.00413229] | 0.0625 | 1 |
| 0.3 | source_conditioned_decoder | mae | 0.0114463 | [0.0105654, 0.0123271] | 0.0625 | 1 |
| 0.3 | source_conditioned_decoder | top_set_disagreement | 0.511709 | [0.486273, 0.537144] | 0.0625 | 1 |
| 0.3 | source_conditioned_decoder | mean_regret | 0.00412864 | [0.00397041, 0.00428687] | 0.0625 | 1 |
| 0.3 | per_source_models | mae | 0.00983913 | [0.00868936, 0.0109889] | 0.0625 | 1 |
| 0.3 | per_source_models | top_set_disagreement | 0.494146 | [0.473569, 0.514723] | 0.0625 | 1 |
| 0.3 | per_source_models | mean_regret | 0.00400258 | [0.00391701, 0.00408815] | 0.0625 | 1 |
| 0.3 | mechanism_separated | mae | 0.0111219 | [0.0102537, 0.01199] | 0.0625 | 1 |
| 0.3 | mechanism_separated | top_set_disagreement | 0.472951 | [0.454807, 0.491096] | 0.0625 | 1 |
| 0.3 | mechanism_separated | mean_regret | 0.00397806 | [0.00388527, 0.00407086] | 0.0625 | 1 |
| 0.3 | oracle_u_aware | mae | -0.000139178 | [-0.000897108, 0.000618753] | 0.75 | 1 |
| 0.3 | oracle_u_aware | top_set_disagreement | 0.0241845 | [0.00561093, 0.042758] | 0.0625 | 1 |
| 0.3 | oracle_u_aware | mean_regret | 0.000110333 | [6.40979e-05, 0.000156568] | 0.0625 | 1 |
| 0.3 | source_shuffle | mae | 0.0111758 | [0.0101671, 0.0121846] | 0.0625 | 1 |
| 0.3 | source_shuffle | top_set_disagreement | 0.488841 | [0.461744, 0.515939] | 0.0625 | 1 |
| 0.3 | source_shuffle | mean_regret | 0.00406385 | [0.00391635, 0.00421136] | 0.0625 | 1 |
| 0.3 | no_behavior | mae | 0.0114868 | [0.0104831, 0.0124905] | 0.0625 | 1 |
| 0.3 | no_behavior | top_set_disagreement | 0.503932 | [0.48352, 0.524344] | 0.0625 | 1 |
| 0.3 | no_behavior | mean_regret | 0.00415754 | [0.00399093, 0.00432416] | 0.0625 | 1 |
| 0.3 | source_dependent_reward | mae | 0.0108896 | [0.0102967, 0.0114826] | 0.0625 | 1 |
| 0.3 | source_dependent_reward | top_set_disagreement | 0.506227 | [0.472638, 0.539815] | 0.0625 | 1 |
| 0.3 | source_dependent_reward | mean_regret | 0.0041317 | [0.00393452, 0.00432888] | 0.0625 | 1 |

## Cross-mixture paired contrasts

Positive comparator-minus-mechanism values favor mechanism stability. The source-balanced pooled contrast is structurally zero for that method itself and should not be interpreted as learned invariance.

| Comparator | Improvement | 95% CI | exact p | Holm p |
|---|---:|---:|---:|---:|
| pooled_mlp | 0.000352293 | [-5.70016e-05, 0.000761587] | 0.125 | 0.5 |
| source_balanced_pooled_mlp | -0.0117783 | [-0.0125858, -0.0109709] | 0.0625 | 0.5 |
| source_conditioned_decoder | -0.000921813 | [-0.00157879, -0.000264839] | 0.0625 | 0.5 |
| per_source_models | -0.00519358 | [-0.00716278, -0.00322438] | 0.0625 | 0.5 |
| oracle_u_aware | -0.00343464 | [-0.00473436, -0.00213493] | 0.0625 | 0.5 |
| source_shuffle | -7.54302e-05 | [-0.000490139, 0.000339278] | 0.625 | 1 |
| no_behavior | 0.000215688 | [-0.000356463, 0.000787839] | 0.5 | 1 |
| source_dependent_reward | -0.000785182 | [-0.00166273, 9.23609e-05] | 0.125 | 0.5 |

## Limitations

- Five seeds provide weak exact-test resolution; corrected p-values are descriptive safeguards, not the main evidence.
- A t interval with four degrees of freedom is sensitive to seed outliers.
- Anchor-level rows are correlated within seed and scenario and were not used to inflate inferential n.
- The AUC gives more numerical weight to wider high-dose λ intervals; per-dose curves must be inspected alongside it.
- This analysis was not preregistered; the primary setting was locked before inspecting the final numeric comparisons in this script.
