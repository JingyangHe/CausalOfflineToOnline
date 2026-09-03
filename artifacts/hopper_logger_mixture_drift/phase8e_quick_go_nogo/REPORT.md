# Phase 8E-Q Quick Go/No-Go Report

This is an exploratory quick gate in the controlled three-action, one-step setting.

## Table 1 — Primary metrics (lambda=0.05)

| Setting | Method | B | Do MAE | Rank error | Regret |
|---|---|---:|---:|---:|---:|
| M2_diverse | MSCSC_correct_source | 0 | 0.0338658 +/- 0.00339 | 0.917749 +/- 0.0198 | 0.00560057 +/- 9.89e-05 |
| M2_diverse | MSCSC_correct_source | 16 | 0.0435086 +/- 0.00557 | 0.587879 +/- 0.118 | 0.00242702 +/- 0.00101 |
| M2_diverse | MSCSC_correct_source | 64 | 0.035319 +/- 0.00383 | 0.387879 +/- 0.154 | 0.00128082 +/- 0.000534 |
| M2_diverse | MSCSC_source_shuffle | 0 | 0.0337316 +/- 0.00332 | 0.909091 +/- 0.0225 | 0.00555196 +/- 0.000202 |
| M2_diverse | MSCSC_source_shuffle | 16 | 0.0399322 +/- 0.00339 | 0.620779 +/- 0.123 | 0.00235801 +/- 0.000959 |
| M2_diverse | MSCSC_source_shuffle | 64 | 0.0354529 +/- 0.0039 | 0.390476 +/- 0.151 | 0.0012566 +/- 0.000512 |
| M2_diverse | pooled_rank0 | 0 | 0.0336764 +/- 0.00266 | 0.91342 +/- 0.0075 | 0.00558118 +/- 2.4e-05 |
| M2_diverse | pooled_rank0 | 16 | 0.0398121 +/- 0.00231 | 0.604329 +/- 0.158 | 0.00229394 +/- 0.00101 |
| M2_diverse | pooled_rank0 | 64 | 0.0349366 +/- 0.00224 | 0.392208 +/- 0.189 | 0.00133161 +/- 0.000618 |
| M5_diverse | MSCSC_correct_source | 0 | 0.0363744 +/- 0.00198 | 0.904762 +/- 0.027 | 0.00545795 +/- 0.000181 |
| M5_diverse | MSCSC_correct_source | 16 | 0.0442228 +/- 0.00339 | 0.561905 +/- 0.143 | 0.00229862 +/- 0.000985 |
| M5_diverse | MSCSC_correct_source | 64 | 0.0381774 +/- 0.00551 | 0.390476 +/- 0.201 | 0.00134113 +/- 0.000671 |
| M5_diverse | MSCSC_source_shuffle | 0 | 0.035854 +/- 0.00169 | 0.904762 +/- 0.027 | 0.0054841 +/- 0.000197 |
| M5_diverse | MSCSC_source_shuffle | 16 | 0.0440314 +/- 0.000576 | 0.525541 +/- 0.164 | 0.00211408 +/- 0.00104 |
| M5_diverse | MSCSC_source_shuffle | 64 | 0.0370147 +/- 0.002 | 0.384416 +/- 0.103 | 0.00133708 +/- 0.00031 |
| M5_diverse | pooled_rank0 | 0 | 0.0346427 +/- 0.00243 | 0.904762 +/- 0.0075 | 0.00547012 +/- 0.000138 |
| M5_diverse | pooled_rank0 | 16 | 0.0400922 +/- 0.00373 | 0.578355 +/- 0.167 | 0.00230065 +/- 0.00107 |
| M5_diverse | pooled_rank0 | 64 | 0.0348671 +/- 0.00307 | 0.402597 +/- 0.198 | 0.00134527 +/- 0.000578 |
| M5_redundant | MSCSC_correct_source | 0 | 0.0332243 +/- 0.00245 | 0.909091 +/- 0.013 | 0.00556501 +/- 9.07e-05 |
| M5_redundant | MSCSC_correct_source | 16 | 0.0430294 +/- 0.00289 | 0.608658 +/- 0.145 | 0.00234561 +/- 0.000947 |
| M5_redundant | MSCSC_correct_source | 64 | 0.0350775 +/- 0.00471 | 0.4329 +/- 0.163 | 0.0014196 +/- 0.000502 |
| M5_redundant | MSCSC_source_shuffle | 0 | 0.0338145 +/- 0.00285 | 0.922078 +/- 0.013 | 0.00564471 +/- 0.000131 |
| M5_redundant | MSCSC_source_shuffle | 16 | 0.0410866 +/- 0.00632 | 0.625108 +/- 0.201 | 0.00247784 +/- 0.00123 |
| M5_redundant | MSCSC_source_shuffle | 64 | 0.0343218 +/- 0.00419 | 0.435498 +/- 0.156 | 0.00146543 +/- 0.000579 |
| M5_redundant | pooled_rank0 | 0 | 0.0338032 +/- 0.00375 | 0.917749 +/- 0.027 | 0.00557962 +/- 0.000106 |
| M5_redundant | pooled_rank0 | 16 | 0.0396308 +/- 0.00341 | 0.603463 +/- 0.157 | 0.00229538 +/- 0.001 |
| M5_redundant | pooled_rank0 | 64 | 0.0344354 +/- 0.00419 | 0.41039 +/- 0.214 | 0.0013856 +/- 0.000634 |

## Table 2 — Go/no-go contrasts

| Comparison | Do MAE difference | Rank difference | Regret difference |
|---|---:|---:|---:|
| Correct - Shuffle, M=5 diverse | 0.00116267 +/- 0.00402 | 0.00606061 +/- 0.0994 | 4.0525e-06 +/- 0.00037 |
| M=5 diverse - M=2 diverse | 0.00285836 +/- 0.00168 | 0.0025974 +/- 0.0481 | 6.03091e-05 +/- 0.000137 |
| M=5 redundant - M=5 diverse | -0.00309989 +/- 0.000869 | 0.0424242 +/- 0.0692 | 7.84693e-05 +/- 0.000209 |
| B=64 - B=0 | 0.00180295 +/- 0.00391 | -0.514286 +/- 0.225 | -0.00411682 +/- 0.000757 |

Paired seed-direction consistency (fraction of three seeds in the reported mean direction):

- Correct - Shuffle, M=5 diverse: 0.667, 0.333, 0.333
- M=5 diverse - M=2 diverse: 1, 0.333, 0.667
- M=5 redundant - M=5 diverse: 1, 0.667, 0.667
- B=64 - B=0: 0.667, 1, 1

## Direct answers

- correct_source_beats_shuffle: not supported
- M5_diverse_beats_M2_diverse: not supported
- M5_redundant_has_no_equal_benefit: not supported
- B0_to_B16_to_B64_improves_all_three_metrics: not supported
- lambda_zero_safely_returns_to_rank0: not supported
- continuous_action_extension_recommended: not supported

No confirmatory significance claim is made from three model seeds.