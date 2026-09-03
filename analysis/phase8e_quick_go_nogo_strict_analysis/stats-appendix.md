# Phase 8E-Q 统计附录

## 完整性与样本层级

- 原始目录：`artifacts/hopper_logger_mixture_drift/phase8e_quick_go_nogo`
- 原始文件：120 个，共 17,847,187 bytes（17.02 MiB）
- 场景：10；模型：90；原始硬检查：23/23 通过
- anchors：512；train/observational-validation/do-calibration/test = 333/51/51/77
- 独立分析单位：3 个模型 seed
- 嵌套重复：每个 seed 5 次 calibration replicate，先在 seed 内求均值
- 最终测试指标均基于固定的 77 个 test anchors

轻量存储目标达成：实际只有 120 个文件和 17.02 MiB，低于 preflight 的 48.55 MiB 估计，没有出现原 formal 设计中的小文件爆炸。

## 推断方法

对每个配对比较，令三个模型 seed 的差为 \(d_1,d_2,d_3\)。报告 \(\bar d\)、样本标准差、自由度 2 的 t 型 95% 区间、配对标准化效应 \(d_z=\bar d/s_d\)，以及枚举八种符号翻转后以 \(|\bar d|\) 为统计量的精确双侧 p 值。12 个主比较 p 值用 Holm 法共同校正。n=3 时双侧精确 p 值最小为 0.25，所以这些结果只能作描述性和机制性判断。

所有误差类指标越低越好，因此左减右为负代表左侧更好。

## 主配对比较

| 比较 | 指标 | 均值差 | 95% CI | dz | 精确 p | Holm p | 负/正 seed |
|---|---|---:|---:|---:|---:|---:|---:|
| Correct−Shuffle | Do-MAE | 0.001163 | [−0.008820, 0.011145] | 0.289 | 0.75 | 1.00 | 1/2 |
| Correct−Shuffle | 排序错误 | 0.006061 | [−0.240932, 0.253053] | 0.061 | 1.00 | 1.00 | 2/1 |
| Correct−Shuffle | Regret | 0.000004 | [−0.000914, 0.000923] | 0.011 | 1.00 | 1.00 | 2/1 |
| M5 diverse−M2 diverse | Do-MAE | 0.002858 | [−0.001316, 0.007033] | 1.701 | 0.25 | 1.00 | 0/3 |
| M5 diverse−M2 diverse | 排序错误 | 0.002597 | [−0.116901, 0.122096] | 0.054 | 1.00 | 1.00 | 1/1，另 1 个为零 |
| M5 diverse−M2 diverse | Regret | 0.000060 | [−0.000280, 0.000400] | 0.441 | 0.75 | 1.00 | 1/2 |
| M5 redundant−M5 diverse | Do-MAE | −0.003100 | [−0.005258, −0.000941] | −3.567 | 0.25 | 1.00 | 3/0 |
| M5 redundant−M5 diverse | 排序错误 | 0.042424 | [−0.129422, 0.214271] | 0.613 | 0.50 | 1.00 | 1/2 |
| M5 redundant−M5 diverse | Regret | 0.000078 | [−0.000440, 0.000597] | 0.376 | 0.75 | 1.00 | 1/2 |
| B64−B0 | Do-MAE | 0.001803 | [−0.007912, 0.011518] | 0.461 | 0.75 | 1.00 | 1/2 |
| B64−B0 | 排序错误 | −0.514286 | [−1.074077, 0.045505] | −2.282 | 0.25 | 1.00 | 3/0 |
| B64−B0 | Regret | −0.004117 | [−0.005998, −0.002236] | −5.437 | 0.25 | 1.00 | 3/0 |

虽然若只看 t 型区间，个别项目不跨零，但 n=3 下该区间高度依赖正态假设；精确检验和多重比较校正均不支持确认性显著性结论。方向一致性和实际效应大小用于 quick gate 决策更合适。

## 校准预算分解

| 比较 | 指标 | 均值差 | 相对变化 | 95% CI | 精确 p | 负/正 seed |
|---|---|---:|---:|---:|---:|---:|
| B16−B0 | Do-MAE | 0.007848 | +21.58% | [0.003513, 0.012183] | 0.25 | 0/3 |
| B16−B0 | 排序错误 | −0.342857 | −37.89% | [−0.755351, 0.069636] | 0.25 | 3/0 |
| B16−B0 | Regret | −0.003159 | −57.88% | [−0.005730, −0.000589] | 0.25 | 3/0 |
| B64−B16 | Do-MAE | −0.006045 | −13.67% | [−0.016849, 0.004758] | 0.25 | 3/0 |
| B64−B16 | 排序错误 | −0.171429 | −30.51% | [−0.323161, −0.019696] | 0.25 | 3/0 |
| B64−B16 | Regret | −0.000957 | −41.65% | [−0.001831, −0.000084] | 0.25 | 3/0 |
| B64−B0 | Do-MAE | 0.001803 | +4.96% | [−0.007912, 0.011518] | 0.75 | 1/2 |
| B64−B0 | 排序错误 | −0.514286 | −56.84% | [−1.074077, 0.045505] | 0.25 | 3/0 |
| B64−B0 | Regret | −0.004117 | −75.43% | [−0.005998, −0.002236] | 0.25 | 3/0 |

## Adaptive 相对 pooled rank-0

在 M=5 diverse、λ=0.05、confounded、B=64：

| 方法减 pooled | Do-MAE 差 | 排序错误差 | Regret 差 |
|---|---:|---:|---:|
| Correct-source | +0.003310 | −0.012121 | −0.000004 |
| Source-shuffle | +0.002148 | −0.018182 | −0.000008 |

Correct-source 的 Do-MAE 在 3/3 seed 上比 pooled 更差。两个 adaptive 方法的排序和 regret 增量都很小，而且 shuffle 的点估计并不逊于 correct。这表明总体校准收益来自 rank-0 动作校准。

## 控制场景

λ=0、B=16 时 correct-source 相对 pooled 的 Do-MAE、排序错误和 regret 相对变化为 +11.19%、+17.24% 和 +8.50%；B=64 时为 +2.22%、−3.39% 和 −1.12%，均缺乏种子方向一致性。Independent-latents、B=64 时 correct-source 相对 shuffle 的对应变化为 +2.42%、+13.94% 和 +15.14%。

Rank-1 BIC 选择次数以 3 seed × 5 replicate = 15 次为分母：阳性 correct 为 3/15（B16）和 2/15（B64）；阳性 shuffle 同样为 3/15 和 2/15；λ=0 correct 为 4/15 和 3/15；independent correct 为 1/15 和 1/15。

## 机制与有限样本密度

| 设置/动作 | 总体领先奇异值 | 经验 rank-1 残差 | Correct centered norm | 残差占比 | Shuffle norm |
|---|---:|---:|---:|---:|---:|
| M2 diverse / minus | 0.5543 | ≈0 | 0.5548 | ≈0% | 0.1869 |
| M2 diverse / plus | 0.5543 | ≈0 | 0.5428 | ≈0% | 0.1837 |
| M5 diverse / minus | 0.6197 | 0.5080 | 0.8250 | 61.6% | 0.5957 |
| M5 diverse / plus | 0.6197 | 0.4837 | 0.7986 | 60.6% | 0.6092 |
| M5 redundant / minus | 0 | 0.5073 | 0.5932 | 85.5% | 0.6102 |
| Independent / minus | 0 | 0.5563 | 0.6581 | 84.5% | 0.6657 |

M2 的经验 rank-1 残差必为零，因为两个来源中心化后矩阵秩至多为一。它不能作为 M2 恢复真实结构更好的独立证据。

训练公共表只覆盖 train+observational-validation 的 384 个 anchors。取 333 个训练 anchors 后，M2 有 1,998 个 source×anchor×action 单元、每单元 21–22 条记录；M5 有 4,995 个单元、每单元 8–9 条记录。这个精确计数解释了固定总预算下 M5 的方差劣势。

## 可复现文件

数值由 `scripts/analyze_phase8e_quick_go_nogo_results.py` 从只读原始 CSV/NPZ 重算。派生表包括：

- `primary-paired-contrasts.csv`
- `adaptive-vs-pooled-contrasts.csv`
- `calibration-budget-levels.csv`
- `calibration-budget-contrasts.csv`
- `control-paired-contrasts.csv`
- `rank1-selection-rates.csv`
- `mechanism-subspace-summary.csv`
- `uncalibrated-rank1-summary.csv`
- `training-cell-counts.csv`
- `analysis-manifest.json`

