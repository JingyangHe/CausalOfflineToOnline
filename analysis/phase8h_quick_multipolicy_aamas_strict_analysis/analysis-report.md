# Phase 8H-Q 严格只读结果分析

## 结论与决策

本次科学结论是 **No-Go（不应直接升级为完整多策略 AAMAS 或在线 SAC）**。实现本身有效：28 项硬检查全部通过，512 个固定 anchors、24 个 best-only 模型和 41 个结果文件均完整，输入哈希未改变。

action-level minimum 确实降低了数值拟合误差：相对 balanced pooled AAMAS，Do-Bellman MAE 从 3.183 降至 2.372（25.5%），potential MAE 从 4.358 降至 3.040（30.2%）。但是动作排序仍几乎失效：top-action disagreement 为 90.0%，mean regret 只从 0.839 降至 0.808（3.7%）。

更关键的是，action-level 与 state-level 的 mean regret 几乎完全相同（0.808379 对 0.808369），说明代数上的 action-wise tightening 没有转化为可靠决策收益。三 seed 配对检验中，6 个主比较经 Holm 校正后均未提供确认性证据；n=3 时精确双侧符号翻转检验的最小 p 值本来也只能达到 0.25。

因此当前证据支持的窄结论是：**逐动作跨 source 取最小值能改善近似 Q 的绝对数值，但没有解决真正关键的 action ranking；观察到的收益也不能归因于隐藏混杂恢复。**

## 主要结果

所有数值均为 confounded 条件下三个模型 seed 的均值；越低越好。

| 方法 | Do-MAE | 排序错误率 | Mean regret | Potential MAE |
|---|---:|---:|---:|---:|
| Source 1 | 3.653 ± 0.179 | 0.909 ± 0.022 | 0.892 ± 0.035 | 4.856 ± 0.570 |
| Source 2 | 3.249 ± 0.556 | 0.909 ± 0.000 | 0.865 ± 0.157 | 4.821 ± 1.160 |
| Source 3 | 3.657 ± 0.648 | 0.909 ± 0.034 | 0.771 ± 0.091 | 5.371 ± 1.716 |
| Action-level min | 2.372 ± 0.207 | 0.900 ± 0.030 | 0.808 ± 0.093 | 3.040 ± 0.489 |
| State-level min | 2.492 ± 0.207 | 0.905 ± 0.030 | 0.808 ± 0.109 | 3.181 ± 0.585 |
| Pooled balanced | 3.183 ± 0.293 | 0.909 ± 0.022 | 0.839 ± 0.142 | 4.358 ± 0.152 |

### 预先锁定的配对比较

差值均为 action-level min 减 baseline；负值有利于 action-level。区间为以模型 seed 为单位的 t 型 95% CI，仅用于展示 n=3 的不确定性。

| 比较 | Do-MAE 差 [95% CI] | 排序错误差 [95% CI] | Regret 差 [95% CI] |
|---|---:|---:|---:|
| Action min − pooled | -0.8110 [-1.5699, -0.0520] | -0.0087 [-0.0758, +0.0585] | -0.0310 [-0.6138, +0.5518] |
| Action min − state min | -0.1200 [-0.2311, -0.0088] | -0.0043 [-0.0788, +0.0702] | +0.0000 [-0.0561, +0.0561] |

- 对 pooled 的 Do-MAE 改善在 3/3 seeds 方向一致，但对 ranking 和 regret 并不稳定；seed 1 的 regret 反而更差。
- 对 state-level 的 Do-MAE 改善为 3/3 seeds 一致；ranking 只有 1/3 seed 改善，regret 一好、一坏、一次近乎相同。
- action-level 的 Do-MAE 优于三个单来源，但 regret 并未优于最好的固定单来源：Source 3 的平均 regret 为 0.771，低于 action-level 的 0.808。

## 为什么数值变准却没有选对动作

三个 source 在同一状态内切换最小来源的比例平均为 71.9%（SD 17.1%），所以 action-wise envelope 并非退化为某一个固定 source。state-level 与 action-level potential 的平均代数差为 0.166。然而该 tightening 与 regret 改善的 anchor-level Spearman 相关仅为 0.104（描述性 p=0.117；anchors 在 seed 内聚类，不能当作独立重复）。这说明“更低的近似上包络”与“更正确的动作次序”是两件不同的事。

oracle-only 的逐 anchor 最佳来源诊断可把 mean regret 降到 0.415，比 action-level min 低 48.7%。这表示 source 模型集合中仍包含可用的排序信号，但当前逐动作取最小规则没有正确识别应信任的 source。该 oracle 诊断不可部署，也不能作为正式方法成绩。

此外，action-level 输出低估 do-oracle 的比例为 36.1%（pooled 为 15.4%）。由于本实验没有 finite-sample confidence correction，这些输出只能称为 approximate upper backups，不能解释为覆盖有保证的上界。

## 负对照与因果解释边界

independent-latents 只有 seed 0，不能做推断统计，但 action-level 相对 pooled 的改善仍然存在：Do-MAE 改善 19.1%，ranking error 改善 4.2%，regret 改善 30.1%。收益在不存在 U_behavior–U_environment 相关性的对照中保留，因此目前更符合普通 ensemble/coverage 效应，而不是恢复 hidden confounding 的特有信息。

pooled 模型对来源配比也很敏感：不同配比之间预测 MAE 最大为 0.800，top-action disagreement 最大为 61.0%。但这种明显预测漂移没有形成稳定的 oracle 改善，不能作为多来源机制成功的证据。

## 下一步决策

不建议继续扩大 anchors、updates 或直接接 online SAC。最有信息量的下一步是一个更小的机制诊断：保持同一 union candidate set，训练一个只使用 public validation 信号的 source reliability/gating 规则，并与 action-wise hard minimum、state-wise minimum、固定 Source 3 比较。只有当 gating 在至少 5 个模型 seeds 上稳定降低 ranking error 和 regret，且 independent-latents 对照不保留同等收益时，才值得推进完整训练。

## 边界

这是一步 frozen-reference Bellman backup 的 quick gate，不是完整 Bellman fixed point，也没有短预算或长期 online return。confounded 主分析只有 3 个模型 seeds；independent-latents 只有 1 个 seed。所有统计结论都受该样本量限制。
