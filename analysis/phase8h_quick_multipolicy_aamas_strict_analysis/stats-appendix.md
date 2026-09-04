# Phase 8H-Q 统计附录

## 统计单位与方法

- 主推断单位：模型初始化 seed（confounded: n=3；seeds 0, 1, 2）。
- 测试 anchors：每个 seed 固定 77 个；anchors 是 seed 内重复测量，未被伪装成 231 个独立实验。
- 主比较：action-level minimum 对 balanced pooled union，以及 action-level minimum 对 state-level minimum。
- 主指标：Do-Bellman MAE、top-action disagreement、mean decision regret，均越低越好。
- 检验：对 seed 差值枚举全部 2^3 个符号翻转，双侧精确 p；6 个主检验用 Holm 校正。
- 效应量：配对 Cohen dz；同时报告均值差、差值 SD、t 型 95% CI 和改善方向的 seed 数。
- n=3 无法可靠判断正态性；t 区间仅表达极大的估计不确定性，不作为确认性显著性依据。

## 完整配对结果

见 `paired-seed-contrasts.csv`。6 个主检验的 Holm-adjusted p 均不小于 1。方向一致的 Do-MAE 效应仍只能视为探索性结果。

## Anchor-level 分解

`anchor-level-decomposition.csv` 报告每个 seed 内 action-level 相对 state-level/pooled 的 anchor 差值、胜率及 envelope 结构。它用于机制解释，不用于把 anchors 当作独立实验进行显著性检验。

Figure 2 中 Spearman rho=0.103510, p=0.116671；该 p 值仅为描述性，因为 77 anchors 在每个模型 seed 内共享训练模型且跨 seed 使用相同测试状态。

## 完整性和局限

- Phase 8H 的 28 项 hard checks 全部通过。
- 输入文件在分析前后以 SHA-256 验证，分析只写入独立 `analysis/` 目录。
- `best_single_source_posthoc` 使用 do-oracle 逐 anchor 选择来源，仅是不可部署的诊断 ceiling。
- native 与 union candidate-set 指标的动作集合不同，不能将其差值纯粹解释为模型质量变化。
- 没有有限样本置信修正，underestimation 直接否定 certified-upper-bound 解读。
