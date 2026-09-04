# Phase 8H-Q 图表目录

## figure-01-confounded-method-comparison.pdf

- Purpose：在完全相同的 union candidate set 上比较三单来源、action/state envelope 与 balanced pooled AAMAS。
- Data：`seed_metrics.csv`，confounded seeds 0/1/2；点为独立模型 seed，黑线为均值。
- Notice：action-level 明显降低 Do-MAE，但 ranking error 仍约 90%，regret 优势很小且跨 seed 不稳定。
- Implication：数值拟合改善不足以支持方法升级，决策指标应作为 gate。

## figure-02-tightening-versus-regret.pdf

- Purpose：检查代数 tightening 是否在同一个 anchor 上转化为 regret 改善。
- Data：三个 confounded seeds × 77 anchors；横轴为 state phi − action phi，纵轴为 action regret − state regret。
- Notice：大量点虽有正 tightening，但 regret 差围绕零且缺少有用单调关系。
- Implication：hard minimum 的几何性质没有自动带来动作排序收益。
- Caveat：点在 seed 内聚类，相关系数仅为描述性。

## figure-03-negative-control-retention.pdf

- Purpose：判断 action-level 对 pooled 的收益是否特属于 confounding 条件。
- Data：confounded n=3 seeds；independent-latents n=1 seed。纵轴为相对 pooled 的百分比改善。
- Notice：独立潜变量对照仍保留明显收益。
- Implication：当前收益更像普通多模型覆盖/集成效应，不能归因为 hidden-confounding recovery。
- Caveat：negative control 只有一个 seed，不能作确认性比较。
