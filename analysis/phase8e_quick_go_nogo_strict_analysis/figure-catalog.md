# Phase 8E-Q 图目录

所有图同时提供 PDF（论文/缩放优先）和 PNG（快速预览）。误差棒若出现，表示以 3 个模型 seed 为单位的配对均值 t 型 95% 区间；点表示单个 seed，不把 5 个嵌套 calibration replicate 当成独立样本。

## Figure 01 — Do-MAE paired contrasts

- 文件：[PDF](figures/figure-01-do_mae-paired-contrasts.pdf)；[PNG](figures/figure-01-do_mae-paired-contrasts.png)
- 问题：正确 source、多来源、多样性和校准预算是否改善 Do-oracle MAE？
- 主要信息：四个主对比中没有一个显示目标方法稳定降低 Do-MAE；M5 对 M2 为 3/3 seed 更差，B64 对 B0 的均值也略差。
- 限制：n=3，区间很宽；图用于 quick gate，不用于确认性推断。

## Figure 02 — Ranking-error paired contrasts

- 文件：[PDF](figures/figure-02-top_set_disagreement-paired-contrasts.pdf)；[PNG](figures/figure-02-top_set_disagreement-paired-contrasts.png)
- 问题：哪些设计因素改善最优动作集合判断？
- 主要信息：只有 B64−B0 显示三个 seed 同方向的大幅下降；correct-vs-shuffle 和 M5-vs-M2 都接近零且方向混合。
- 限制：测试集为固定 77 anchors，seed 间差异明显。

## Figure 03 — One-step-regret paired contrasts

- 文件：[PDF](figures/figure-03-mean_regret-paired-contrasts.pdf)；[PNG](figures/figure-03-mean_regret-paired-contrasts.png)
- 问题：哪些因素真正降低一步动作选择损失？
- 主要信息：B64−B0 在 3/3 seed 上降低 regret；多来源相关对比没有稳定收益。
- 限制：这里只是一步 regret，不是 Hopper 长期 return。

## Figure 04 — Calibration budget relative to B=0

- 文件：[PDF](figures/figure-04-calibration-budget-relative.pdf)；[PNG](figures/figure-04-calibration-budget-relative.png)
- 问题：增加干预校准数据时，点预测和决策指标是否同步改善？
- 编码：每个指标归一化为自身 B=0 均值的 100%。
- 主要信息：排序错误和 regret 随预算明显下降；Do-MAE 在 B16 恶化，B64 仍略高于 B0。
- 限制：归一化图展示相对变化，绝对值见统计附录。

## Figure 05 — BIC rank-1 selection rate

- 文件：[PDF](figures/figure-05-rank1-selection-rate.pdf)；[PNG](figures/figure-05-rank1-selection-rate.png)
- 问题：rank-1 选择能否区分真实 source contrast、shuffle 和空信号？
- 主要信息：阳性 correct 与阳性 shuffle 的选择率完全相同，且阳性 correct 不高于 λ=0 correct；选择器缺少阳性/阴性分离能力。
- 限制：每个点是 15 次嵌套选择的比例，但真正独立的模型 seed 仍只有 3 个。

## Figure 06 — Population signal versus empirical noise

- 文件：[PDF](figures/figure-06-population-signal-vs-empirical-noise.pdf)；[PNG](figures/figure-06-population-signal-vs-empirical-noise.png)
- 问题：No-Go 来自 DGP 无信号，还是有限样本恢复失败？
- 编码：总体 rank-1 信号、正确 source 经验矩阵的 off-rank 残差和 shuffle source 的 centered norm 均以训练 anchors 上的 L2 norm 表示。
- 主要信息：M5 diverse 的总体信号非零，但经验 off-rank 残差与 shuffle 噪声同量级；M5 redundant 的总体信号为零而经验噪声依然很大。
- 限制：M2 的 off-rank 残差代数上必为零，不能直接与 M5 作公平的恢复难度比较。

