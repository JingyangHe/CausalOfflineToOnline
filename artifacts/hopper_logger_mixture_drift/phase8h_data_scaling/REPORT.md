# Phase 8H-DS Source-Wise Data Scaling Diagnostic

The independent statistical unit is model seed (n=3); anchors are repeated measurements.

## 1. Does Action-min Do-MAE continually decrease?

Values for n=[16, 32, 64, 128]: [2.870274293024725, 2.3723345837920093, 1.926756305340402, 1.6824328543405374]. Monotone decrease: True.

## 2. Does the roughly 36% underestimation decrease?

Values: [0.5756029684601113, 0.3611626468769326, 0.3106060606060606, 0.2898886827458256]. Monotone decrease: True.

## 3. Do mean, median, P90, and CVaR90 regret improve?

regret_mean: [0.7587967805527223, 0.8083792774678411, 0.7486063563090717, 0.5661327617942313] (monotone=False); regret_median: [0.5640030675373092, 0.5133811153639461, 0.4933653492431442, 0.4053753706430901] (monotone=True); regret_p90: [1.5525779737147019, 1.6904379466010189, 1.600852577848918, 1.2616019821941566] (monotone=False); regret_cvar90: [2.7391801839012735, 3.3335750277946112, 2.691691701838368, 1.822736745007732] (monotone=False).

## 4. Does Action-min stably beat fixed Source 3?

ActionMin−Source3 mean-regret gaps: [-0.022955380272021892, 0.03723865558525963, 0.04370378353773108, 0.020255978209237042]; negative favors Action-min.

## 5. Does the advantage over pooled grow with data?

Pooled−ActionMin mean-regret gains: [0.024234919821776946, 0.03102838666657881, -0.05870859185575691, 0.036578833045453285].

## 6. What is the dominant bottleneck?

Seed-0 mean regret is 0.728862 at n32, 0.564067 for n32 with 4000 updates, and 0.620678 at n128. Interpretation must separate finite-sample variance, optimization budget, and hard-min decision coherence. No automatic success threshold is used.
Independent-latents is a one-seed secondary control: [('n128', 0.056279580044119815), ('n32', -0.29273654579338737)]. Any similar scaling benefit there indicates generic coverage/ensemble/variance reduction, not identification of hidden U.
