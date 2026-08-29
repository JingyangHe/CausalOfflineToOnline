# Phase 8A — Hopper Logger-Mixture Causal Drift DGP

This increment constructs fixed Hopper simulator anchors, three controlled diagnostic loggers,
two completely enumerated latent conditions, four logger mixtures implemented only through sample
weights, and a mixture-independent two-point `do(action)` oracle.

It trains no policy or world model and does not use AAMAS, rho, Joint LP, or Relaxed LP. Public NPZ
files contain only the 12D public state, commanded action, outcome, logger ID, anchor ID, row ID, and
kappa. Hidden latents, applied actions, simulator state, and action keys remain in separate audit files.

The recommended totals 512 and 2048 are not divisible by three. To preserve the requested total,
anchor origin quotas are deterministic and differ by at most one; exact equality occurs whenever
`--num-anchors` is divisible by three. This discrepancy is recorded rather than hidden.

Smoke:

```bash
python scripts/run_hopper_logger_mixture_drift.py \
  --num-anchors 24 --kappas 0.0 0.2 --behavior-offset 0.2 --seed 0 \
  --output-root artifacts/hopper_logger_mixture_drift/smoke
```

Server pilot:

```bash
python scripts/run_hopper_logger_mixture_drift.py \
  --num-anchors 512 --kappas 0.0 0.1 0.2 0.3 --behavior-offset 0.2 --seed 0 \
  --output-root artifacts/hopper_logger_mixture_drift/controlled_loggers_seed0
```
