# Phase 8B-RS — Direct U-to-Reward Signal Calibration

This positive-control stage added `lambda_reward * u_env` to copied rewards.  It
did not replace the physical confounding DGP.  All population identities were
verified before neural training, and all read-only input hashes were unchanged.

The neural audit used only `('observation', 'commanded_action')` (15D), the
fixed width-256 reward architecture, 3
model seeds, and held-out anchor evaluation.  Best-validation checkpoints were
used for reported predictions; final checkpoints were also retained.

## Interpretation logic

Case A: correct signed slopes, near-zero base/independent slopes, and paired
increment errors below absolute fit errors indicate that stronger direct reward
signal is learnable and that the original failure was substantially a
signal-to-approximation problem.

Case B: improved absolute fit with incorrect slopes or failed negative controls
indicates persistent cross-action interference or state generalization error.

Case C: failure to recover the lambda=0.20 slopes indicates failure even on this
strong positive control.

No effect-size threshold or automatic scientific verdict is applied.  The saved
metrics require manual review.
