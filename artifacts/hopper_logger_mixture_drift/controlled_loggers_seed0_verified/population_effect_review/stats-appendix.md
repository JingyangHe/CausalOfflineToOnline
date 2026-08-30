# Statistical appendix

- Unit of analysis and bootstrap cluster: anchor_id.
- Analyzed anchors: 2048 of 2048, selected by sorted ID prefix only.
- Descriptives: mean, sample SD, median, P10, P25, P75, P90, maximum.
- Uncertainty: 95% percentile interval from 2000 paired anchor bootstraps, seed 0.
- Transition rows are repeated support cells, not independent replicates.
- No hypothesis test, p-value, or multiple-comparison claim is made because the DGP is fully
  enumerated at fixed anchors and only one behavior-policy training seed is available.
- The drift/action-gap ratio divides bootstrap aggregate means, never individual anchor gaps.
