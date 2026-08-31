# Figure catalog

## balanced_reward_bias_vs_lambda.png

- Purpose: Audit balanced observational-do bias.
- Observation: Plus/minus separate linearly while base remains at its physical baseline.
- Implication: Source balancing does not remove reward confounding.

## heavy_reward_drift_vs_lambda.png

- Purpose: Audit logger-mixture drift.
- Observation: The signed plus/minus drift changes with slopes +/-14/45.
- Implication: Mixture composition changes observational reward without changing do reward.

## physical_vs_direct_bias.png

- Purpose: Separate the two bias channels.
- Observation: Physical bias is lambda-invariant and direct bias is additive.
- Implication: The direct channel does not replace actuator-mediated bias.

## total_bias_decomposition.png

- Purpose: Verify total-bias additivity.
- Observation: Physical plus direct coincides with total bias.
- Implication: The two mechanisms are numerically identifiable in the oracle audit.

## confounded_vs_independent_reward_bias.png

- Purpose: Check the latent-independence control.
- Observation: Only the confounded condition grows with lambda.
- Implication: The direct bias requires association between action and U_env.

## plus_minus_base_bias_vs_lambda.png

- Purpose: Show the clean kappa-zero experiment.
- Observation: Slopes are +0.6, -0.6, and 0.
- Implication: At kappa zero the total bias is purely direct.

## population_signal_vs_previous_neural_error.png

- Purpose: Compare signal and old fit-error scales.
- Observation: The fixed lambda grid crosses the previous neural error scale.
- Implication: This comparison diagnoses scale only; it does not establish learnability.
