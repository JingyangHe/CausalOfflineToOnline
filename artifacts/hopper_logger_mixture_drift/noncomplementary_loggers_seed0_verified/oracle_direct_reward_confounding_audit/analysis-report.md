# Analysis report

- Q1: Yes. The confounded observational-do reward bias grows exactly linearly with lambda.
- Q2: Yes. P(S,A) is unchanged; the added bias is exactly lambda E[U_env|S,A].
- Q3: Yes. The balanced source mixture retains slopes -0.6 and +0.6.
- Q4: No. The direct term cancels under the symmetric do(U_env) average.
- Q5: Yes. Independent latents remove the direct and total observational-do bias.
- Q6: Yes. The base action has zero direct bias for every anchor and lambda.
- Q7: Yes. Total bias equals physical bias plus direct U-to-reward bias within tolerance.

The exact numeric evidence is saved in the CSV and NPZ artifacts.
