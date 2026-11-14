# B01 Model specification and variance estimation

Point estimates use the TALIS final teacher weight (`TCHWGT`) and fixed effects for all education systems (`C(CNTRY)`). The a path is a weighted linear model. The b, c and c-prime paths are weighted logistic models reported on the log-odds scale and, where appropriate, as odds ratios.

Design-based standard errors use `TRWGT1` through `TRWGT100` with Fay balanced repeated replication (Fay factor 0.5): `sqrt(sum((theta_r-theta)^2)/(R*(1-f)^2))`. Replicate estimates and failures are retained as public model-level audit outputs.
