# KHB statistical decomposition

## Analysis specification

The KHB-style decomposition used the taskbook-specified primary available-item
sample (n = 72,000) and strict complete-case sample (n = 60,447) separately.
In each sample, the reduced weighted logistic model was AI use on AI-related
professional learning and education-system fixed effects. The perceived-utility
composite was projected by weighted linear regression on the same covariates.
Its respondent-level residual was included in a residualized logistic model.
The full logistic model included the original perceived-utility composite. The
KHB statistical indirect component was beta_total_adjusted minus beta_direct,
and the rescaling component was beta_total_adjusted minus beta_total_raw.

Point estimates use TCHWGT. For each of 100 Fay-BRR replicate weights, the
mediator projection, residual, all three logistic models, and both differences
were re-estimated. Standard errors use
sqrt(sum((theta_r-theta)^2)/(100*(1-0.5)^2)); 95% intervals are estimate plus
or minus 1.96 standard errors. Coefficients are on the logistic log-odds scale.
This is a KHB statistical decomposition following Karlson, Holm, and Breen
(2012), not a causal mediation analysis: the cross-sectional design does not
establish temporal ordering or rule out unmeasured confounding.

## Manuscript-ready English Methods

For a supplementary non-linear decomposition, we applied a KHB-style
residualization procedure separately in the available-item and complete-case
samples. We estimated a weighted reduced logistic model for AI use, a weighted
linear projection of perceived utility on AI-related professional learning and
education-system fixed effects, a weighted full logistic model including
perceived utility, and a residualized logistic model including the projection
residual. The KHB indirect and rescaling components were defined as differences
between the residualized, full, and reduced professional-learning coefficients.
Point estimates used TALIS final teacher weights; all steps were repeated for
the 100 Fay-BRR weights (Fay factor 0.5). Results are reported on the log-odds
scale and are interpreted as cross-sectional statistical decompositions rather
than causal mediation effects.

## Manuscript-ready English Results

In the primary available-item sample, the KHB statistical indirect component
was 0.321 log odds (Fay-BRR SE
0.025, 95% CI
0.272 to 0.371);
the corresponding direct component was 1.387 log
odds (95% CI 1.284 to
1.490). In the strict complete-case analysis,
the KHB indirect component was 0.365 log odds
(95% CI 0.310 to
0.420). These estimates are statistical
decomposition components and are not reported as causal indirect effects.

## Implementation validation

khb_identity_checks.csv verifies both decomposition identities for each point
estimate and all 200 replicate analyses. Automated tests use a synthetic,
fixed-effect data set to verify weighted residual orthogonality to the
projection design and the two difference identities. This is an equivalence
check of the documented KHB residualization algorithm; it does not claim
validation against an unavailable external microdata implementation.
