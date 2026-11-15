# Explicit nonresponse-indicator sensitivity analysis

## English Methods text

The original TT4G35A-C response codes were retained before recoding. For each
item, `5` (I don't know) was coded as a separate binary indicator, whereas 8
and 9 were retained as a distinct other-special-missing category. `DK_ANY`
indicates at least one code 5 and `DK_COUNT` counts the three code-5 responses.
`N_VALID` counts responses coded 1-4 and `NO_VALID` identifies respondents with
no valid utility item. The perceived-utility mean was unchanged: it used only
1-4 responses, and code 5 was never assigned a numeric scale value.

Models A0 and A1 were fitted in the same available-item sample (n=72,000),
with A1 adding `DK_ANY` to the baseline X, utility, and education-system fixed-
effect logistic model. Models B0 and B1 used the expanded X/Y/weight-valid
sample (n=84,770); B1 added `DK_ANY`, `NO_VALID`, and a centered observed-item
mean. The centering constant was the TCHWGT-weighted mean in the n=72,000
sample (mu=2.9020231226); for N_VALID=0, the centered mean was set to zero solely
as a mathematical placeholder and does not represent a neutral response.
All models were re-estimated with each of 100 TALIS replicate weights. Fay-BRR
standard errors used factor 0.5 and 95% intervals used estimate +/- 1.96 SE.
This is a descriptive missingness sensitivity analysis and does not identify
MAR or MNAR mechanisms.

## English Results text (generated from the result CSV)

A0 (n=72,000): ai_training beta=1.3870 (SE=0.0525); mediator beta=1.1592 (SE=0.0347).
A1 (n=72,000): ai_training beta=1.3526 (SE=0.0512); mediator beta=1.1733 (SE=0.0346); DK_ANY beta=-0.6129 (SE=0.1126).
B0 (n=84,770): ai_training beta=1.6655 (SE=0.0559).
B1 (n=84,770): ai_training beta=1.3467 (SE=0.0525); M_CENTER beta=1.1587 (SE=0.0344); DK_ANY beta=-0.6113 (SE=0.1108); NO_VALID beta=-1.7271 (SE=0.2976).
