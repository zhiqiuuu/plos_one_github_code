# PLOS ONE revision-analysis materials

This repository contains the code, non-identifying aggregate data, figures, and audit outputs for the revision of the TALIS 2024 analysis. The restricted TALIS teacher microdata and row-level linkage keys are **not** included and must be obtained through the authorised OECD route.

## Scope of the final package

- Final primary results report design-adjusted association paths `a`, `b`, `c`, and `c'` only.
- No coefficient-product, proportion-mediated, or probability-scale indirect effect is presented as a final result.
- `results/output_final/B02/association_only_decision.md` records the remaining author decision required before any quantitative mediation claim is restored.

## Reproduce

1. Install the pinned packages in `environment/requirements.lock`.
2. Run `python code/run_all.py --raw-file <path-to-ttgintt4.csv>`. The entry point prepares the derived analysis file, runs the supplementary analyses, and renders the figure.

The audit-ready outputs are organised as `results/output_final/B01` through `results/output_final/B06`. Raw data, respondent identifiers, and any absolute local paths are excluded.

`data/derived/public_system_summary.csv` is a non-identifying, education-system-level aggregate used to document the public release; the row-level working file created during local reproduction is ignored by Git.
