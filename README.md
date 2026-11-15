# PLOS ONE revision-analysis code and public outputs

This public release contains executable analysis scripts, tests, dependency information, non-identifying aggregate result tables, figures, and a reproducibility audit for the TALIS 2024 revision. It deliberately excludes the restricted TALIS teacher microdata, any row-level derived file, teacher or school identifiers, and row-level linkage keys. Authorised users must obtain the original TALIS teacher data through the OECD route.

## Scope of the final package

- Final primary results report design-adjusted association paths `a`, `b`, `c`, and `c'` only.
- No coefficient-product, proportion-mediated, or probability-scale indirect effect is presented as a final result.
- `results/output_final/B02/association_only_decision.md` records the remaining author decision required before any quantitative mediation claim is restored.
- The supplementary KHB-style output is a logistic-coefficient statistical decomposition, not a causal mediation estimate or an ACME.
- The Q35 code-5 analysis is a descriptive nonresponse sensitivity analysis; it does not establish MAR or MNAR.

## Reproduce

1. Create the documented environment with the pinned packages in `environment/requirements.lock`.
2. Run `python code/run_all.py --raw-file <path-to-ttgintt4.csv> --run-dir results/release_v0.3.1`. The entry point prepares the derived analysis file, runs the primary, KHB-style, and explicit Q35 nonresponse analyses, and renders the figure. A full run performs 100 Fay-BRR replicates for the primary analysis, item-completeness analyses, each of 20 multiply imputed datasets, both KHB samples, and all four nonresponse-indicator models.
3. Inspect `<run-dir>/output/B01/reproduction.log`, `<run-dir>/output/B06/run_metadata.json`, and `<run-dir>/output/B05/release_manifest.csv`. The metadata records the code revision, software environment, external raw-file checksum, sample counts, and run mode without publishing the restricted data. The temporary `<run-dir>/restricted_working` directory is ignored by Git.

The public aggregate outputs are in `results/output_final/B01` through `B06`, with Figure 1 in `figures/`. `B02/khb_*` contains the coefficient decomposition, replicate estimates, identity checks, and methods. `B03/nonresponse_indicator_*` contains raw-code diagnostics, 100-replicate model results, design-matrix checks, and methods. A fresh local run writes the same structure below `<run-dir>/output`. Raw data, respondent identifiers, and any absolute local paths are excluded from this release.

## Inference labels

- `TableS1`: Fay-BRR 95% confidence intervals.
- `TableS3`: Fay-BRR 95% confidence intervals for item-completeness analyses; MNAR delta scenarios are point estimates only.
- `TableS4`: paths involving the imputed mediator combine within-imputation Fay-BRR variance with between-imputation variance using Rubin's rules; c does not depend on the mediator and therefore has its direct Fay-BRR interval.
- `TableS5`: KHB-style coefficient decomposition with Fay-BRR 95% confidence intervals; it is not a causal indirect-effect table.
- `TableS6`: explicit Q35 code-5 nonresponse indicator sensitivity models with Fay-BRR 95% confidence intervals.
- `TableS0`: `TCHWGT`-weighted means, population SDs, skewness, and excess kurtosis for the exposure, outcome, TT4G35A-C items, and available-item composite; descriptive point estimates only, with no confidence intervals.
- `TableS2`: specification-comparison point estimates only, with no design-based confidence intervals.

## Data and code availability

The analysis reuses the OECD TALIS 2024 teacher microdata. Those teacher-level records are third-party restricted and are not redistributed here. Eligible users must obtain the source file through the OECD TALIS access route and run the scripts locally. This repository provides the executable code, locked dependency list, aggregate non-identifying tables, figure, provenance hashes, and run metadata needed to audit the reported calculations. It does not grant a licence to redistribute the OECD microdata. The public release has no DOI until an authorised maintainer creates a GitHub Release and archives it through Zenodo.

## Version and DOI

The prepared package version is recorded in `VERSION`. The repository has no permanent DOI until an authorised maintainer creates a GitHub Release and archives it through Zenodo; see `ARCHIVING.md`.
