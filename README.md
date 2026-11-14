# PLOS ONE revision-analysis code

This code-only release contains the executable analysis scripts, tests, dependency lock file, and reproducibility instructions for the TALIS 2024 revision. It deliberately does not contain data, figures, result tables, restricted teacher microdata, or row-level linkage keys. Authorised users must obtain the original TALIS teacher data through the OECD route.

## Scope of the final package

- Final primary results report design-adjusted association paths `a`, `b`, `c`, and `c'` only.
- No coefficient-product, proportion-mediated, or probability-scale indirect effect is presented as a final result.
- `results/output_final/B02/association_only_decision.md` records the remaining author decision required before any quantitative mediation claim is restored.

## Reproduce

1. Create the documented environment with the pinned packages in `environment/requirements.lock`.
2. Run `python code/run_all.py --raw-file <path-to-ttgintt4.csv> --run-dir results/release_v0.2.0`. The entry point prepares the derived analysis file, runs the supplementary analyses, and renders the figure. A full run performs 100 Fay-BRR replicates for the primary analysis, item-completeness analyses, and each of 20 multiply imputed datasets.
3. Inspect `<run-dir>/output/B01/reproduction.log`, `<run-dir>/output/B06/run_metadata.json`, and `<run-dir>/output/B05/release_manifest.csv`. The metadata records the code revision, software environment, external raw-file checksum, sample counts, and run mode without publishing the restricted data. The temporary `<run-dir>/restricted_working` directory is ignored by Git.

After a local run, audit-ready outputs are organised as `<run-dir>/output/B01` through `<run-dir>/output/B06`. Raw data, respondent identifiers, and any absolute local paths are excluded from this release.

## Inference labels

- `TableS1`: Fay-BRR 95% confidence intervals.
- `TableS3`: Fay-BRR 95% confidence intervals for item-completeness analyses; MNAR delta scenarios are point estimates only.
- `TableS4`: paths involving the imputed mediator combine within-imputation Fay-BRR variance with between-imputation variance using Rubin's rules; c does not depend on the mediator and therefore has its direct Fay-BRR interval.
- `TableS0` and `TableS2`: descriptive or specification-comparison point estimates, with no design-based confidence intervals.

## Version and DOI

The prepared package version is recorded in `VERSION`. The repository has no permanent DOI until an authorised maintainer creates a GitHub Release and archives it through Zenodo; see `ARCHIVING.md`.
