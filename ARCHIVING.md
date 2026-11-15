# Release and archival checklist

This public package advances the reproducibility release to `v0.3.1`, prepared
on the public release branch `code-release-v0.2.0`. Its annotated Git tag
`v0.3.1` is created only after the KHB, Q35 nonresponse, and Table S0 weighted
descriptive-statistics outputs, tests, and public-data scan are final. A GitHub
commit hash or tag alone is not a DOI.

1. Confirm that no restricted TALIS row-level file or linkage key has been added.
2. Create a GitHub Release from the annotated tag `v0.3.1` after confirming
   the release branch and tests.
3. Connect the repository to Zenodo, archive that release, and insert the minted
   DOI into this file, `CITATION.cff`, and the manuscript Data/Code Availability
   statement.
4. Preserve the release assets and `B06/run_metadata.json`; do not upload the
   restricted OECD teacher microdata or local working file.

DOI status: `PENDING_ZENODO_ARCHIVAL`.
