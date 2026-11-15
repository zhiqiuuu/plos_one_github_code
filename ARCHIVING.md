# Release and archival checklist

This public package is the reproducibility release `v0.3.0`, prepared on the
public release branch `code-release-v0.2.0`. Its annotated Git tag `v0.3.0` is
created only after the KHB and Q35 nonresponse outputs, tests, and public-data
scan in this package are final. A GitHub commit hash or tag alone is not a DOI.

1. Confirm that no restricted TALIS row-level file or linkage key has been added.
2. Create a GitHub Release from the annotated tag `v0.3.0` after confirming
   the release branch and tests.
3. Connect the repository to Zenodo, archive that release, and insert the minted
   DOI into this file, `CITATION.cff`, and the manuscript Data/Code Availability
   statement.
4. Preserve the release assets and `B06/run_metadata.json`; do not upload the
   restricted OECD teacher microdata or local working file.

DOI status: `PENDING_ZENODO_ARCHIVAL`.
