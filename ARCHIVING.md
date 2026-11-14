# Release and archival checklist

The public release branch is `code-release-v0.2.0`. The reproducibility
release is fixed by annotated tag `v0.2.0` at commit `cef4ffa`. It does not yet
have a permanent archive DOI; a GitHub commit hash or tag alone is not a DOI.

1. Confirm that no restricted TALIS row-level file or linkage key has been added.
2. Create a GitHub Release from the already-pushed annotated tag `v0.2.0` after
   confirming the release branch and tests.
3. Connect the repository to Zenodo, archive that release, and insert the minted
   DOI into this file, `CITATION.cff`, and the manuscript Data/Code Availability
   statement.
4. Preserve the release assets and `B06/run_metadata.json`; do not upload the
   restricted OECD teacher microdata or local working file.

DOI status: `PENDING_ZENODO_ARCHIVAL`.
