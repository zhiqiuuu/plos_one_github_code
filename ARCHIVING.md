# Release and archival checklist

This repository is prepared for a public code release, but it does not yet have
a permanent archive DOI. A GitHub commit hash alone is not a DOI.

1. Confirm that no restricted TALIS row-level file or linkage key has been added.
2. Create a GitHub Release from the annotated tag `v0.2.0` after the full raw-data
   reproduction run and release tests pass.
3. Connect the repository to Zenodo, archive that release, and insert the minted
   DOI into this file, `CITATION.cff`, and the manuscript Data/Code Availability
   statement.
4. Preserve the release assets and `B06/run_metadata.json`; do not upload the
   restricted OECD teacher microdata or local working file.

DOI status: `PENDING_ZENODO_ARCHIVAL`.
