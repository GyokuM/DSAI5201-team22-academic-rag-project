# Data Assets Note

This repository includes the code, notebooks, summary tables, plots, and application components required for the final DSAI5201 project submission.

Some large experimental assets are intentionally excluded from standard Git tracking because they are too large for a normal GitHub repository or are better treated as local-only reproducible artifacts. These excluded assets include:

- `materials/project_part1/data/open_ragbench/`
- `materials/project_part1/data/processed/*.jsonl`
- `materials/project_part1/data/rawpdf/`
- `materials/project_part2/*.npy`

These files are omitted for the following reasons:

- several exceed GitHub's per-file size limit
- some are generated artifacts rather than source code
- local cache files do not need to be versioned in the final submission repository

If long-term versioning of these assets is required in the future, Git LFS or a separate data storage workflow should be used.
