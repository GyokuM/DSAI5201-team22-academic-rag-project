# Data Assets Policy

This repository keeps the project code, notebooks, summaries, CSV results, plots, and integration app in Git.

Large local-only assets are intentionally not tracked in normal Git because they are too large for a standard GitHub repo:

- `materials/project_part1/data/open_ragbench/`
- `materials/project_part1/data/processed/*.jsonl`
- `materials/project_part1/data/rawpdf/`
- `materials/project_part2/*.npy`

Why:

- several files exceed GitHub's 100 MB per-file limit
- pushing them would fail on a normal repository
- they are reproducible or local experiment artifacts rather than code deliverables

If you later want to version those files, use Git LFS instead of normal Git.
