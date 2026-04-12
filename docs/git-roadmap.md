# Git Roadmap

Recommended commit sequence for showing your integration work clearly:

## Commit 1

Add the merged source materials only:

- `.gitignore`
- `materials/project_part1/`
- `materials/project_part2/`
- `materials/Project_part_3/`
- `materials/README.md`
- `docs/data-assets.md`

Suggested message:

```bash
git commit -m "chore: import merged project materials from three workstreams"
```

## Commit 2

Add the integrated app shell:

- `backend/`
- `frontend/`
- root `README.md`
- `docs/git-roadmap.md`

Suggested message:

```bash
git commit -m "feat: add integrated academic rag studio app"
```

## Commit 3+

Continue with incremental work:

- rerun and fix Part 3 evaluation
- replace fallback demo answer generation
- improve frontend polish
- add upload PDF if needed

Suggested style:

```bash
git commit -m "feat: rerun part3 evaluation against answers.json"
git commit -m "feat: connect model-backed generation to demo"
git commit -m "refactor: reorganize backend loaders and results endpoints"
```
