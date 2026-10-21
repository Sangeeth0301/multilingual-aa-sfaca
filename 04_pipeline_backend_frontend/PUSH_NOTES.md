# Push notes: four-member layout

This branch (`four-member-layout`) reorganizes the repository into four member folders plus shared files. It changes no
behaviour: all 47 tests pass from the repository root and the real model served an analysis from the new paths.

## What moved

| Old location | New location |
|:-|:-|
| `src/clusters.py`, `data_loader.py`; `scripts/` data scripts; `docs/DATA_CARD.md` | `01_data_engineering/` |
| `src/model.py`, `contrastive_loss.py`, `compat.py`, `train.py`; training notebooks; `EXPERIMENTS.md` | `02_model_training/` |
| `src/inference.py` and the evaluation, calibration and Hub scripts | `03_inference_evaluation/` |
| `src/analyzer.py`, `sanitizer.py`, `router.py`, `file_parser.py`, `exporters.py`; `backend/`; `docs/index.html`; `deploy/`; launcher | `04_pipeline_backend_frontend/` |
| `augment_gemini_claude.py`, `sfaca_ui.html`, `scratch/` | `archive/old_prototypes/` (kept, not deleted) |

New shared files: `sfaca_paths.py` (adds every member's `src/` to the import path), `conftest.py`, `TEAM_PLAN.md`,
`.github/workflows/pages.yml`, and a `REPORT.md` in each member folder.

## Before merging: switch GitHub Pages

Pages used to publish `docs/`, which no longer exists. After merging to the branch Pages publishes from:

1. Repository Settings, Pages, Source: **GitHub Actions** (or run
   `gh api -X PUT repos/OWNER/REPO/pages -f build_type=workflow`).
2. The workflow `.github/workflows/pages.yml` then publishes `04_pipeline_backend_frontend/web` on every change there,
   including the tunnel address that `ops/start_public.ps1` writes to `web/api_url.txt`.

Until Pages is switched the live site keeps serving the old `docs/` copy from `main`.

## Not changed

Adapters (`results/adapters`, 36 MB) and data are gitignored and must be provided separately. The Docker bundle in
`deploy/` has still not been built. See each member's `REPORT.md` for limits.
