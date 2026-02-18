# Pre-Deployment Cleanup Design

**Date:** 2026-02-18
**Status:** Approved

## Context

SOUSA has been through several development iterations and is now being deployed. This cleanup removes dead code, development artifacts, and fixes issues discovered during a full repo audit.

## Changes

### 1. Archive to branches, then delete from main

**SFZ synthesizer** → branch `archive/sfz-synthesizer`
- `dataset_gen/audio_synth/sfz_parser.py`
- `dataset_gen/audio_synth/sfz_renderer.py`
- `dataset_gen/audio_synth/sfz_synthesizer.py`
- `scripts/setup_sfz.py`

**ML baselines** → branch `archive/ml-baselines`
- `examples/baselines/` (8 files, ~136K of untested ML training code)

### 2. Delete dead files

- `docs/plans/` — all 8 completed plan/design docs (including this one, after implementation)
- `experiments/` — entire directory (score_analysis.py, results/)
- `scripts/run_experiments.py`
- `scripts/generate_clean_dataset.py` (undocumented duplicate of generate_dataset.py)

### 3. Fix CI and config

- `.github/workflows/test.yml` — `actions/checkout@v6` → `@v4` (v6 doesn't exist; bad dependabot bump)
- `.github/workflows/codeql.yml` — `actions/checkout@v6` → `@v4`
- `pyproject.toml` — remove duplicate `matplotlib` from `[analysis]` extra (keep in `[dev]`), remove commented-out wandb line
- `notebooks/getting_started.ipynb` — fix broken link from "rudimentary" → "SOUSA"

### 4. Wire rudiment specs into documentation

- Add "Technical Reference > Rudiment Specs" section to `mkdocs.yml` navigation
- All 40 PAS rudiment spec files in `docs/specs/rudiments/` become visible on the docs site

### 5. Update references

- Remove references to deleted files/directories from CLAUDE.md and README.md
- Clean up any stale cross-references

## Approach

Single branch, logically grouped commits, one PR. Most changes are deletions.
