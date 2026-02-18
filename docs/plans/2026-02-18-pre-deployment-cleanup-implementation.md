# Pre-Deployment Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove dead code, development artifacts, and orphaned files; fix minor config issues; wire rudiment specs into docs site.

**Architecture:** Single branch with logically grouped commits. Archive reusable code to named branches before deleting from main. All other changes are deletions or small edits.

**Tech Stack:** Git, mkdocs.yml (YAML), pyproject.toml (TOML), GitHub Actions YAML

---

### Task 1: Archive SFZ synthesizer to a branch

**Files:**
- Archive (then delete): `dataset_gen/audio_synth/sfz_parser.py`
- Archive (then delete): `dataset_gen/audio_synth/sfz_renderer.py`
- Archive (then delete): `dataset_gen/audio_synth/sfz_synthesizer.py`
- Archive (then delete): `scripts/setup_sfz.py`

**Step 1: Create the archive branch from current main**

```bash
git branch archive/sfz-synthesizer main
```

This preserves the SFZ code at the current main HEAD, so it can be recovered later.

**Step 2: Delete the SFZ files from working branch**

```bash
git rm dataset_gen/audio_synth/sfz_parser.py
git rm dataset_gen/audio_synth/sfz_renderer.py
git rm dataset_gen/audio_synth/sfz_synthesizer.py
git rm scripts/setup_sfz.py
```

**Step 3: Commit**

```bash
git commit -m "chore: remove orphaned SFZ synthesizer (archived to archive/sfz-synthesizer)"
```

---

### Task 2: Archive ML baselines to a branch

**Files:**
- Archive (then delete): `examples/baselines/` (all 8 files)

**Step 1: Create the archive branch from current main**

```bash
git branch archive/ml-baselines main
```

**Step 2: Delete baselines directory**

```bash
git rm -r examples/baselines/
```

**Step 3: Commit**

```bash
git commit -m "chore: remove untested ML baselines (archived to archive/ml-baselines)"
```

---

### Task 3: Delete experiments directory and related script

**Files:**
- Delete: `experiments/` (entire directory — `__init__.py`, `augmentation_ablation.py`, `score_analysis.py`, `soundfont_ablation.py`, `split_validation.py`, `results/`)
- Delete: `scripts/run_experiments.py`

**Step 1: Delete experiments and the runner script**

```bash
git rm -r experiments/
git rm scripts/run_experiments.py
```

**Step 2: Commit**

```bash
git commit -m "chore: remove experiments directory and runner script"
```

---

### Task 4: Delete plan docs

**Files:**
- Delete: all files in `docs/plans/` (9 files including the cleanup design doc)

**Step 1: Delete the plans directory**

```bash
git rm -r docs/plans/
```

**Step 2: Commit**

```bash
git commit -m "chore: remove completed plan/design docs"
```

---

### Task 5: Delete duplicate generation script

**Files:**
- Delete: `scripts/generate_clean_dataset.py`

**Step 1: Delete the file**

```bash
git rm scripts/generate_clean_dataset.py
```

**Step 2: Commit**

```bash
git commit -m "chore: remove duplicate generate_clean_dataset.py script"
```

---

### Task 6: Fix pyproject.toml

**Files:**
- Modify: `pyproject.toml:49-52` (analysis extra) and `pyproject.toml:78` (wandb comment)

**Step 1: Remove duplicate matplotlib from [analysis] extra**

Change `pyproject.toml` lines 49-52 from:

```toml
analysis = [
    "madmom @ git+https://github.com/CPJKU/madmom.git",
    "matplotlib>=3.7.0",
]
```

to:

```toml
analysis = [
    "madmom @ git+https://github.com/CPJKU/madmom.git",
]
```

(matplotlib is already in the `[dev]` extra at line 58)

**Step 2: Remove commented-out wandb dependency**

Change `pyproject.toml` lines 67-79 from:

```toml
ml = [
    # Core ML
    "torch>=2.0",
    "torchaudio>=2.0",
    "transformers>=4.35",
    # Training & logging
    "tensorboard>=2.14",
    # Analysis
    "scikit-learn>=1.3",
    "seaborn>=0.12",
    # Optional but recommended
    # "wandb>=0.16",  # Uncomment for weights & biases logging
]
```

to:

```toml
ml = [
    # Core ML
    "torch>=2.0",
    "torchaudio>=2.0",
    "transformers>=4.35",
    # Training & logging
    "tensorboard>=2.14",
    # Analysis
    "scikit-learn>=1.3",
    "seaborn>=0.12",
]
```

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: clean up pyproject.toml extras (remove duplicate matplotlib, wandb comment)"
```

---

### Task 7: Fix notebook broken link

**Files:**
- Modify: `notebooks/getting_started.ipynb` (last markdown cell, cell-27)

**Step 1: Fix the link in the last cell**

In cell-27 (the "Next Steps" markdown cell), change the link from:

```markdown
For more information, see the [SOUSA documentation](https://github.com/zkeown/rudimentary).
```

to:

```markdown
For more information, see the [SOUSA documentation](https://github.com/zakkeown/SOUSA).
```

**Step 2: Commit**

```bash
git add notebooks/getting_started.ipynb
git commit -m "fix: correct broken documentation link in getting_started notebook"
```

---

### Task 8: Standardize CI action versions

**Files:**
- Modify: `.github/workflows/docs.yml:21` (checkout@v4 → v6)
- Modify: `.github/workflows/claude.yml:29` (checkout@v4 → v6)
- Modify: `.github/workflows/claude-code-review.yml:30` (checkout@v4 → v6)

Some workflows already use v6 (test.yml, codeql.yml). Standardize the rest.

**Step 1: Update docs.yml**

Change line 21 from `actions/checkout@v4` to `actions/checkout@v6`.

**Step 2: Update claude.yml**

Change line 29 from `actions/checkout@v4` to `actions/checkout@v6`.

**Step 3: Update claude-code-review.yml**

Change line 30 from `actions/checkout@v4` to `actions/checkout@v6`.

**Step 4: Commit**

```bash
git add .github/workflows/docs.yml .github/workflows/claude.yml .github/workflows/claude-code-review.yml
git commit -m "ci: standardize actions/checkout to v6 across all workflows"
```

---

### Task 9: Wire rudiment specs into mkdocs.yml

**Files:**
- Modify: `mkdocs.yml:135-141` (Technical Reference nav section)

**Step 1: Add Rudiment Specs section to the nav**

In `mkdocs.yml`, expand the "Technical Reference" section (currently lines 135-141) to include the rudiment specs. The index file is `docs/specs/rudiments/_index.md`. Add after the existing entries:

```yaml
  - Technical Reference:
    - reference/index.md
    - Architecture: reference/architecture.md
    - Rudiment Schema: reference/rudiment-schema.md
    - Score Computation: reference/score-computation.md
    - Audio Processing: reference/audio-processing.md
    - Data Format: reference/data-format.md
    - Rudiment Specs:
      - specs/rudiments/_index.md
      - Conventions: specs/rudiments/_conventions.md
      - Single Stroke Roll: specs/rudiments/01_single_stroke_roll.md
      - Single Stroke Four: specs/rudiments/02_single_stroke_four.md
      - Single Stroke Seven: specs/rudiments/03_single_stroke_seven.md
      - Multiple Bounce Roll: specs/rudiments/04_multiple_bounce_roll.md
      - Triple Stroke Roll: specs/rudiments/05_triple_stroke_roll.md
      - Double Stroke Open Roll: specs/rudiments/06_double_stroke_open_roll.md
      - Five Stroke Roll: specs/rudiments/07_five_stroke_roll.md
      - Six Stroke Roll: specs/rudiments/08_six_stroke_roll.md
      - Seven Stroke Roll: specs/rudiments/09_seven_stroke_roll.md
      - Nine Stroke Roll: specs/rudiments/10_nine_stroke_roll.md
      - Ten Stroke Roll: specs/rudiments/11_ten_stroke_roll.md
      - Eleven Stroke Roll: specs/rudiments/12_eleven_stroke_roll.md
      - Thirteen Stroke Roll: specs/rudiments/13_thirteen_stroke_roll.md
      - Fifteen Stroke Roll: specs/rudiments/14_fifteen_stroke_roll.md
      - Seventeen Stroke Roll: specs/rudiments/15_seventeen_stroke_roll.md
      - Single Paradiddle: specs/rudiments/16_single_paradiddle.md
      - Double Paradiddle: specs/rudiments/17_double_paradiddle.md
      - Triple Paradiddle: specs/rudiments/18_triple_paradiddle.md
      - Paradiddle-Diddle: specs/rudiments/19_paradiddle_diddle.md
      - Flam: specs/rudiments/20_flam.md
      - Flam Accent: specs/rudiments/21_flam_accent.md
      - Flam Tap: specs/rudiments/22_flam_tap.md
      - Flamacue: specs/rudiments/23_flamacue.md
      - Flam Paradiddle: specs/rudiments/24_flam_paradiddle.md
      - Single Flammed Mill: specs/rudiments/25_single_flammed_mill.md
      - Flam Paradiddle-Diddle: specs/rudiments/26_flam_paradiddle_diddle.md
      - Pataflafla: specs/rudiments/27_pataflafla.md
      - Swiss Army Triplet: specs/rudiments/28_swiss_army_triplet.md
      - Inverted Flam Tap: specs/rudiments/29_inverted_flam_tap.md
      - Flam Drag: specs/rudiments/30_flam_drag.md
      - Drag: specs/rudiments/31_drag.md
      - Single Drag Tap: specs/rudiments/32_single_drag_tap.md
      - Double Drag Tap: specs/rudiments/33_double_drag_tap.md
      - Lesson 25: specs/rudiments/34_lesson_25.md
      - Single Dragadiddle: specs/rudiments/35_single_dragadiddle.md
      - Drag Paradiddle 1: specs/rudiments/36_drag_paradiddle_1.md
      - Drag Paradiddle 2: specs/rudiments/37_drag_paradiddle_2.md
      - Single Ratamacue: specs/rudiments/38_single_ratamacue.md
      - Double Ratamacue: specs/rudiments/39_double_ratamacue.md
      - Triple Ratamacue: specs/rudiments/40_triple_ratamacue.md
```

**Step 2: Commit**

```bash
git add mkdocs.yml
git commit -m "docs: wire 40 PAS rudiment specs into documentation site"
```

---

### Task 10: Update CLAUDE.md references

**Files:**
- Modify: `CLAUDE.md:119-121` (experiments reference)

**Step 1: Remove the score correlation analysis section**

In `CLAUDE.md`, remove the lines referencing `experiments.score_analysis` (around line 119-121):

```markdown
# 7. Score correlation analysis — identifies redundant scores, PCA, clustering
python -m experiments.score_analysis --data-dir output/dataset
```

Remove these 2 lines from the "Deep Analysis" code block.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: remove stale experiments reference from CLAUDE.md"
```

---

### Task 11: Run tests to verify nothing broke

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass. Nothing should have been affected since we only removed orphaned/unused code.

**Step 2: Run ruff to check for broken imports**

```bash
ruff check dataset_gen/
```

Expected: No new errors introduced.

**Step 3: Verify mkdocs config is valid**

```bash
pip install -e '.[docs]' 2>/dev/null && mkdocs build --strict 2>&1 | tail -20
```

Expected: Build succeeds (or warns about missing API doc references, which is pre-existing).
