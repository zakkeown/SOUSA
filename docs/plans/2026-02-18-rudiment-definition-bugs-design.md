# Fix Three Rudiment Definition/Generation Bugs

**Date**: 2026-02-18

## Problem

Three rudiment definitions produce incorrect or undifferentiated output:

1. **Double Drag Tap (PAS #33)** — identical to Single Drag Tap. Missing an entire drag per half-cycle (8 strokes instead of 12).
2. **Swiss Army Triplet (PAS #28)** — sticking identical to Flam Accent. Should be R L L | L R R, not R L R | L R L.
3. **Multiple Bounce Roll (PAS #4)** — `roll_type: buzz` and `buzz_strokes_range` params exist but are ignored. Output is indistinguishable from single-stroke-roll.

## Fixes

### Bug 1: Double Drag Tap

Update YAML from 8 to 12 strokes per cycle. Each half-cycle gets two drag pairs before the accent and tap:

```
Current:  GG A T | GG A T          (8 strokes, single drag per accent)
Fixed:    GG GG A T | GG GG A T   (12 strokes, double drag per accent)
```

Switch subdivision to `triplet` (6 strokes per beat divides naturally).

### Bug 2: Swiss Army Triplet

Change tap hands on strokes 3 and 7:

```
Current:  G-R L R | G-L R L   (alternating, same as flam accent)
Fixed:    G-R L L | G-L R R   (doubled last stroke)
```

### Bug 3: Multiple Bounce Roll

Support both buzz modes via `buzz_detail` parameter:

- **`sub_strokes`**: Expand each BUZZ primary into 3-8 rapid sub-strokes in MIDI with velocity decay and profile-driven density.
- **`marking`**: Mark strokes as `buzz` type in labels for differentiation, add velocity jitter, but no MIDI sub-strokes.

Changes required:

| File | Change |
|------|--------|
| `definitions/33_double_drag_tap.yaml` | Add second drag pair (8 → 12 strokes), triplet subdivision |
| `definitions/28_swiss_army_triplet.yaml` | Fix sticking on strokes 3 and 7 |
| `definitions/04_multiple_bounce_roll.yaml` | Switch to `B` notation, add `buzz_detail: sub_strokes` |
| `rudiments/schema.py` | Add `buzz_detail` field to `RudimentParams` |
| `rudiments/loader.py` | Support `B` character in simple pattern parser → `StrokeType.BUZZ` |
| `midi_gen/articulations.py` | Implement `sub_strokes` mode: expand BUZZ into rapid sub-strokes with decay |
| `labels/compute.py` | Handle `BUZZ` stroke type in label generation |
| `tests/` | Regression tests for all three fixes |
