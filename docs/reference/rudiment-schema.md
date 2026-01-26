# Rudiment Schema Specification

This document defines the YAML schema for rudiment definitions in SOUSA. All 40 PAS rudiments are defined in `dataset_gen/rudiments/definitions/`.

## Schema Overview

Each rudiment is defined in a YAML file with the following structure:

```yaml
name: <string>           # Official PAS name
slug: <string>           # URL-safe identifier
category: <enum>         # roll | diddle | flam | drag
pas_number: <int>        # Official PAS number (1-40)
description: <string>    # Brief description

pattern:
  strokes: <list>        # Stroke definitions
  beats_per_cycle: <float>  # Duration of one pattern cycle

subdivision: <enum>      # Base note subdivision
tempo_range: [min, max]  # Recommended BPM range

params:                  # Category-specific parameters
  <param>: <value>
```

---

## Field Reference

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Official PAS rudiment name |
| `slug` | string | Yes | URL-safe identifier (lowercase, underscores only) |
| `category` | enum | Yes | One of: `roll`, `diddle`, `flam`, `drag` |
| `pas_number` | int | No | Official PAS number (1-40) |
| `description` | string | No | Brief description of the rudiment |
| `pattern` | object | Yes | Sticking pattern definition |
| `subdivision` | enum | No | Base subdivision (default: `sixteenth`) |
| `tempo_range` | [int, int] | No | Min/max BPM (default: [60, 180]) |
| `params` | object | No | Category-specific parameters |
| `starts_on_left` | bool | No | If true, alternate version starts on left hand |

### Pattern Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `strokes` | list | Yes | List of stroke definitions |
| `beats_per_cycle` | float | No | Beats per pattern cycle (default: 1.0) |

### Stroke Object

Each stroke in the `strokes` list:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `hand` | enum | Yes | `R` (right) or `L` (left) |
| `type` | enum | No | Stroke type (default: `tap`) |
| `grace_offset` | float | No | Timing offset in beats for grace notes |
| `diddle_position` | int | No | Position in diddle (1 or 2) |

---

## Stroke Types

SOUSA supports five stroke types, each with distinct velocity and timing characteristics:

### `tap`

Standard unaccented stroke. Default velocity: 85.

```yaml
- {hand: R, type: tap}
```

### `accent`

Emphasized stroke with higher velocity. Default velocity: 110.

```yaml
- {hand: R, type: accent}
```

### `grace`

Grace note preceding a primary stroke. Requires `grace_offset` to specify timing.

```yaml
- {hand: L, type: grace, grace_offset: -0.05}  # 0.05 beats before primary
- {hand: R, type: accent}                       # Primary stroke
```

!!! note "Grace Note Timing"
    `grace_offset` is specified in beats (negative = before the beat). Typical values:

    - Flams: `-0.05` (single grace note)
    - Drags: `-0.08` and `-0.04` (two grace notes)

### `diddle`

Double stroke (one of two strokes played with the same hand). Must specify `diddle_position`.

```yaml
- {hand: R, type: diddle, diddle_position: 1}  # First stroke
- {hand: R, type: diddle, diddle_position: 2}  # Second stroke
```

### `buzz`

Press roll stroke with multiple bounces. Used in buzz rolls.

```yaml
- {hand: R, type: buzz}
```

---

## Subdivisions

The `subdivision` field specifies the base rhythmic subdivision:

| Value | Notes per Beat | Common Usage |
|-------|----------------|--------------|
| `quarter` | 1 | Slow exercises |
| `eighth` | 2 | Drags, some flams |
| `triplet` | 3 | Triplet-based rudiments |
| `sixteenth` | 4 | Most rudiments (default) |
| `sextuplet` | 6 | Fast rolls |
| `thirtysecond` | 8 | Double-time passages |

---

## Category Parameters

Category-specific parameters control articulation-specific behaviors.

### Flam Parameters

For rudiments in the `flam` category:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `flam_spacing_range` | [float, float] | Min/max grace note spacing in ms | `[15, 50]` |

```yaml
params:
  flam_spacing_range: [15, 50]  # Grace note 15-50ms before primary
```

### Diddle Parameters

For rudiments in the `diddle` category:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `diddle_ratio_range` | [float, float] | Ratio range between first/second stroke duration | `[0.9, 1.1]` |

```yaml
params:
  diddle_ratio_range: [0.9, 1.1]  # Nearly even strokes
```

### Roll Parameters

For rudiments in the `roll` category:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `roll_type` | enum | `open`, `closed`, or `buzz` | `"open"` |
| `roll_strokes_per_beat` | int | Number of strokes per beat | `4` |

```yaml
params:
  roll_type: open
  roll_strokes_per_beat: 4
```

### Drag Parameters

For rudiments in the `drag` category:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `drag_spacing_range` | [float, float] | Min/max spacing between drag grace notes in ms | `[10, 30]` |

```yaml
params:
  drag_spacing_range: [10, 30]
```

### Buzz Roll Parameters

For buzz/press roll rudiments:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `buzz_strokes_range` | [int, int] | Min/max bounce strokes per primary | `[3, 8]` |

```yaml
params:
  buzz_strokes_range: [3, 8]
```

---

## Complete Examples

### Single Paradiddle (Diddle Category)

```yaml
# File: 16_single_paradiddle.yaml
#
# The single paradiddle: RLRR LRLL
# One of the most fundamental paradiddle patterns with accents
# on the first stroke of each group.

name: Single Paradiddle
slug: single_paradiddle
category: diddle
pas_number: 16
description: RLRR LRLL - the most fundamental paradiddle pattern

pattern:
  strokes:
    # First group: RLRR
    - {hand: R, type: accent}              # Accented primary
    - {hand: L, type: tap}                 # Single tap
    - {hand: R, type: diddle, diddle_position: 1}  # First diddle
    - {hand: R, type: diddle, diddle_position: 2}  # Second diddle
    # Second group: LRLL
    - {hand: L, type: accent}              # Accented primary
    - {hand: R, type: tap}                 # Single tap
    - {hand: L, type: diddle, diddle_position: 1}  # First diddle
    - {hand: L, type: diddle, diddle_position: 2}  # Second diddle
  beats_per_cycle: 2  # 8 sixteenths = 2 beats

subdivision: sixteenth
tempo_range: [60, 200]

params:
  diddle_ratio_range: [0.9, 1.1]  # Diddles should be nearly even
```

### Flam Tap (Flam Category)

```yaml
# File: 22_flam_tap.yaml
#
# Alternating flams with a tap on each hand.
# Pattern: flam-R tap-R | flam-L tap-L

name: Flam Tap
slug: flam_tap
category: flam
pas_number: 22
description: Alternating flams with a tap on each hand

pattern:
  strokes:
    # Flam R, tap R
    - {hand: L, type: grace, grace_offset: -0.05}  # Grace note (left hand)
    - {hand: R, type: accent}                       # Primary (right hand)
    - {hand: R, type: tap}                          # Tap (right hand)
    # Flam L, tap L
    - {hand: R, type: grace, grace_offset: -0.05}  # Grace note (right hand)
    - {hand: L, type: accent}                       # Primary (left hand)
    - {hand: L, type: tap}                          # Tap (left hand)
  beats_per_cycle: 1.5  # Triplet feel

subdivision: sixteenth
tempo_range: [60, 180]

params:
  flam_spacing_range: [15, 50]  # Grace note 15-50ms before primary
```

### Drag (Drag Category)

```yaml
# File: 31_drag.yaml
#
# Two grace notes (diddle) followed by primary stroke.
# Also known as a "ruff" in some contexts.

name: Drag
slug: drag
category: drag
pas_number: 31
description: Two grace notes (diddle) followed by primary stroke

pattern:
  strokes:
    # Drag R (grace LL + accent R)
    - {hand: L, type: grace, grace_offset: -0.08}  # First grace
    - {hand: L, type: grace, grace_offset: -0.04}  # Second grace
    - {hand: R, type: accent}                       # Primary stroke
    # Drag L (grace RR + accent L)
    - {hand: R, type: grace, grace_offset: -0.08}  # First grace
    - {hand: R, type: grace, grace_offset: -0.04}  # Second grace
    - {hand: L, type: accent}                       # Primary stroke
  beats_per_cycle: 1

subdivision: eighth
tempo_range: [60, 160]

params:
  drag_spacing_range: [10, 30]  # Spacing between the two grace notes
```

---

## Validation Rules

The schema enforces these validation rules:

### Slug Format

```python
# Must be lowercase alphanumeric with underscores only
if not re.match(r"^[a-z0-9_]+$", slug):
    raise ValueError("Invalid slug format")
```

### Diddle Position

Diddle strokes must specify position 1 or 2:

```python
if stroke.stroke_type == StrokeType.DIDDLE:
    assert stroke.diddle_position in (1, 2)
```

### Grace Note Offset

Grace notes must have a `grace_offset` value:

```python
if stroke.stroke_type == StrokeType.GRACE:
    assert stroke.grace_offset is not None
    assert stroke.grace_offset < 0  # Must be before the beat
```

### Category-Parameter Matching

Category-specific parameters are validated against category:

| Category | Valid Parameters |
|----------|------------------|
| `flam` | `flam_spacing_range` |
| `diddle` | `diddle_ratio_range` |
| `roll` | `roll_type`, `roll_strokes_per_beat`, `buzz_strokes_range` |
| `drag` | `drag_spacing_range` |

---

## Creating New Rudiments

To add a custom rudiment:

1. Create a YAML file in `dataset_gen/rudiments/definitions/`
2. Follow the naming convention: `{number}_{slug}.yaml`
3. Include all required fields
4. Add category-appropriate parameters
5. Test loading with:

```python
from dataset_gen.rudiments.loader import load_rudiment

rudiment = load_rudiment("my_custom_rudiment")
print(rudiment.pattern.stroke_count())
```

!!! warning "PAS Numbers"
    PAS numbers 1-40 are reserved for official PAS rudiments. Use numbers > 40 for custom rudiments.

---

## Complete Rudiment List

### Roll Rudiments (1-15)

| # | Slug | Contains |
|---|------|----------|
| 1 | `single_stroke_roll` | - |
| 2 | `single_stroke_four` | - |
| 3 | `single_stroke_seven` | - |
| 4 | `multiple_bounce_roll` | Buzz |
| 5 | `triple_stroke_roll` | - |
| 6 | `double_stroke_open_roll` | Diddle |
| 7-15 | `five_stroke_roll` through `seventeen_stroke_roll` | Diddle |

### Paradiddle Rudiments (16-19)

| # | Slug | Contains |
|---|------|----------|
| 16 | `single_paradiddle` | Diddle |
| 17 | `double_paradiddle` | Diddle |
| 18 | `triple_paradiddle` | Diddle |
| 19 | `paradiddle_diddle` | Diddle |

### Flam Rudiments (20-30)

| # | Slug | Contains |
|---|------|----------|
| 20 | `flam` | Flam |
| 21 | `flam_accent` | Flam |
| 22 | `flam_tap` | Flam |
| 23 | `flamacue` | Flam |
| 24 | `flam_paradiddle` | Flam, Diddle |
| 25 | `single_flammed_mill` | Flam, Diddle |
| 26 | `flam_paradiddle_diddle` | Flam, Diddle |
| 27 | `pataflafla` | Flam |
| 28 | `swiss_army_triplet` | Flam |
| 29 | `inverted_flam_tap` | Flam |
| 30 | `flam_drag` | Flam, Drag |

### Drag Rudiments (31-40)

| # | Slug | Contains |
|---|------|----------|
| 31 | `drag` | Drag |
| 32 | `single_drag_tap` | Drag |
| 33 | `double_drag_tap` | Drag |
| 34 | `lesson_25` | Drag |
| 35 | `single_dragadiddle` | Drag, Diddle |
| 36 | `drag_paradiddle_1` | Drag, Diddle |
| 37 | `drag_paradiddle_2` | Drag, Diddle |
| 38 | `single_ratamacue` | Drag |
| 39 | `double_ratamacue` | Drag |
| 40 | `triple_ratamacue` | Drag |

## See Also

- [Architecture](architecture.md) - Pipeline overview
- [Score Computation](score-computation.md) - How scores are calculated from rudiment performances
