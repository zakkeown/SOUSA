# Conventions & Shared Definitions

This document defines all shared vocabulary, scales, and notation conventions used across the 40 PAS rudiment engineering specs. Individual specs reference these definitions rather than redefining them.

---

## Beat Fraction Notation

All timing in these specs is expressed as **fractions of a beat**. No millisecond values appear anywhere.

Rules:

- Always use simplified fractions: `1/4` not `2/8`
- Position `0` = the downbeat of the cycle
- Negative fractions = before the beat (grace notes, ornaments)
- The beat unit is always the **quarter note** unless stated otherwise
- Positions within a multi-beat cycle are expressed as whole + fraction (e.g., `1 + 1/4` = the "e" of beat 2)

---

## Grid Slot Naming

Standard counting names for each subdivision level:

| Subdivision | Slots per Beat | Names |
|-------------|----------------|-------|
| Quarter | 1 | 1 |
| Eighth | 2 | 1, & |
| Triplet | 3 | 1, &, a |
| Sixteenth | 4 | 1, e, &, a |
| Sextuplet | 6 | 1, la, li, &, la, li |
| 32nd | 8 | 1, &, e, &, a, &, e, & |

Beat numbers increment for multi-beat cycles (e.g., beat 2 sixteenths: 2, 2e, 2&, 2a).

---

## Stroke Types

| Abbreviation | Type | Description |
|--------------|------|-------------|
| A | accent | Emphasized stroke, full arm/wrist motion |
| t | tap | Unaccented single stroke |
| g | grace | Grace note (flam or drag ornament) |
| d1 | diddle pos 1 | First stroke of a double (same hand) |
| d2 | diddle pos 2 | Second stroke of a double (same hand) |
| b | buzz | Press/buzz roll stroke with multiple bounces |

---

## Velocity Ratio Scale

All velocities are expressed relative to **accent = 1.0**.

| Stroke Type | Ratio Range | Notes |
|-------------|-------------|-------|
| accent | 1.0 | Reference level |
| tap | 0.65 - 0.77 | Unaccented single stroke |
| grace (flam) | 0.50 - 0.70 | Single grace note before primary |
| grace (drag) | 0.45 - 0.65 | Each of the two drag grace notes |
| diddle pos 1 | matches parent | Same as the accent/tap it belongs to |
| diddle pos 2 | 0.90 - 0.98 x pos 1 | Slight decay on second bounce |
| buzz bounce | 0.60 - 0.80 x primary | Per-bounce, decaying |

---

## Stroke Height Classes

| Class | Stick Height | Typical Use |
|-------|--------------|-------------|
| full | 12"+ above head | Accents, loud strokes |
| half | 6 - 12" | Medium dynamics |
| low | 2 - 6" | Taps, unaccented strokes |
| tap | < 2" | Grace notes, buzz bounces |

---

## Motion Types

| Type | Description | Typical Stroke |
|------|-------------|----------------|
| wrist + arm | Full Moeller whip motion | Accents at moderate tempos |
| wrist | Primary motion from wrist | Most strokes at moderate tempos |
| fingers | Finger control, minimal wrist | Grace notes, fast passages, diddle pos 2 |
| rebound | Controlled bounce off head | Buzz rolls, fast diddles |

---

## Categories

| Category | PAS Numbers | Description |
|----------|-------------|-------------|
| Roll | 1 - 15 | Single strokes, buzz/multiple bounce, double strokes, counted rolls |
| Diddle | 16 - 19 | Paradiddle family (single + double stroke combinations) |
| Flam | 20 - 30 | Patterns featuring single grace note ornaments |
| Drag | 31 - 40 | Patterns featuring double grace note ornaments |

---

## Ornament Timing Conventions

### Flam Grace Notes

- Expressed as a **negative beat fraction** from the primary stroke
- Standard offset: **-1/32 beat** (at moderate tempos)
- Range: **-1/64 to -1/16** depending on tempo and style
- Grace note always played by the **opposite hand** from the primary

### Drag Grace Notes

- Two grace notes preceding the primary stroke
- First grace: **-1/16 beat** before primary
- Second grace: **-1/32 beat** before primary
- Both grace notes played by the **same hand** (opposite from primary)
- Grace notes form a quick diddle (same hand, two bounces)

### Diddle Timing

- Two strokes sharing a single grid position's duration
- Each diddle stroke occupies **1/2 of the parent grid slot**
- At slow tempos: **open** (distinguishable as two separate strokes)
- At fast tempos: **closed** (nearly blurred together)

---

## Notation Conventions

### Stick Notation

- `R` = right hand stroke
- `L` = left hand stroke
- `>` = accent marker (above the stroke)
- `(g)` = grace note prefix
- `|` = bar/group separator

### Flam Notation

- `lR` = left grace + right primary (right-hand flam)
- `rL` = right grace + left primary (left-hand flam)

### Drag Notation

- `llR` = left-left grace + right primary (right-hand drag)
- `rrL` = right-right grace + left primary (left-hand drag)

---

## Families

Sub-groupings within each category:

### Roll Category

| Family | PAS Numbers | Description |
|--------|-------------|-------------|
| single stroke | 1 - 3 | Alternating single strokes |
| buzz | 4 | Multiple bounce (press) roll |
| triple stroke | 5 | Three strokes per hand |
| double stroke | 6 | Open double stroke roll (diddles alternating) |
| counted roll | 7 - 15 | Specific stroke counts (5, 6, 7, 9, 10, 11, 13, 15, 17) |

### Diddle Category

| Family | PAS Numbers | Description |
|--------|-------------|-------------|
| paradiddle | 16 - 19 | Single, double, triple paradiddles + paradiddle-diddle |

### Flam Category

| Family | PAS Numbers | Description |
|--------|-------------|-------------|
| basic flam | 20 - 22 | Flam, flam accent, flam tap |
| compound flam | 23 - 26 | Flamacue, flam paradiddle, single flammed mill, flam paradiddle-diddle |
| advanced flam | 27 - 30 | Pataflafla, Swiss army triplet, inverted flam tap, flam drag |

### Drag Category

| Family | PAS Numbers | Description |
|--------|-------------|-------------|
| basic drag | 31 - 33 | Drag, single drag tap, double drag tap |
| compound drag | 34 - 37 | Lesson 25, single dragadiddle, drag paradiddle #1, drag paradiddle #2 |
| ratamacue | 38 - 40 | Single, double, triple ratamacue |
