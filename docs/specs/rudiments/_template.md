# PAS #{NUMBER}: {RUDIMENT NAME}

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | {1-40} |
| **Name** | {Official PAS name} |
| **Category** | {roll / diddle / flam / drag} |
| **Family** | {e.g., single stroke, paradiddle, basic flam, ratamacue} |
| **Composed Of** | {Primitive components, e.g., "singles + diddle", "flam + paradiddle"} |
| **Related** | {Links to related rudiments by PAS number and name} |
| **NARD Original** | {Yes / No} |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | {e.g., 4/4} |
| **Base Subdivision** | {quarter / eighth / triplet / sixteenth / sextuplet / 32nd} |
| **Cycle Length** | {N beats} |
| **Strokes Per Cycle** | {Total count including grace notes} |
| **Primary Strokes Per Cycle** | {Count excluding grace notes} |

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | 0 | 1 | {R/L} | {A/t/g/d1/d2/b} |
| 2 | {fraction} | {slot name} | {R/L} | {type} |
| ... | ... | ... | ... | ... |

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
{e.g., R L R R | L R L L}
```

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | {R/L} | {accent/tap/grace/d1/d2/buzz} | {primary / grace / none} | {# or n/a} | {Any special notes} |
| ... | ... | ... | ... | ... | ... |

### Ornament Timing

<!-- Only include sections relevant to this rudiment. Omit sections that do not apply. -->

**Flam Grace Notes** (if applicable):
- Grace offset from primary: {negative beat fraction, e.g., -1/32}
- Grace hand: {opposite of primary}

**Drag Grace Notes** (if applicable):
- First grace offset: {e.g., -1/16}
- Second grace offset: {e.g., -1/32}
- Grace hand: {same hand for both, opposite of primary}

**Diddle Timing** (if applicable):
- Each diddle stroke: {fraction of parent grid slot, e.g., 1/2 of 1/4 beat = 1/8 beat each}

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent | 1.0 | {Reference level} |
| tap | {0.65 - 0.77} | {Context-specific value or range} |
| {other types} | {ratio} | {notes} |

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
{e.g., >  -  -  >  |  >  -  -  >}
```

### Dynamic Contour

<!-- Describe the overall dynamic shape of the rudiment. Is there a crescendo, decrescendo, or repeating accent pattern? How does the dynamic contour support the musical character? -->

{Description of the dynamic shape and character}

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | {R/L} | {type} | {full/half/low/tap} | {wrist + arm / wrist / fingers / rebound} |
| ... | ... | ... | ... | ... |

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow ({BPM range}) | {Description of technique at slow tempos} |
| Moderate ({BPM range}) | {Description of technique at moderate tempos} |
| Fast ({BPM range}) | {Description of technique at fast tempos} |

---

## 6. Variations & Pedagogy

### Common Variations

<!-- List known variations: NARD version differences, regional interpretations, hybrid forms. -->

- {Variation 1}
- {Variation 2}

### Prerequisites

<!-- Rudiments that should be learned before this one. Reference by PAS number and name. -->

- {PAS #N: Rudiment Name}

### Builds Toward

<!-- Rudiments that build on this one. Reference by PAS number and name. -->

- {PAS #N: Rudiment Name}

### Teaching Notes

<!-- Pedagogical advice: common mistakes, practice strategies, what to listen for. -->

{Teaching guidance}

### Historical Context

<!-- NARD origins, historical significance, notable performers or literature. -->

{Historical notes}

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
{e.g., >       >
       lR  L  rL  R
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1     e     &     a   | 2     e     &     a   |
Hand:    | R     L     R     R   | L     R     L     L   |
Type:    | A     t     A     d1  | A     t     A     d1  |
Accent:  | >           >         | >           >         |
```
