# Rudiment Engineering Specs Design

**Date**: 2026-02-17
**Status**: Approved
**Scope**: Research & documentation — 40 PAS drum rudiment engineering specifications

## Goal

Create a tempo-agnostic, comprehensive engineering specification for each of the 40 PAS drum rudiments. These specs serve as the authoritative reference for the SOUSA project, and will later drive a migration of the existing YAML definitions in `dataset_gen/rudiments/definitions/`.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Timing model | Beat fractions only | Tempo agnostic — no ms values. All timing as fractions of a beat. |
| Spec depth | Comprehensive reference | Structural + dynamics + physical model + pedagogy + notation |
| Format | One markdown file per rudiment | Human-readable, versionable, individually updatable |
| Approach | 7-layer spec | Layered from abstract to concrete; each layer independently useful |
| Research | Web-verified | Every rudiment cross-checked against online PAS references |
| Relationship to YAMLs | Authoritative source → future migration | Specs become the source of truth; YAMLs will be regenerated from them |

## File Structure

```
docs/specs/rudiments/
├── _conventions.md              # Shared definitions (DRY)
├── _index.md                    # Master index with summary table
├── 01_single_stroke_roll.md
├── 02_single_stroke_four.md
├── ...
├── 40_triple_ratamacue.md
└── _template.md                 # Blank template for reference
```

## Spec Template (7 Layers)

Each rudiment spec follows this structure:

### Layer 1: Identity

| Field | Description |
|-------|-------------|
| PAS Number | Official number (1-40) |
| Name | Official PAS name |
| Category | roll / diddle / flam / drag |
| Family | Grouping (e.g., "paradiddle family", "stroke roll family") |
| Composed Of | Primitive components (e.g., "singles + diddle") |
| Related | Links to related rudiments |

### Layer 2: Rhythmic Structure

- **Time signature** assumed for the pattern
- **Base subdivision** (quarter, eighth, triplet, sixteenth, sextuplet, thirtysecond)
- **Cycle length** in beats
- **Strokes per cycle** count
- **Stroke grid table**: each stroke's position as a beat fraction

Example grid:

| # | Position (beats) | Grid Slot |
|---|------------------|-----------|
| 1 | 0                | 1         |
| 2 | 1/4              | 1 e       |
| 3 | 1/2              | 1 &       |
| 4 | 3/4              | 1 a       |

### Layer 3: Sticking & Articulation

- **Sticking sequence** in standard notation (R L R R | L R L L)
- **Stroke detail table**: hand, stroke type, ornament specifics
- **Ornament timing** in beat fractions:
  - Flam grace note: offset from primary in beat fractions
  - Drag grace notes: two offsets from primary in beat fractions
  - Diddle strokes: each occupies a fraction of the beat

### Layer 4: Dynamics Model

- **Velocity ratio table**: each stroke type's velocity relative to accent = 1.0
  - accent: 1.0
  - tap: 0.65–0.77
  - grace (flam): 0.50–0.70
  - grace (drag): 0.45–0.65
  - diddle pos 1: matches parent
  - diddle pos 2: 0.90–0.98 of pos 1
  - buzz: 0.60–0.80 per bounce, decaying
- **Accent pattern** visualization

### Layer 5: Physical / Kinesthetic

- **Stroke height classes** per stroke (full / half / low / tap)
- **Motion types** (wrist + arm / wrist / fingers / rebound)
- **Tempo-dependent constraints**:
  - How technique changes at tempo extremes
  - Where diddles transition from open to closed
  - Grace note compression behavior

### Layer 6: Variations & Pedagogy

- **Common variations** (NARD vs PAS, regional differences)
- **Teaching progression** (prerequisites, what this builds toward)
- **Relationship map** (how this connects to other rudiments)
- **Historical notes** where relevant

### Layer 7: Notation

- **Stick notation** (text-based R/L with accent markers)
- **Grid visualization** (beat grid with hand, type, accent aligned)

## Conventions Document (_conventions.md)

Defines once, referenced by all 40 specs:

- **Beat fraction notation**: always simplified (1/4 not 2/8)
- **Velocity ratio scale**: accent = 1.0 as reference point
- **Stroke height classes**: full (12"+ above head), half (6-12"), low (2-6"), tap (<2")
- **Motion types**: wrist + arm, wrist, fingers, rebound
- **Abbreviations**: A=accent, t=tap, g=grace, d1/d2=diddle pos 1/2, b=buzz
- **Grid slot naming**: standard counting (1 e & a for sixteenths, 1 & a for triplets)

## Research Methodology

For each rudiment:

1. **Verify sticking** against PAS standard references
2. **Compute rhythmic grid** — place each stroke on beat-fraction positions
3. **Define ornament timing** in beat fractions (replacing current ms-based values)
4. **Document dynamics** — velocity ratios from drumming pedagogy
5. **Describe physical model** — stroke heights and motion types
6. **Catalog variations** — NARD vs PAS, hybrid forms
7. **Map relationships** — compositional relationships between rudiments

Web research will be used to verify every rudiment, with extra attention to:
- Multiple bounce roll (buzz mechanics)
- Counted stroke rolls (5, 6, 7, 9, 10, 11, 13, 15, 17 — stroke count vs notation)
- Compound rudiments: Lesson 25, flam drag, pataflafla, Swiss army triplet
- Ratamacues (single, double, triple — the drag + triplet interaction)

## Implementation Batches

| Batch | Rudiments | Count | Notes |
|-------|-----------|-------|-------|
| 0 | Conventions + template + index | 3 files | Foundation files |
| 1 | Roll rudiments (#1-15) | 15 | Includes buzz roll and 9 counted stroke rolls |
| 2 | Paradiddle rudiments (#16-19) | 4 | Straightforward diddle patterns |
| 3 | Flam rudiments (#20-30) | 11 | Complex compounds (flam drag, pataflafla, Swiss army) |
| 4 | Drag rudiments (#31-40) | 10 | Ratamacues, Lesson 25, dragadiddle |

Total: 43 files (40 specs + conventions + template + index)

## Success Criteria

- All 40 PAS rudiments have a complete spec covering all 7 layers
- All timing expressed in beat fractions (zero ms values)
- Every rudiment cross-referenced against PAS standard via web research
- Consistent structure across all 40 specs
- Conventions document keeps specs DRY
- Specs contain sufficient detail to regenerate the existing YAML definitions
