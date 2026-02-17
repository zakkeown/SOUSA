# Rudiment Engineering Specs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Research and write tempo-agnostic engineering specifications for all 40 PAS drum rudiments, organized as individual markdown docs with a shared conventions file.

**Architecture:** Each rudiment gets a 7-layer markdown spec (identity, rhythmic structure, sticking & articulation, dynamics, physical model, pedagogy, notation). A conventions file defines shared vocabulary. An index file links everything. Every rudiment is web-verified against PAS references before writing.

**Tech Stack:** Markdown files in `docs/specs/rudiments/`, web research for verification, git for version control.

**Design doc:** `docs/plans/2026-02-17-rudiment-engineering-specs-design.md`

**Key references:**
- [PAS Official Rudiments](https://pas.org/rudiments/)
- [PAS Rudiments PDF](https://pas.org/wp-content/uploads/2024/04/pas-rudiments.pdf)
- [Vic Firth 40 Essential Rudiments](https://ae.vicfirth.com/education/40-essential-rudiments/)
- [UAB 40 PAS Rudiments PDF](https://www.uab.edu/cas/uabbands/images/documents/uab-percussion/studio-resources/40-pas-rudiments.pdf)
- [drumming.com 40 Rudiments](https://www.drumming.com/drum-lesson/40-essential-drum-rudiments)
- Existing YAML definitions: `dataset_gen/rudiments/definitions/`
- Existing schema: `dataset_gen/rudiments/schema.py`
- Existing reference: `docs/reference/rudiment-schema.md`

---

## Task 0: Create Foundation Files

**Files:**
- Create: `docs/specs/rudiments/_conventions.md`
- Create: `docs/specs/rudiments/_template.md`
- Create: `docs/specs/rudiments/_index.md`

**Step 1: Create the directory**

```bash
mkdir -p docs/specs/rudiments
```

**Step 2: Write `_conventions.md`**

This file defines all shared vocabulary used across the 40 specs. Write once, reference everywhere.

```markdown
# Rudiment Spec Conventions

This document defines shared vocabulary and notation for all 40 PAS rudiment engineering specs.

## Beat Fraction Notation

All timing is expressed as fractions of a beat. No millisecond or absolute time values.

- Always use simplified fractions: `1/4` not `2/8`
- Position 0 = the downbeat
- Negative fractions = before the beat (grace notes)
- The beat unit is always the quarter note unless stated otherwise

## Grid Slot Naming

| Subdivision | Slots per Beat | Names |
|------------|----------------|-------|
| Quarter    | 1 | 1 |
| Eighth     | 2 | 1, & |
| Triplet    | 3 | 1, &, a |
| Sixteenth  | 4 | 1, e, &, a |
| Sextuplet  | 6 | 1, la, li, &, la, li |
| 32nd       | 8 | 1, &, e, &, a, &, e, & |

## Stroke Types

| Abbreviation | Type | Description |
|-------------|------|-------------|
| A | accent | Emphasized stroke, full arm/wrist motion |
| t | tap | Unaccented single stroke |
| g | grace | Grace note (flam or drag ornament) |
| d1 | diddle pos 1 | First stroke of a double (same hand) |
| d2 | diddle pos 2 | Second stroke of a double (same hand) |
| b | buzz | Press/buzz roll stroke with multiple bounces |

## Velocity Ratio Scale

All velocities expressed relative to accent = 1.0:

| Stroke Type | Ratio Range | Notes |
|------------|-------------|-------|
| accent | 1.0 | Reference level |
| tap | 0.65-0.77 | Unaccented single stroke |
| grace (flam) | 0.50-0.70 | Single grace note before primary |
| grace (drag) | 0.45-0.65 | Each of the two drag grace notes |
| diddle pos 1 | matches parent | Same as the accent/tap it belongs to |
| diddle pos 2 | 0.90-0.98x pos 1 | Slight decay on second bounce |
| buzz bounce | 0.60-0.80x primary | Per-bounce, decaying |

## Stroke Height Classes

| Class | Stick Height | Typical Use |
|-------|-------------|-------------|
| full | 12"+ above head | Accents, loud strokes |
| half | 6-12" | Medium dynamics |
| low | 2-6" | Taps, unaccented strokes |
| tap | <2" | Grace notes, buzz bounces |

## Motion Types

| Type | Description | Typical Stroke |
|------|-------------|---------------|
| wrist + arm | Full Moeller whip motion | Accents at moderate tempos |
| wrist | Primary motion from wrist | Most strokes at moderate tempos |
| fingers | Finger control, minimal wrist | Grace notes, fast passages, diddle pos 2 |
| rebound | Controlled bounce off head | Buzz rolls, fast diddles |

## Categories

The 40 PAS rudiments are organized into four categories:

| Category | PAS Numbers | Description |
|----------|------------|-------------|
| Roll | 1-15 | Single strokes, buzz/multiple bounce, double strokes, counted rolls |
| Diddle | 16-19 | Paradiddle family (single + double stroke combinations) |
| Flam | 20-30 | Patterns featuring single grace note ornaments |
| Drag | 31-40 | Patterns featuring double grace note ornaments |

## Ornament Timing Conventions

### Flam Grace Notes
- Expressed as a negative beat fraction from the primary stroke
- Standard offset: **-1/32 beat** (at moderate tempos)
- Range: -1/64 to -1/16 depending on tempo and style
- Grace note always played by the opposite hand from the primary

### Drag Grace Notes
- Two grace notes preceding the primary stroke
- First grace: **-1/16 beat** before primary
- Second grace: **-1/32 beat** before primary
- Both grace notes played by the opposite hand from the primary
- Grace notes form a quick diddle (same hand, two bounces)

### Diddle Timing
- Two strokes sharing a single grid position's duration
- Each diddle stroke occupies **1/2 of the parent grid slot**
- At slow tempos: open (distinguishable as two separate strokes)
- At fast tempos: closed (nearly blurred together)

## Notation Conventions

### Stick Notation
```
R  = right hand stroke
L  = left hand stroke
>  = accent marker (above the stroke)
(g)= grace note prefix
|  = bar/group separator
```

### Flam Notation
```
lR = left grace + right primary (right-hand flam)
rL = right grace + left primary (left-hand flam)
```

### Drag Notation
```
llR = left-left grace + right primary (right-hand drag)
rrL = right-right grace + left primary (left-hand drag)
```
```

**Step 3: Write `_template.md`**

Write the blank template that each rudiment spec follows. Include all 7 layers with placeholder text and instructions.

```markdown
# PAS #NN: [Rudiment Name]

## 1. Identity

| Field | Value |
|-------|-------|
| PAS Number | NN |
| Name | [Official PAS name] |
| Category | [roll / diddle / flam / drag] |
| Family | [e.g., "single stroke family", "paradiddle family"] |
| Composed Of | [primitive components, e.g., "singles + diddle"] |
| Related | [links to related rudiment specs] |
| NARD Original | [Yes/No — was this in the original 26?] |

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| Time Signature | [e.g., 4/4] |
| Base Subdivision | [e.g., sixteenth] |
| Cycle Length | [N beats] |
| Strokes Per Cycle | [N (counting grace notes separately)] |
| Primary Strokes Per Cycle | [N (excluding grace notes)] |

### Stroke Grid

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | 0 | 1 | R | accent |
| 2 | 1/4 | 1 e | L | tap |
| … | … | … | … | … |

> **Note:** Grace notes have negative offsets from their parent stroke's position.
> Diddle strokes share a grid slot, each occupying half the subdivision duration.

## 3. Sticking & Articulation

### Sticking Sequence

```
[Full sticking with hand labels, e.g.: R L R R | L R L L]
[Accent line, e.g.:                   > . . . | > . . .]
```

### Stroke Detail Table

| # | Hand | Type | Detail |
|---|------|------|--------|
| 1 | R | accent | — |
| 2 | L | tap | — |
| … | … | … | … |

### Ornament Timing

[If the rudiment contains flams, drags, or diddles, specify beat-fraction timing here]

- **[Ornament type]**: [offset in beat fractions] from primary

## 4. Dynamics Model

### Velocity Ratios

| Stroke | Ratio (accent = 1.0) | Notes |
|--------|---------------------|-------|
| [type] | [ratio] | [context] |

### Accent Pattern

```
[Visual accent pattern, e.g.: > . . . | > . . .]
```

### Dynamic Contour

[Describe any velocity shaping across the cycle — e.g., "slight crescendo into accent"]

## 5. Physical / Kinesthetic

### Stroke Map

| # | Stroke | Height Class | Motion Type |
|---|--------|-------------|-------------|
| 1 | accent | full | wrist + arm |
| 2 | tap | low | wrist |
| … | … | … | … |

### Tempo-Dependent Behavior

| Tempo Range | Technique Adaptation |
|------------|---------------------|
| < 80 BPM | [description] |
| 80-140 BPM | [description] |
| 140-180 BPM | [description] |
| > 180 BPM | [description] |

## 6. Variations & Pedagogy

### Common Variations

[List known variations, NARD vs PAS differences, regional interpretations]

### Prerequisites

[What rudiments should be mastered before learning this one]

### Builds Toward

[What more complex rudiments this one feeds into]

### Teaching Notes

[Pedagogical tips, common mistakes, practice strategies]

### Historical Context

[Origin, NARD history, when added to PAS list if applicable]

## 7. Notation

### Stick Notation

```
[Full text-based stick notation]
```

### Grid Visualization

```
Beat: [beat grid labels]
Hand: [R/L per position]
Type: [A/t/g/d1/d2/b per position]
```
```

**Step 4: Write `_index.md` skeleton**

Write the index with all 40 entries linked but marked as "pending". This will be updated as each spec is completed.

**Step 5: Commit foundation files**

```bash
git add docs/specs/rudiments/_conventions.md docs/specs/rudiments/_template.md docs/specs/rudiments/_index.md
git commit -m "docs: add foundation files for rudiment engineering specs"
```

---

## Task 1: Single Stroke Rudiments (#1-3)

**Files:**
- Create: `docs/specs/rudiments/01_single_stroke_roll.md`
- Create: `docs/specs/rudiments/02_single_stroke_four.md`
- Create: `docs/specs/rudiments/03_single_stroke_seven.md`

**Research:** These are the simplest rudiments — alternating single strokes. Verify against PAS references:
- #1: Continuous alternating RLRL, sixteenth notes
- #2: Four-note grouping RLRL with accents on 1 and 3
- #3: Seven-note grouping RLRLRLR, often in a 7-note feel

**Step 1: Web research**

Search for each rudiment to verify:
- Exact sticking pattern
- Correct subdivision (sixteenths vs other)
- Accent placement
- Whether the cycle alternates starting hand

Cross-reference: `dataset_gen/rudiments/definitions/01_single_stroke_roll.yaml` through `03_single_stroke_seven.yaml`

**Step 2: Write specs**

For each rudiment, fill in all 7 layers of the template. Key details:

**#1 Single Stroke Roll:**
- Category: roll (single stroke sub-family)
- Pattern: R L R L R L R L (continuous alternation)
- Subdivision: sixteenth notes
- Cycle: 2 beats (8 strokes at sixteenth-note subdivision)
- Accents: typically none (all even) or accent on beat 1
- NARD original: Yes
- Physical: all strokes at same height class, pure alternation
- Composed of: nothing — this is the most primitive rudiment

**#2 Single Stroke Four:**
- Category: roll (single stroke sub-family)
- Pattern: R L R L (4 notes)
- Subdivision: sixteenth notes
- Cycle: 1 beat
- Accents: on stroke 1
- NARD original: No (added by PAS)
- Related: subset of #1

**#3 Single Stroke Seven:**
- Category: roll (single stroke sub-family)
- Pattern: R L R L R L R (7 notes)
- Subdivision: typically sixteenth notes, sometimes written as two groups
- Cycle: 2 beats (7 sixteenths + 1 sixteenth rest, or 7/8)
- Accents: on stroke 1
- NARD original: No (added by PAS)
- Related: extended version of #2

**Step 3: Verify completeness**

Check each spec has all 7 layers filled, no ms values, all timing in beat fractions.

**Step 4: Commit**

```bash
git add docs/specs/rudiments/01_single_stroke_roll.md docs/specs/rudiments/02_single_stroke_four.md docs/specs/rudiments/03_single_stroke_seven.md
git commit -m "docs: add specs for single stroke rudiments (#1-3)"
```

---

## Task 2: Multiple Bounce & Triple Stroke (#4-5)

**Files:**
- Create: `docs/specs/rudiments/04_multiple_bounce_roll.md`
- Create: `docs/specs/rudiments/05_triple_stroke_roll.md`

**Research:** These need extra attention:
- #4 is the buzz/press roll — unique mechanics (multiple bounces per primary)
- #5 is three strokes per hand (RRR LLL)

**Step 1: Web research**

Search specifically for:
- Multiple bounce roll: how many bounces per primary (varies by player/tempo), how buzz strokes interact, the press technique
- Triple stroke roll: exact stroke grouping, whether written as sextuplets or triplets, accent pattern

Cross-reference: `dataset_gen/rudiments/definitions/04_multiple_bounce_roll.yaml` and `05_triple_stroke_roll.yaml`

**Step 2: Write specs**

**#4 Multiple Bounce Roll (Buzz Roll):**
- Category: roll (multiple bounce sub-family)
- Pattern: R(buzz) L(buzz) R(buzz) L(buzz) — each primary has 3-8 bounces
- Subdivision: the primary strokes are eighth or sixteenth; bounces fill the space
- Unique layer 5 content: press technique, rebound control, maintaining even buzz density
- NARD original: Yes (as "Long Roll" in closed form)

**#5 Triple Stroke Roll:**
- Category: roll
- Pattern: RRR LLL RRR LLL
- Subdivision: sextuplets (6 per beat) or sixteenth-note triplets
- Cycle: 2 beats (12 strokes)
- NARD original: No (added by PAS)
- Physical: requires finger control for the third stroke in each group

**Step 3: Verify completeness**

**Step 4: Commit**

```bash
git add docs/specs/rudiments/04_multiple_bounce_roll.md docs/specs/rudiments/05_triple_stroke_roll.md
git commit -m "docs: add specs for multiple bounce and triple stroke (#4-5)"
```

---

## Task 3: Double Stroke Open Roll (#6)

**Files:**
- Create: `docs/specs/rudiments/06_double_stroke_open_roll.md`

**Research:** The double stroke open roll is foundational — it's the basis for all counted stroke rolls (#7-15).

**Step 1: Web research**

Verify:
- Pattern: RR LL RR LL (continuous diddles)
- The "open" vs "closed" distinction
- How it relates to the counted stroke rolls
- Diddle evenness requirements

Cross-reference: `dataset_gen/rudiments/definitions/06_double_stroke_open_roll.yaml`

**Step 2: Write spec**

**#6 Double Stroke Open Roll:**
- Category: roll (double stroke sub-family)
- Pattern: RR LL RR LL (continuous alternating diddles)
- Subdivision: sixteenths (each diddle pair occupies one eighth note)
- Cycle: 2 beats (8 strokes, 4 diddle pairs)
- NARD original: Yes (the "Daddy" of rolls)
- Key dynamics: diddle pos 2 velocity decay, evenness
- Key physical: open vs closed transition as tempo increases
- Composed of: diddles only — the simplest diddle rudiment
- Builds toward: every counted stroke roll (#7-15), paradiddles

**Step 3: Verify completeness**

**Step 4: Commit**

```bash
git add docs/specs/rudiments/06_double_stroke_open_roll.md
git commit -m "docs: add spec for double stroke open roll (#6)"
```

---

## Task 4: Counted Stroke Rolls (#7-15)

**Files:**
- Create: `docs/specs/rudiments/07_five_stroke_roll.md`
- Create: `docs/specs/rudiments/08_six_stroke_roll.md`
- Create: `docs/specs/rudiments/09_seven_stroke_roll.md`
- Create: `docs/specs/rudiments/10_nine_stroke_roll.md`
- Create: `docs/specs/rudiments/11_ten_stroke_roll.md`
- Create: `docs/specs/rudiments/12_eleven_stroke_roll.md`
- Create: `docs/specs/rudiments/13_thirteen_stroke_roll.md`
- Create: `docs/specs/rudiments/14_fifteen_stroke_roll.md`
- Create: `docs/specs/rudiments/15_seventeen_stroke_roll.md`

**Research:** These are structurally similar (N diddles + accent). The key variable is stroke count, which determines:
- Number of diddle pairs
- Cycle length in beats
- Where the ending accent falls
- Whether the roll starts and ends on the same hand

**CRITICAL RESEARCH POINT:** The stroke counts in the names refer to total strokes including the final accent. Verify the exact breakdown:
- 5-stroke: 2 diddles + accent (RRLL + R) = 5 strokes
- 6-stroke: written as R L RR LL (often taught as accent-accent-diddle-diddle, a "reverse" roll)
- 7-stroke: 3 diddles + accent (RRLLRR + L) = 7 strokes
- 9-stroke: 4 diddles + accent = 9 strokes
- 10-stroke: 4 diddles + 2 accents (or 5 diddles?) — verify this carefully
- 11-stroke: 5 diddles + accent = 11 strokes
- 13-stroke: 6 diddles + accent = 13 strokes
- 15-stroke: 7 diddles + accent = 15 strokes
- 17-stroke: 8 diddles + accent = 17 strokes

The 6-stroke roll and 10-stroke roll have non-standard interpretations that need extra verification.

**Step 1: Web research**

Search for each counted stroke roll, focusing on:
- Exact stroke count decomposition (diddles + accents)
- Rhythmic context (which note value does each roll replace?)
- Starting hand alternation rules
- Common performance contexts

**Step 2: Write specs (batch)**

These share almost identical dynamics, physical, and pedagogy layers. Write the 5-stroke roll first as the "canonical" counted roll, then adapt for the others, noting what differs:
- Stroke count and diddle count
- Cycle length
- Rhythmic placement (quarter note, dotted quarter, half note, etc.)
- Starting hand

**Step 3: Verify completeness across all 9**

Ensure consistency: all use the same velocity ratio conventions, same ornament timing for diddles, same physical model.

**Step 4: Commit**

```bash
git add docs/specs/rudiments/07_five_stroke_roll.md through docs/specs/rudiments/15_seventeen_stroke_roll.md
git commit -m "docs: add specs for counted stroke rolls (#7-15)"
```

---

## Task 5: Paradiddle Rudiments (#16-19)

**Files:**
- Create: `docs/specs/rudiments/16_single_paradiddle.md`
- Create: `docs/specs/rudiments/17_double_paradiddle.md`
- Create: `docs/specs/rudiments/18_triple_paradiddle.md`
- Create: `docs/specs/rudiments/19_paradiddle_diddle.md`

**Research:** The paradiddle family combines single strokes with diddles.

**Step 1: Web research**

Verify for each:
- Exact sticking pattern and accent placement
- Relationship to each other (single → double → triple as extensions)
- Paradiddle-diddle's unique structure (paradiddle + extra diddle vs. rearranged strokes)

**Step 2: Write specs**

**#16 Single Paradiddle:**
- Pattern: RLRR LRLL (accents on 1 and 5)
- Subdivision: sixteenths
- Cycle: 2 beats (8 strokes)
- Composed of: 2 singles + 1 diddle, mirrored
- NARD original: Yes

**#17 Double Paradiddle:**
- Pattern: RLRLRR LRLRLL (accents on 1 and 7)
- Subdivision: sixteenths
- Cycle: 3 beats (12 strokes)
- Composed of: 4 singles + 1 diddle, mirrored
- NARD original: Yes

**#18 Triple Paradiddle:**
- Pattern: RLRLRLRR LRLRLRLL (accents on 1 and 9)
- Subdivision: sixteenths
- Cycle: 4 beats (16 strokes)
- Composed of: 6 singles + 1 diddle, mirrored
- NARD original: No (added by PAS)

**#19 Paradiddle-Diddle:**
- Pattern: RLRRLL LRLLRR (accent on 1, then mirror)
- Subdivision: sixteenths or sextuplets (6-note grouping)
- Cycle: varies by interpretation
- Composed of: 2 singles + 2 diddles
- NARD original: Yes

**Step 3: Verify completeness**

**Step 4: Commit**

```bash
git add docs/specs/rudiments/16_single_paradiddle.md through docs/specs/rudiments/19_paradiddle_diddle.md
git commit -m "docs: add specs for paradiddle rudiments (#16-19)"
```

---

## Task 6: Basic Flam Rudiments (#20-22)

**Files:**
- Create: `docs/specs/rudiments/20_flam.md`
- Create: `docs/specs/rudiments/21_flam_accent.md`
- Create: `docs/specs/rudiments/22_flam_tap.md`

**Research:** These are the foundational flam patterns.

**Step 1: Web research**

Verify:
- #20: Basic flam — alternating lR rL, subdivision, grace note timing
- #21: Flam accent — triplet-based pattern with flam on beat 1 of each group
- #22: Flam tap — flam + tap, alternating hands

**Step 2: Write specs**

**#20 Flam:**
- Pattern: lR rL (alternating flams)
- Subdivision: eighth notes (flam on each eighth)
- Cycle: 1 beat (2 flams)
- Grace note timing: -1/32 beat before primary
- NARD original: Yes
- This is the foundational ornament for all flam rudiments

**#21 Flam Accent:**
- Pattern: lR L R | rL R L (flam on 1 of each triplet group)
- Subdivision: triplets
- Cycle: 2 beats (6 strokes + 2 grace notes)
- NARD original: Yes
- Unique: triplet subdivision, most rudiments are in sixteenths

**#22 Flam Tap:**
- Pattern: lR R | rL L (flam + tap, alternating)
- Subdivision: eighth notes (or sixteenths depending on context)
- Cycle: 1 beat
- NARD original: Yes

**Step 3: Verify completeness**

**Step 4: Commit**

```bash
git add docs/specs/rudiments/20_flam.md docs/specs/rudiments/21_flam_accent.md docs/specs/rudiments/22_flam_tap.md
git commit -m "docs: add specs for basic flam rudiments (#20-22)"
```

---

## Task 7: Compound Flam Rudiments (#23-26)

**Files:**
- Create: `docs/specs/rudiments/23_flamacue.md`
- Create: `docs/specs/rudiments/24_flam_paradiddle.md`
- Create: `docs/specs/rudiments/25_single_flammed_mill.md`
- Create: `docs/specs/rudiments/26_flam_paradiddle_diddle.md`

**Research:** These combine flams with other rudiment types. Extra verification needed.

**Step 1: Web research**

These need careful verification:
- #23 Flamacue: flam + 4 notes, specific accent on the last note. Search for exact sticking: lR L R L R(accent), then mirror.
- #24 Flam Paradiddle: flam + paradiddle. Sticking: lR L R R | rL R L L
- #25 Single Flammed Mill: a "mill" is a series of strokes on the same hand. The flammed mill adds a flam. Search for exact pattern: lR R L | rL L R (flam on 1, then same-hand double, then opposite single)
- #26 Flam Paradiddle-Diddle: flam + paradiddle-diddle. Sticking: lR L R R L L | rL R L L R R

**Step 2: Write specs**

Pay special attention to:
- Flamacue: the accent on the last stroke before the flam repeats is a key identifying feature
- Single Flammed Mill: this involves a "mill stroke" (3 strokes, same-hand emphasis) with flam added
- Flam Paradiddle-Diddle: 12 strokes per full cycle (two mirrored halves)

**Step 3: Verify completeness**

**Step 4: Commit**

```bash
git add docs/specs/rudiments/23_flamacue.md through docs/specs/rudiments/26_flam_paradiddle_diddle.md
git commit -m "docs: add specs for compound flam rudiments (#23-26)"
```

---

## Task 8: Advanced Flam Rudiments (#27-30)

**Files:**
- Create: `docs/specs/rudiments/27_pataflafla.md`
- Create: `docs/specs/rudiments/28_swiss_army_triplet.md`
- Create: `docs/specs/rudiments/29_inverted_flam_tap.md`
- Create: `docs/specs/rudiments/30_flam_drag.md`

**Research:** These are the most complex flam rudiments. All need careful web verification.

**Step 1: Web research**

- #27 Pataflafla: "pa-ta-fla-fla" = single, single, flam, flam. Four sixteenth notes with flams on 3rd and 4th. Back-to-back flams is the challenge. Pattern: R L lR rL | L R rL lR
- #28 Swiss Army Triplet: triplet-based, flam on first of each triplet group, specific sticking. Pattern: lR R L | rL L R (or with a specific crossover pattern). Needs verification — some sources show a crossover flam.
- #29 Inverted Flam Tap: differs from Flam Tap (#22) — the grace note comes on the upstroke instead. Pattern: lR L | rL R (the "inversion" swaps where the flam sits relative to the tap).
- #30 Flam Drag: combines a flam and a drag in the same pattern. Pattern: lR L llR | rL R rrL. This is a cross-category rudiment (flam + drag).

**Step 2: Write specs**

Key attention areas:
- Pataflafla: the back-to-back flams (consecutive grace notes) require special dynamics modeling
- Swiss Army Triplet: triplet subdivision (like flam accent), verify exact flam placement
- Inverted Flam Tap: clearly distinguish from Flam Tap (#22) — document what "inverted" means
- Flam Drag: dual ornament types in one rudiment, needs both flam and drag params

**Step 3: Verify completeness**

**Step 4: Commit**

```bash
git add docs/specs/rudiments/27_pataflafla.md through docs/specs/rudiments/30_flam_drag.md
git commit -m "docs: add specs for advanced flam rudiments (#27-30)"
```

---

## Task 9: Basic Drag Rudiments (#31-33)

**Files:**
- Create: `docs/specs/rudiments/31_drag.md`
- Create: `docs/specs/rudiments/32_single_drag_tap.md`
- Create: `docs/specs/rudiments/33_double_drag_tap.md`

**Research:** The drag family. A drag is two grace notes (a quick diddle) before a primary stroke.

**Step 1: Web research**

- #31 Drag (Ruff): the basic drag. Pattern: llR rrL. Two grace notes + primary, alternating.
- #32 Single Drag Tap: drag + single tap. Pattern: llR R | rrL L
- #33 Double Drag Tap: two drags + single tap. Pattern: llR llR R | rrL rrL L

**Step 2: Write specs**

**#31 Drag:**
- The foundational ornament for all drag rudiments
- Subdivision: eighth notes
- Grace note timing: first grace at -1/16, second at -1/32 from primary
- NARD original: Yes (as "Ruff")
- Relationship: this is to drag rudiments what the flam (#20) is to flam rudiments

**#32 Single Drag Tap:**
- Drag + tap, alternating. NARD original: Yes
- Subdivision: sixteenth notes or eighth notes

**#33 Double Drag Tap:**
- Two drags + tap, alternating. NARD original: Yes
- Verify exact rhythmic placement of the two drags

**Step 3: Verify completeness**

**Step 4: Commit**

```bash
git add docs/specs/rudiments/31_drag.md docs/specs/rudiments/32_single_drag_tap.md docs/specs/rudiments/33_double_drag_tap.md
git commit -m "docs: add specs for basic drag rudiments (#31-33)"
```

---

## Task 10: Compound Drag Rudiments (#34-37)

**Files:**
- Create: `docs/specs/rudiments/34_lesson_25.md`
- Create: `docs/specs/rudiments/35_single_dragadiddle.md`
- Create: `docs/specs/rudiments/36_drag_paradiddle_1.md`
- Create: `docs/specs/rudiments/37_drag_paradiddle_2.md`

**Research:** These combine drags with other elements. Lesson 25 needs special attention.

**Step 1: Web research**

- #34 Lesson 25: named after the 25th lesson in an 1869 drumming manual. Pattern involves two drags and a flam. This is a cross-category rudiment. The existing YAML shows: drag R, drag L, flam L | drag L, drag R, flam R. Verify this carefully.
- #35 Single Dragadiddle: drag + paradiddle. Pattern: llR L R R | rrL R L L (drag replaces the first stroke of a paradiddle).
- #36 Drag Paradiddle #1: a specific drag-paradiddle combination. Pattern: R llR L R R | L rrL R L L (tap + drag + paradiddle).
- #37 Drag Paradiddle #2: variant of #36. Pattern: R llR L L R R | L rrL R R L L (different from #1). Verify the exact difference.

**Step 2: Write specs**

Key attention:
- Lesson 25: document the historical origin (1869 manual), the cross-category nature (drag + flam)
- Drag Paradiddle #1 vs #2: clearly document what distinguishes them

**Step 3: Verify completeness**

**Step 4: Commit**

```bash
git add docs/specs/rudiments/34_lesson_25.md through docs/specs/rudiments/37_drag_paradiddle_2.md
git commit -m "docs: add specs for compound drag rudiments (#34-37)"
```

---

## Task 11: Ratamacue Rudiments (#38-40)

**Files:**
- Create: `docs/specs/rudiments/38_single_ratamacue.md`
- Create: `docs/specs/rudiments/39_double_ratamacue.md`
- Create: `docs/specs/rudiments/40_triple_ratamacue.md`

**Research:** Ratamacues combine drags with triplet-feel single strokes and an accent.

**Step 1: Web research**

- #38 Single Ratamacue: one drag + triplet + accent. Pattern: llR L R L(accent). The drag precedes a triplet group ending with an accent.
- #39 Double Ratamacue: two drags + triplet + accent. Pattern: llR llR L R L(accent).
- #40 Triple Ratamacue: three drags + triplet + accent. Pattern: llR llR llR L R L(accent).

Verify:
- The triplet feel (are the post-drag strokes in triplet subdivision?)
- Accent placement (always on the last stroke?)
- How the drags interact with the triplet grid

**Step 2: Write specs**

These are structurally parallel — the only variable is how many drags precede the triplet. Write the single ratamacue first, then adapt for double and triple.

**Step 3: Verify completeness**

**Step 4: Commit**

```bash
git add docs/specs/rudiments/38_single_ratamacue.md docs/specs/rudiments/39_double_ratamacue.md docs/specs/rudiments/40_triple_ratamacue.md
git commit -m "docs: add specs for ratamacue rudiments (#38-40)"
```

---

## Task 12: Final Review & Index Completion

**Files:**
- Modify: `docs/specs/rudiments/_index.md` (update all entries from "pending" to linked)

**Step 1: Review all 40 specs for consistency**

Check across all files:
- All 7 layers present in every spec
- No ms values anywhere (only beat fractions)
- Velocity ratios consistent with conventions document
- Stroke height classes used consistently
- All ornament timing uses beat fractions
- Cross-references between related rudiments are bidirectional
- Grid visualizations align with stroke detail tables

**Step 2: Complete the index**

Update `_index.md` with:
- All 40 entries linked to their spec files
- Summary table with: PAS #, Name, Category, Subdivision, Cycle Length, Strokes Per Cycle
- Category grouping headers

**Step 3: Final commit**

```bash
git add docs/specs/rudiments/_index.md
git commit -m "docs: complete rudiment spec index and final review"
```

---

## Summary

| Task | Description | Files | Dependencies |
|------|-------------|-------|-------------|
| 0 | Foundation files | 3 | None |
| 1 | Single stroke rudiments (#1-3) | 3 | Task 0 |
| 2 | Multiple bounce & triple (#4-5) | 2 | Task 0 |
| 3 | Double stroke open roll (#6) | 1 | Task 0 |
| 4 | Counted stroke rolls (#7-15) | 9 | Task 3 (reference #6) |
| 5 | Paradiddles (#16-19) | 4 | Task 0 |
| 6 | Basic flams (#20-22) | 3 | Task 0 |
| 7 | Compound flams (#23-26) | 4 | Task 6 |
| 8 | Advanced flams (#27-30) | 4 | Task 6 |
| 9 | Basic drags (#31-33) | 3 | Task 0 |
| 10 | Compound drags (#34-37) | 4 | Task 9 |
| 11 | Ratamacues (#38-40) | 3 | Task 9 |
| 12 | Final review & index | 1 | All above |

**Parallelizable groups** (no dependencies between these):
- Tasks 1, 2, 3, 5, 6, 9 can all run in parallel (all depend only on Task 0)
- Tasks 4, 7, 8, 10, 11 can run after their prerequisites complete
- Task 12 must run last

**Total files:** 43 (40 specs + 3 foundation)
