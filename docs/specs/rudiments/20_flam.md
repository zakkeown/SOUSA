# PAS #20: Flam

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 20 |
| **Name** | Flam |
| **Category** | flam |
| **Family** | basic flam |
| **Composed Of** | None -- this is the primitive flam ornament, a single grace note preceding a primary stroke |
| **Related** | #21 Flam Accent (flam + triplet taps), #22 Flam Tap (flam + tap, double-stroke framework), #31 Drag (double grace note counterpart) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | eighth |
| **Cycle Length** | 1 beat |
| **Strokes Per Cycle** | 4 (2 grace notes + 2 primary strokes) |
| **Primary Strokes Per Cycle** | 2 |

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/32 | (grace before 1) | L | g |
| 2 | 0 | 1 | R | A |
| 3 | 1/2 - 1/32 | (grace before 1&) | R | g |
| 4 | 1/2 | 1& | L | A |

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>       >
lR      rL
```

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | grace | #2 | Grace note preceding right-hand primary |
| 2 | R | accent | primary | n/a | Right-hand flam primary stroke |
| 3 | R | grace | grace | #4 | Grace note preceding left-hand primary |
| 4 | L | accent | primary | n/a | Left-hand flam primary stroke |

### Ornament Timing

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- The grace note and primary are heard as a single unified "fatter" note, not two distinct notes
- The grace should "kiss" the drumhead just before the primary lands

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (primary) | 1.0 | Full stroke, the dominant sound of the flam |
| grace (flam) | 0.50 - 0.70 | Soft lead-in; should be noticeably quieter than the primary |

The dynamic contrast between grace and primary is fundamental to the flam's character. If the grace note is too loud, the two strokes sound like a double stop (flat flam) rather than the characteristic "breadth" of a well-executed flam. If too quiet, the grace becomes inaudible and the ornament effect is lost.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>     >
lR    rL
```

Both primary strokes are accented. The grace notes are inherently soft and do not receive accent treatment.

### Dynamic Contour

The flam has a symmetrical dynamic contour: each half-beat contains one soft-loud pair (grace-primary). The pattern is identical on both halves but alternates between hands. The dynamic shape is best described as two matched impulses per beat, each preceded by a soft onset. There is no crescendo or decrescendo across the cycle; the pattern repeats identically.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace | tap (< 2") | fingers |
| 2 | R | accent | full (12"+) | wrist + arm |
| 3 | R | grace | tap (< 2") | fingers |
| 4 | L | accent | full (12"+) | wrist + arm |

The physical essence of the flam is simultaneous preparation at two different stick heights: one hand low (grace) and the other high (primary). Both hands begin their downstroke at the same moment, but the low stick contacts the head first because it has less distance to travel, producing the characteristic grace-before-primary sound.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Wide separation between grace and primary. Grace note is clearly audible as a distinct soft stroke before the primary. Full wrist + arm motion for primary, deliberate finger placement for grace. The "gap" between grace and primary can approach 1/16 beat. |
| Moderate (100-140 BPM) | Standard flam spacing. Grace and primary are close together but distinguishable. Grace offset settles at approximately 1/32 beat. Primary uses wrist motion; grace is finger-controlled. |
| Fast (140-180 BPM) | Tight flam spacing approaching 1/64 beat. Grace and primary nearly simultaneous but still perceptibly ordered. Primary height drops to half class; grace remains at tap. The stick height differential decreases but must not collapse to a flat flam (double stop). |

---

## 6. Variations & Pedagogy

### Common Variations

- **Flat flam (undesirable)**: Grace and primary strike simultaneously, producing a double stop. This is a common beginner error, not a valid variation.
- **Open flam**: Exaggerated spacing between grace and primary for stylistic effect, common in field drumming and rudimental solos.
- **Closed flam**: Very tight grace-to-primary spacing, used at fast tempos and in orchestral contexts.
- **Grace note on the beat**: In some interpretations (particularly Swiss and Basel drumming), the grace note is placed on the beat and the primary follows. This is the inverse of the standard PAS interpretation.
- **Repeated same-hand flam**: All flams led by the same hand (all lR or all rL), used as an exercise but not the standard rudiment.

### Prerequisites

- #1 Single Stroke Roll -- basic hand alternation and stroke control
- Ability to play at different stick heights with control (height differentiation between grace and primary)

### Builds Toward

- #21 Flam Accent (flam + triplet taps)
- #22 Flam Tap (flam + tap in eighth-note framework)
- #23 Flamacue (flam + accent pattern in sixteenth notes)
- #24 Flam Paradiddle (flam + paradiddle)
- #25 Single Flammed Mill (flam + mill pattern)
- #27 Pataflafla (compound flam pattern)
- #29 Inverted Flam Tap (inverted version of flam tap)
- #30 Flam Drag (flam combined with drag)
- All flam-category rudiments (PAS #20-30) build directly on this primitive

### Teaching Notes

The flam is the most fundamental ornament in rudimental drumming and the basis of all flam-category rudiments. Despite its apparent simplicity, achieving consistent, well-balanced flams is a skill that requires extensive practice.

**Common mistakes:**
- Flat flams (double stops): both sticks striking simultaneously, producing a "clap" rather than a "flam" sound
- Inconsistent spacing: the gap between grace and primary varies from flam to flam
- Grace note too loud: the grace overpowers or competes with the primary, destroying the single-note illusion
- Grace note too quiet or absent: the stick is too low to produce an audible grace note
- Same-width flams on both hands: the non-dominant hand often produces wider or tighter flams than the dominant hand
- Rushing the grace: placing the grace too far ahead of the primary, creating a diddle-like sound

**Practice strategies:**
- Start with the "stick height" approach: hold one stick at tap height and the other at full height, then drop both simultaneously
- Practice each hand's flam separately (all lR, then all rL) before alternating
- Use a metronome and place the primary on the click; the grace should fall naturally just before
- Record and listen back, checking that left-lead and right-lead flams sound identical
- Practice at very slow tempos to develop the correct muscle memory for height differentiation

**What to listen for:**
- Each flam should sound like one "fat" note, not two separate notes
- Left-lead and right-lead flams should be indistinguishable in quality
- Consistent spacing from flam to flam
- Clear dynamic separation between grace (soft) and primary (full)

### Historical Context

The flam is one of the oldest and most fundamental drum ornaments, predating any formal rudiment codification. The word "flam" likely derives from the onomatopoeic representation of the sound it produces. The flam was included in the original NARD (National Association of Rudimental Drummers) standard 26 rudiments when NARD was founded in 1933, and its roots trace back through centuries of military field drumming in both European and American traditions. Charles Stewart Ashworth's *A New, Useful and Complete System of Drum Beating* (1812) includes flam patterns, and the ornament appears in even earlier European military manuals. The flam was retained as PAS #20 (first of the flam rudiments) when the Percussive Arts Society expanded the list to 40 in 1984. It serves as the primitive building block for all 11 flam-category rudiments (PAS #20-30), just as the single stroke (#1) is the primitive for all roll rudiments.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>           >
lR          rL
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1           &         |
Hand:    | (l)R        (r)L      |
Type:    |  g  A        g  A     |
Accent:  |     >           >     |
```

Grace notes shown in parentheses indicate they fall just before the grid position (offset -1/32 beat). The primary strokes land on the grid positions 1 and &.
