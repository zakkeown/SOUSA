# PAS #2: Single Stroke Four

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 2 |
| **Name** | Single Stroke Four |
| **Category** | roll |
| **Family** | single stroke |
| **Composed Of** | Four alternating singles with accent — a defined grouping of the single stroke roll primitive |
| **Related** | #1 Single Stroke Roll (unaccented continuous form), #3 Single Stroke Seven (seven-stroke grouped variant) |
| **NARD Original** | No (added by PAS in 1984) |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 1 beat |
| **Strokes Per Cycle** | 4 |
| **Primary Strokes Per Cycle** | 4 |

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | 0 | 1 | R | A |
| 2 | 1/4 | 1e | L | t |
| 3 | 1/2 | 1& | R | t |
| 4 | 3/4 | 1a | L | t |

The cycle naturally alternates starting hand: the first cycle begins R, the second begins L, and so on. This is because four strokes with alternating sticking automatically shifts the lead hand each cycle.

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>
R  L  R  L | L  R  L  R
```

Two consecutive cycles shown. The accent (>) falls on stroke 1 of each group. The lead hand alternates naturally between cycles.

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | R | accent | none | n/a | Accented downbeat of the group; full stroke |
| 2 | L | tap | none | n/a | Unaccented single stroke |
| 3 | R | tap | none | n/a | Unaccented single stroke |
| 4 | L | tap | none | n/a | Unaccented single stroke; lead hand shifts on next cycle |

### Ornament Timing

No ornaments. All strokes are primary singles.

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent | 1.0 | Reference level; stroke 1 of each group |
| tap | 0.65 - 0.77 | Strokes 2, 3, and 4; even and matched |

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>  -  -  -
```

Single accent on the first stroke of each four-note group. When played across a full measure of 4/4, the accent falls on every beat (1, 2, 3, 4), creating a clear quarter-note pulse within the sixteenth-note stream.

### Dynamic Contour

The single stroke four has a repeating "accent-decay" contour: a strong first stroke followed by three even, softer strokes. This creates a clear rhythmic grouping where the accent defines the start of each four-note cell. The three tap strokes should be at a matched dynamic level with no decay or crescendo between them. The contrast between accent and taps is the defining musical characteristic — without it, the pattern reverts to an undifferentiated single stroke roll. The accent-to-tap ratio should be clearly audible but not exaggerated; the taps carry the rhythmic subdivision while the accent anchors the pulse.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | R | accent | full | wrist + arm |
| 2 | L | tap | low | wrist |
| 3 | R | tap | low | wrist |
| 4 | L | tap | low | wrist |

The accent on stroke 1 is a downstroke: the stick starts high (full height class, 12"+) and stops low after striking. Strokes 2 and 3 are taps played from a low position. Stroke 4 is an upstroke: starting low and lifting the stick to full height to prepare the accent on the next cycle's downbeat. This downstroke-tap-tap-upstroke motion forms the foundational four-stroke sequence (Moeller method).

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full arm + wrist motion for accents with clear height differential. Taps played with deliberate low strokes. Each of the four motions (down, tap, tap, up) is distinct and visible. |
| Moderate (100-160 BPM) | Accent motion becomes more wrist-driven; arm assists but does not dominate. Tap strokes rely on controlled rebound. The up-down-tap-tap cycle becomes fluid and connected. |
| Fast (160-200 BPM) | Height differential decreases but accent remains perceptibly louder. Finger control supplements wrist for taps. Moeller whip technique becomes essential for accents to maintain separation without tension. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Accent displacement**: Move the accent to stroke 2, 3, or 4 within the group (accent on "e", "&", or "a") to develop dynamic independence
- **Double accent**: Accent strokes 1 and 3 to create an accent pattern on the beats and "ands"
- **Cross-surface orchestration**: Play accents on one surface (e.g., snare rim or tom) and taps on another to create timbral grouping
- **Dynamic gradient**: Replace the binary accent/tap with a four-level dynamic contour (f, mf, mp, p) across the four strokes
- **Continuous alternation across bar lines**: Chain groups without pause, emphasizing that the lead hand alternates every group

### Prerequisites

- #1 Single Stroke Roll — must be able to play even alternating strokes before adding accent grouping

### Builds Toward

- #3 Single Stroke Seven (extends the grouping concept to seven strokes)
- #16 Single Paradiddle (introduces diddle within accented grouping)
- All accented rudiments benefit from the accent/tap independence developed here

### Teaching Notes

The single stroke four is typically the second rudiment taught after the single stroke roll. It introduces the critical concept of dynamic contrast within an alternating pattern — the ability to play one stroke louder while keeping subsequent strokes even and controlled.

**Common mistakes:**
- Accent bleeds into stroke 2 (the tap immediately following the accent is too loud due to uncontrolled rebound)
- Taps are uneven — stroke 4 is often louder than strokes 2-3 because the player lifts the stick early to prepare the next accent
- Loss of rhythmic evenness when adding the accent (the accent arrives early or the tap after the accent is late)
- Identical dynamics on all four strokes (failure to differentiate accent from taps)

**Practice strategies:**
- Begin by practicing the single stroke roll with no accents, then gradually introduce the accent on beat 1 while maintaining tap evenness
- Practice the four-stroke motion sequence in isolation: downstroke, tap, tap, upstroke (Moeller motion)
- Use a metronome at slow tempos and check that the accent falls precisely on the click, not before it
- Practice leading with both hands: R-L-R-L and L-R-L-R groups
- Record and compare the volume of strokes 2, 3, and 4 to verify they are matched

**What to listen for:**
- Clear accent/tap contrast without exaggeration
- Strokes 2, 3, and 4 at identical dynamic level
- Perfectly even sixteenth-note spacing (no rhythmic distortion from the accent)
- Consistent tone quality across both hands

### Historical Context

The single stroke four was not part of the NARD original 26 rudiments. It was added to the PAS 40 International Drum Rudiments list in 1984 when Jay Wanamaker's committee expanded and reorganized the rudiment canon. While the single stroke four is simply a four-note grouping of the single stroke roll with an accent, its formal inclusion recognized the pedagogical importance of defining accented groupings as distinct rudiments. The concept of grouping singles in fours with accents predates its formal PAS codification — it appears implicitly in military drumming manuals and marching percussion literature throughout the 19th and 20th centuries. Its categorization as a separate rudiment reflects a modern pedagogical emphasis on accent control as a discrete skill.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>              >
R  L  R  L  |  L  R  L  R
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1     e     &     a   |
Hand:    | R     L     R     L   |
Type:    | A     t     t     t   |
Accent:  | >                     |
```

Two-cycle view (showing natural hand alternation):

```
Beat:    | 1     e     &     a   | 2     e     &     a   |
Hand:    | R     L     R     L   | L     R     L     R   |
Type:    | A     t     t     t   | A     t     t     t   |
Accent:  | >                     | >                     |
```
