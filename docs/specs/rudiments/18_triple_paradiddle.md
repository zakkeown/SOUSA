# PAS #18: Triple Paradiddle

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 18 |
| **Name** | Triple Paradiddle |
| **Category** | diddle |
| **Family** | paradiddle |
| **Composed Of** | 6 single strokes + 1 diddle per half-cycle ("triple" refers to three pairs of alternating singles before the diddle) |
| **Related** | #16 Single Paradiddle (2 singles + 1 diddle), #17 Double Paradiddle (4 singles + 1 diddle), #19 Paradiddle-Diddle (2 singles + 2 diddles), #1 Single Stroke Roll (the alternating singles component), #6 Double Stroke Open Roll (the diddle component) |
| **NARD Original** | No (first published by J. Burns Moore in *Art of Drumming*, 1937; added to PAS 40 in 1984) |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 4 beats |
| **Strokes Per Cycle** | 16 |
| **Primary Strokes Per Cycle** | 16 |

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | 0 | 1 | R | A |
| 2 | 1/4 | 1e | L | t |
| 3 | 1/2 | 1& | R | t |
| 4 | 3/4 | 1a | L | t |
| 5 | 1 | 2 | R | t |
| 6 | 1 + 1/4 | 2e | L | t |
| 7 | 1 + 1/2 | 2& | R | d1 |
| 8 | 1 + 3/4 | 2a | R | d2 |
| 9 | 2 | 3 | L | A |
| 10 | 2 + 1/4 | 3e | R | t |
| 11 | 2 + 1/2 | 3& | L | t |
| 12 | 2 + 3/4 | 3a | R | t |
| 13 | 3 | 4 | L | t |
| 14 | 3 + 1/4 | 4e | R | t |
| 15 | 3 + 1/2 | 4& | L | d1 |
| 16 | 3 + 3/4 | 4a | L | d2 |

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>                                >
R  L  R  L | R  L  R R | L  R  L  R | L  R  L L
```

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | R | accent | none | n/a | Accented lead stroke, right-hand half begins |
| 2 | L | tap | none | n/a | Unaccented single, first alternation |
| 3 | R | tap | none | n/a | Unaccented single, second alternation |
| 4 | L | tap | none | n/a | Unaccented single, third alternation |
| 5 | R | tap | none | n/a | Unaccented single, fourth alternation |
| 6 | L | tap | none | n/a | Unaccented single, fifth alternation |
| 7 | R | d1 | none | n/a | First stroke of right-hand diddle, wrist-initiated |
| 8 | R | d2 | none | n/a | Second stroke of right-hand diddle, finger-controlled |
| 9 | L | accent | none | n/a | Accented lead stroke, left-hand half begins |
| 10 | R | tap | none | n/a | Unaccented single, first alternation |
| 11 | L | tap | none | n/a | Unaccented single, second alternation |
| 12 | R | tap | none | n/a | Unaccented single, third alternation |
| 13 | L | tap | none | n/a | Unaccented single, fourth alternation |
| 14 | R | tap | none | n/a | Unaccented single, fifth alternation |
| 15 | L | d1 | none | n/a | First stroke of left-hand diddle, wrist-initiated |
| 16 | L | d2 | none | n/a | Second stroke of left-hand diddle, finger-controlled |

### Ornament Timing

**Diddle Timing**:
- Each diddle pair occupies one eighth-note duration (1/2 beat)
- Each individual diddle stroke occupies 1/4 beat (one sixteenth note)
- Right-hand diddle: d1 on 2&, d2 on 2a
- Left-hand diddle: d1 on 4&, d2 on 4a
- At slow tempos: both strokes are distinct and clearly separated (open)
- At fast tempos: strokes are closely spaced, nearly indistinguishable (closed)

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent | 1.0 | Full-stroke lead note at the top of each half-cycle |
| tap | 0.65 - 0.77 | Unaccented single strokes; must contrast clearly against the accent |
| d1 | 0.65 - 0.77 | First diddle stroke at tap level; matches the passage dynamic |
| d2 | 0.90 - 0.98 x d1 | Slight natural decay on the second bounce |

The triple paradiddle has the longest run of unaccented strokes in the paradiddle family: six taps plus the diddle (seven unaccented strokes) between each accent. This extended quiet passage makes each accent feel particularly emphatic. The dynamic contrast between the accent and the surrounding taps must be maintained consistently across both halves despite the longer phrase length.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>  -  -  - | -  -  -  - | >  -  -  - | -  -  -  -
```

One accent per half-cycle, always on the lead stroke (stroke 1 and stroke 9). The accent falls on beat 1 and beat 3, creating an accent pattern that aligns neatly with half-note pulse in 4/4 time. This is unique among the paradiddle family: the triple paradiddle's 8-stroke half-cycle fills exactly one half-bar of 4/4, so the accent pattern does not create cross-rhythms when played continuously.

### Dynamic Contour

The triple paradiddle has a broad, arching dynamic contour: a strong accent followed by a long plateau of seven even unaccented strokes, then another strong accent. The 8-stroke (2-beat) half-cycle creates a slow, stately pulse when accents are emphasized. Unlike the single paradiddle (which pulses every beat) or the double paradiddle (which creates 6-against-4 polyrhythms), the triple paradiddle aligns naturally with the 4/4 meter and creates a half-note accent feel. The extended run of singles between accents demands endurance and consistency, while the diddle at the end of each half provides a brief moment of same-hand activity before the hand switch.

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
| 5 | R | tap | low | wrist |
| 6 | L | tap | low | wrist |
| 7 | R | d1 | low | wrist |
| 8 | R | d2 | tap | fingers |
| 9 | L | accent | full | wrist + arm |
| 10 | R | tap | low | wrist |
| 11 | L | tap | low | wrist |
| 12 | R | tap | low | wrist |
| 13 | L | tap | low | wrist |
| 14 | R | tap | low | wrist |
| 15 | L | d1 | low | wrist |
| 16 | L | d2 | tap | fingers |

The accent is played from full height with combined wrist and arm motion. After the accent, the stick drops to low height for the six alternating taps and the diddle d1. The d2 is generated by finger rebound. The hand completing its diddle has a full 2 beats (8 strokes) before its next accent, providing the most generous preparation window in the paradiddle family. This extended recovery time makes the triple paradiddle somewhat forgiving for accent preparation, but the long run of six alternating singles demands sustained evenness and endurance.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm accents with a clear height differential. All six singles are deliberate, even wrist strokes at low height. Diddles are open — two distinct wrist strokes. The long single-stroke run feels similar to practicing the single stroke roll with a deliberate accent every 8 strokes. Focus on maintaining metronomic evenness across all eight positions in each half. |
| Moderate (100-140 BPM) | Accents use Moeller whip technique. The six alternating singles settle into a relaxed wrist pattern at low height — essentially a short single stroke roll excerpt. Diddles shift to wrist-finger technique. The extended single-stroke run provides natural rhythmic stability, making moderate tempos comfortable. |
| Fast (140-160 BPM) | All motions become compact. Accents are wrist-only from half height. Singles and diddles are finger-controlled at tap height. The long phrase length becomes a physical endurance challenge at speed. The accent-tap contrast narrows but must remain audible. Diddles are essentially closed. Arm fatigue from the continuous single-stroke run is the primary limiting factor. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Accents on alternate strokes**: Move the accent to different positions within the 8-stroke half for coordination and musical variety
- **Triple paradiddle in 4/4**: The natural alignment of the 8-stroke half with 2 beats of 4/4 makes this rudiment particularly well-suited for standard time signature applications
- **Orchestrated fills**: Distribute the accent on toms or cymbals while keeping taps on the snare — the 2-beat accent spacing creates quarter-note or half-note tom patterns
- **Paradiddle family progression**: Play single, double, then triple paradiddles in sequence as a warm-up to develop all three phrase lengths
- **Dynamic shaping within the half**: Apply a crescendo or decrescendo across the six singles leading to the diddle for additional musical expression

### Prerequisites

- #1 Single Stroke Roll — extended runs of even alternating singles (six singles is a significant endurance requirement)
- #6 Double Stroke Open Roll — controlled diddles with matched d1/d2 quality
- #16 Single Paradiddle — the accent-tap-diddle concept in its simplest form
- #17 Double Paradiddle — the intermediate phrase length

### Builds Toward

- Extended paradiddle exercises combining all three paradiddle lengths (single, double, triple)
- Complex fill patterns in drum set playing that exploit the 2-beat half-cycle accent spacing
- Compound rudiments that overlay flam or drag ornaments on the paradiddle structure

### Teaching Notes

The triple paradiddle is the longest member of the standard paradiddle family, with six alternating single strokes before each diddle. The "triple" refers to three pairs of alternating singles (RL RL RL) preceding the diddle (RR), following the paradiddle naming convention: single = 1 pair (RL), double = 2 pairs (RLRL), triple = 3 pairs (RLRLRL). The **hand-lead switching** property is preserved: RLRLRLRR ends on R, so L must lead the next half; LRLRLRLL ends on L, so R must lead.

**Common mistakes:**
- Uneven spacing among the six alternating singles — the long run exposes any rhythmic inconsistency
- Rushing or dragging the diddle relative to the surrounding singles
- Fatigue causing the taps to become louder or uneven toward the end of each half
- Accent imbalance between hands, especially as fatigue sets in during extended practice
- Losing the phrase structure — the accent on stroke 1 must always feel like the "one" of each half

**Practice strategies:**
- Build up from the single paradiddle: play RLRR, then RLRLRR, then RLRLRLRR, adding one pair of singles at a time
- Practice the 8-stroke half in isolation: RLRLRLRR (right lead) and LRLRLRLL (left lead)
- Begin at very slow tempos (60 BPM) to develop endurance across the 6-stroke single run
- Use the natural 4/4 alignment: accent on beats 1 and 3 serves as a helpful metronome anchor
- Compare with the single stroke roll: play 8 bars of singles, then 8 bars of triple paradiddles, feeling how the diddle and accent create structure within the alternating pattern
- Record and critically listen for any unevenness in the long single-stroke runs

**What to listen for:**
- Clear, consistent accent on stroke 1 and stroke 9
- Perfectly even spacing and volume across all six alternating singles
- Clean diddle with matched d1/d2 quality
- Equal accent and tap quality from both hands
- No rhythmic distortion at the single-to-diddle transition or the diddle-to-accent transition

### Historical Context

The triple paradiddle is the only member of the paradiddle family that was not included in the NARD (National Association of Rudimental Drummers) original 26 rudiments of 1933. It was first published by J. Burns Moore in his *Art of Drumming* in 1937, just four years after the NARD standardization. Moore's text featured just one rudiment not found in the NARD 26, and this was it. The triple paradiddle was subsequently adopted into common rudimental practice and was included when the Percussive Arts Society expanded the list to 40 International Drum Rudiments in 1984 under Jay Wanamaker's committee, where it was assigned PAS #18.

The triple paradiddle's 8-stroke half-cycle aligns naturally with 2 beats of 4/4 time, making it metrically simpler than the single paradiddle (4-stroke half = 1 beat, which can feel too short for phrasing) or the double paradiddle (6-stroke half = 1.5 beats, which creates polyrhythmic tension). This clean metric alignment makes the triple paradiddle a practical choice for drum set fills and marching percussion cadences where musical clarity is important.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>                                      >
R  L  R  L  R  L  R R    L  R  L  R  L  R  L L
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1     e     &     a   | 2     e     &     a   | 3     e     &     a   | 4     e     &     a   |
Hand:    | R     L     R     L   | R     L     R     R   | L     R     L     R   | L     R     L     L   |
Type:    | A     t     t     t   | t     t     d1    d2  | A     t     t     t   | t     t     d1    d2  |
Accent:  | >                     |                       | >                     |                       |
```
