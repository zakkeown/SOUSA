# PAS #16: Single Paradiddle

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 16 |
| **Name** | Single Paradiddle |
| **Category** | diddle |
| **Family** | paradiddle |
| **Composed Of** | 2 single strokes + 1 diddle per half-cycle (the "para" = single alternation, "diddle" = double) |
| **Related** | #1 Single Stroke Roll (the alternating singles component), #6 Double Stroke Open Roll (the diddle component), #17 Double Paradiddle (adds two more singles), #18 Triple Paradiddle (adds four more singles), #19 Paradiddle-Diddle (adds a second diddle) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 2 beats |
| **Strokes Per Cycle** | 8 |
| **Primary Strokes Per Cycle** | 8 |

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | 0 | 1 | R | A |
| 2 | 1/4 | 1e | L | t |
| 3 | 1/2 | 1& | R | d1 |
| 4 | 3/4 | 1a | R | d2 |
| 5 | 1 | 2 | L | A |
| 6 | 1 + 1/4 | 2e | R | t |
| 7 | 1 + 1/2 | 2& | L | d1 |
| 8 | 1 + 3/4 | 2a | L | d2 |

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>
R  L  R R | L  R  L L
```

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | R | accent | none | n/a | Accented lead stroke, defines the start of the right-hand half |
| 2 | L | tap | none | n/a | Unaccented single, the second "para" stroke |
| 3 | R | d1 | none | n/a | First stroke of right-hand diddle, wrist-initiated |
| 4 | R | d2 | none | n/a | Second stroke of right-hand diddle, finger-controlled |
| 5 | L | accent | none | n/a | Accented lead stroke, defines the start of the left-hand half |
| 6 | R | tap | none | n/a | Unaccented single, the second "para" stroke |
| 7 | L | d1 | none | n/a | First stroke of left-hand diddle, wrist-initiated |
| 8 | L | d2 | none | n/a | Second stroke of left-hand diddle, finger-controlled |

### Ornament Timing

**Diddle Timing**:
- Each diddle pair occupies one eighth-note duration (1/2 beat)
- Each individual diddle stroke occupies 1/4 beat (one sixteenth note)
- d1 falls on the "&" of the beat (grid position 1& or 2&)
- d2 falls on the "a" of the beat (grid position 1a or 2a)
- At slow tempos: both strokes are distinct and clearly separated (open)
- At fast tempos: strokes are closely spaced, nearly indistinguishable (closed)

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent | 1.0 | Full-stroke lead note at the top of each half-cycle |
| tap | 0.65 - 0.77 | Unaccented single stroke; must contrast clearly against the accent |
| d1 | 0.65 - 0.77 | First diddle stroke at tap level; matches the passage dynamic |
| d2 | 0.90 - 0.98 x d1 | Slight natural decay on the second bounce |

The accent-tap contrast is the defining dynamic feature of the single paradiddle. The accented first stroke of each half-cycle creates a clear rhythmic pulse, while the tap and diddle strokes fill the space between accents at a noticeably lower dynamic. The d1 of each diddle matches the surrounding tap level, and d2 decays slightly from d1 due to the finger-controlled rebound. The listener should hear a strong "one" followed by three quieter strokes in each half-cycle.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>  -  -  - | >  -  -  -
```

One accent per half-cycle, always on the lead stroke (stroke 1 and stroke 5). The accent alternates hands each half-cycle, which is the key pedagogical feature of all paradiddles: forced hand-lead switching.

### Dynamic Contour

The single paradiddle has a repeating two-beat dynamic contour with a strong attack on beat 1 and beat 2, followed by three quieter strokes. This creates a pulse-like feel that makes paradiddles effective as groove patterns in drum set playing. The accent-tap contrast ratio (roughly 1.0 : 0.70) should remain consistent between the right-lead and left-lead halves. A hallmark of mastery is identical accent strength and tap evenness regardless of which hand leads. The diddle at the end of each half naturally sits at the tap dynamic level, creating a smooth transition into the next accent.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | R | accent | full | wrist + arm |
| 2 | L | tap | low | wrist |
| 3 | R | d1 | low | wrist |
| 4 | R | d2 | tap | fingers |
| 5 | L | accent | full | wrist + arm |
| 6 | R | tap | low | wrist |
| 7 | L | d1 | low | wrist |
| 8 | L | d2 | tap | fingers |

The accent (stroke 1, 5) is played from full height with a combined wrist and arm motion (Moeller whip at moderate tempos). After the accent, the stick must drop to low height for the tap and the diddle d1. The diddle d2 is generated by finger control catching the rebound of d1. The hand that just completed its diddle must then lift to full height in preparation for its next accent two beats later, while the opposite hand executes its accented lead stroke.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm strokes for accents with a clear height differential between accents and taps. Diddles are open — two distinct wrist strokes on the same hand. Focus on accent-tap contrast and even diddle quality. Each stroke is deliberately placed. |
| Moderate (100-160 BPM) | Accents transition to wrist-driven with Moeller whip motion for efficiency. Taps remain at low height. Diddles shift to the wrist-finger technique: d1 is a wrist drop, d2 is a finger-controlled rebound catch. The accent naturally prepares the stick height for the following tap via a controlled downstroke. |
| Fast (160-200 BPM) | All motions become compact. Accents are wrist-only from half height; the full arm motion is too slow. Taps and diddles are finger-controlled at tap height. The accent-tap dynamic contrast narrows but must remain audible. Diddles are essentially closed. The pattern becomes a fluid continuous motion with minimal stick lift. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Inverted paradiddle**: Reorder to RRLR LLRL — the diddle moves to the beginning instead of the end
- **Reverse paradiddle**: RRLR LLRL or RRLL RRLL variations that shift the accent placement relative to the diddle
- **Paradiddle with accents on other strokes**: Move the accent to strokes 2, 3, or 4 for coordination development
- **Paradiddle as a groove**: Distribute hands between hi-hat, snare, and bass drum to create linear patterns (common in funk and jazz)
- **Paradiddle between surfaces**: Alternate the accent on a different drum/cymbal than the taps for melodic phrasing
- **Continuous hand-switching exercise**: Play many measures continuously, noting how the lead hand alternates every beat

### Prerequisites

- #1 Single Stroke Roll — clean, even alternating singles
- #6 Double Stroke Open Roll — controlled diddles with matched d1/d2 quality

### Builds Toward

- #17 Double Paradiddle (extends with two additional singles before the diddle)
- #18 Triple Paradiddle (extends with four additional singles before the diddle)
- #19 Paradiddle-Diddle (adds a second diddle)
- #24 Flam Paradiddle (adds a flam grace note to the paradiddle accent)
- #25 Single Flammed Mill (flammed singles within a paradiddle-like structure)
- #26 Flam Paradiddle-Diddle (flam + paradiddle-diddle)
- #35 Single Dragadiddle (drag + diddle in a paradiddle frame)
- #36 Drag Paradiddle #1 (drag ornaments on paradiddle accents)
- #37 Drag Paradiddle #2 (extended drag paradiddle)

### Teaching Notes

The single paradiddle is the most fundamental pattern in the diddle category and one of the most important rudiments in all of drumming. Its unique value lies in the **hand-lead switching** property: because the pattern is RLRR LRLL, each half-cycle ends with a diddle on the same hand that accented, forcing the opposite hand to lead the next half-cycle. This automatic alternation of the lead hand is not present in simple alternating patterns (single stroke roll) or same-hand patterns (double stroke roll), making the paradiddle an essential tool for developing ambidextrous control.

**Common mistakes:**
- Weak accent-tap contrast — the accents should be noticeably louder than the taps and diddle
- Uneven diddle — d2 significantly softer or rushed relative to d1
- Rhythmic unevenness between the two single strokes and the diddle pair
- One hand's accent consistently louder than the other (hand dominance leaking through)
- Losing the sense of pulse — the accent on stroke 1 of each half should always feel like a downbeat

**Practice strategies:**
- Begin at slow tempos (60 BPM) with a metronome, emphasizing the accent-tap contrast
- Practice each half-cycle separately: RLRR (right lead) and LRLL (left lead) in isolation
- Play on a practice pad watching stick heights — accent strokes should visibly rise higher
- Alternate between paradiddle and straight singles to feel the hand-switching moment
- Practice paradiddles in groupings of 2, 4, and 8 repetitions, verifying consistency
- Record and listen for accent balance between right-lead and left-lead halves

**What to listen for:**
- Clear, consistent accent on the first stroke of each half-cycle
- Even volume across the three unaccented strokes (tap, d1, d2)
- Identical rhythmic spacing across all four sixteenth-note positions
- Equal accent strength from both hands
- Smooth, continuous flow without hesitation at the hand-switching point

### Historical Context

The paradiddle is a uniquely English rudiment in origin, with its earliest documented appearances in English military drumming manuscripts from the 17th and 18th centuries. The name "paradiddle" is likely onomatopoeic, imitating the sound of the pattern: "pa-ra" for the two single strokes and "did-dle" for the double. The Greek prefix "para-" (beside, beyond) combined with "diddle" (to move with short rapid motions) also contributes to the etymology.

The single paradiddle was included in the NARD (National Association of Rudimental Drummers) original 26 rudiments when NARD was established in 1933, and its lineage can be traced back through Gardiner Strube's 1870 *Drum and Fife Instructor* and earlier American rudimental texts. It was retained as PAS #16 when the Percussive Arts Society expanded the list to 40 International Drum Rudiments in 1984 under Jay Wanamaker's committee, positioned as the first rudiment in the diddle category.

The paradiddle family holds a special place in modern drum set playing. The hand-lead switching property makes paradiddles uniquely suited for creating linear grooves, fills, and orchestrations around the kit. Virtually every jazz, funk, and rock drummer relies on paradiddle-based patterns. The concept of "paradiddle inversions" — starting the pattern from different points in the cycle — has become a fundamental creative tool in contemporary drumming.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>              >
R  L  R R    L  R  L L
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1     e     &     a   | 2     e     &     a   |
Hand:    | R     L     R     R   | L     R     L     L   |
Type:    | A     t     d1    d2  | A     t     d1    d2  |
Accent:  | >                     | >                     |
```
