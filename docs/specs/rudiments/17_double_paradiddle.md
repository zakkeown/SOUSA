# PAS #17: Double Paradiddle

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 17 |
| **Name** | Double Paradiddle |
| **Category** | diddle |
| **Family** | paradiddle |
| **Composed Of** | 4 single strokes + 1 diddle per half-cycle ("double" refers to two pairs of alternating singles before the diddle) |
| **Related** | #16 Single Paradiddle (shorter version with 2 singles + 1 diddle), #18 Triple Paradiddle (longer version with 6 singles + 1 diddle), #19 Paradiddle-Diddle (2 singles + 2 diddles), #1 Single Stroke Roll (the alternating singles component), #6 Double Stroke Open Roll (the diddle component) |
| **NARD Original** | Yes (listed among the NARD 13 Essential Rudiments) |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 3 beats |
| **Strokes Per Cycle** | 12 |
| **Primary Strokes Per Cycle** | 12 |

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | 0 | 1 | R | A |
| 2 | 1/4 | 1e | L | t |
| 3 | 1/2 | 1& | R | t |
| 4 | 3/4 | 1a | L | t |
| 5 | 1 | 2 | R | d1 |
| 6 | 1 + 1/4 | 2e | R | d2 |
| 7 | 1 + 1/2 | 2& | L | A |
| 8 | 1 + 3/4 | 2a | R | t |
| 9 | 2 | 3 | L | t |
| 10 | 2 + 1/4 | 3e | R | t |
| 11 | 2 + 1/2 | 3& | L | d1 |
| 12 | 2 + 3/4 | 3a | L | d2 |

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>                       >
R  L  R  L | R R  L  R | L  R  L L
```

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | R | accent | none | n/a | Accented lead stroke, right-hand half begins |
| 2 | L | tap | none | n/a | Unaccented single, first alternation |
| 3 | R | tap | none | n/a | Unaccented single, second alternation |
| 4 | L | tap | none | n/a | Unaccented single, third alternation |
| 5 | R | d1 | none | n/a | First stroke of right-hand diddle, wrist-initiated |
| 6 | R | d2 | none | n/a | Second stroke of right-hand diddle, finger-controlled |
| 7 | L | accent | none | n/a | Accented lead stroke, left-hand half begins |
| 8 | R | tap | none | n/a | Unaccented single, first alternation |
| 9 | L | tap | none | n/a | Unaccented single, second alternation |
| 10 | R | tap | none | n/a | Unaccented single, third alternation |
| 11 | L | d1 | none | n/a | First stroke of left-hand diddle, wrist-initiated |
| 12 | L | d2 | none | n/a | Second stroke of left-hand diddle, finger-controlled |

### Ornament Timing

**Diddle Timing**:
- Each diddle pair occupies one eighth-note duration (1/2 beat)
- Each individual diddle stroke occupies 1/4 beat (one sixteenth note)
- Right-hand diddle: d1 on beat 2, d2 on 2e
- Left-hand diddle: d1 on 3&, d2 on 3a
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

The accent-tap contrast in the double paradiddle follows the same principle as the single paradiddle, but with four taps between accents instead of two. The longer run of unaccented strokes makes the accent feel more emphatic by contrast. The diddle at the end of each half-cycle sits at the same dynamic level as the taps, providing a smooth transition into the next accented lead stroke.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>  -  -  - | -  -  >  - | -  -  -  -
```

One accent per half-cycle, always on the lead stroke (stroke 1 and stroke 7). The accent alternates hands every 6 strokes (1.5 beats). The 6-stroke grouping creates a natural cross-rhythm against the 4/4 meter when played continuously, giving the double paradiddle its characteristic "rolling over the barline" feel.

### Dynamic Contour

The double paradiddle has a repeating six-note dynamic contour: one strong accent followed by five quieter strokes. The longer space between accents (compared to the single paradiddle) creates a more flowing, rolling character. The accent functions as a rhythmic anchor every 1.5 beats. When played continuously in 4/4, the accent pattern cycles through different beat positions (beat 1, the "&" of beat 2, beat 4, the "e" of beat 5, etc.), creating a polyrhythmic effect that makes the double paradiddle particularly useful for building complex musical phrases. The d2 decay at the end of each half provides a natural diminuendo leading into the next accent.

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
| 5 | R | d1 | low | wrist |
| 6 | R | d2 | tap | fingers |
| 7 | L | accent | full | wrist + arm |
| 8 | R | tap | low | wrist |
| 9 | L | tap | low | wrist |
| 10 | R | tap | low | wrist |
| 11 | L | d1 | low | wrist |
| 12 | L | d2 | tap | fingers |

The accent is played from full height with a combined wrist and arm motion. After the accent, the stick drops immediately to low height for the four taps and d1. The d2 is generated by finger rebound from d1. The hand completing the diddle then has a full 1.5 beats (6 strokes) before its next accent, providing ample time to lift back to full height. This longer preparation window (compared to the single paradiddle's 1 beat) makes the accent preparation slightly more forgiving.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm strokes for accents with clear height contrast. All four single strokes are deliberate wrist motions at low height. Diddles are open with two distinct wrist strokes. Focus on maintaining even spacing across all six strokes per half-cycle and clear accent-tap differentiation. |
| Moderate (100-150 BPM) | Accents employ Moeller whip technique. Taps are relaxed wrist strokes. Diddles transition to wrist-finger technique (d1 wrist, d2 finger rebound). The four alternating singles should flow smoothly with minimal arm movement, allowing the wrist to settle into a metronomic pattern before the diddle. |
| Fast (150-180 BPM) | All motions become compact. Accents are wrist-only from half height. Taps and diddles are finger-controlled at tap height. The long run of four singles tends to stabilize the tempo, making the double paradiddle somewhat easier to play evenly at speed than the single paradiddle. Diddles are essentially closed. The accent-tap contrast narrows but must remain perceptible. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Accents on different strokes**: Move the accent to strokes 2, 3, 4, 5, or 6 for coordination development and different musical effects
- **Double paradiddle in 6/8**: Play as eighth notes in 6/8 time, one full half-cycle per measure — the six-note grouping fits compound time naturally
- **Orchestrated around the kit**: Distribute accents on toms or cymbals with taps on the snare for fill patterns
- **Continuous double paradiddles over a barline**: Accent pattern shifts through different beat positions in 4/4, creating polyrhythmic effects
- **Double paradiddle with flams**: Add flam grace notes to the accents (approaches #26 Flam Paradiddle-Diddle territory)

### Prerequisites

- #1 Single Stroke Roll — clean, even alternating singles across four-stroke runs
- #6 Double Stroke Open Roll — controlled diddles with matched d1/d2 quality
- #16 Single Paradiddle — the accent-tap-diddle pattern in its simplest form

### Builds Toward

- #18 Triple Paradiddle (extends with two more singles per half-cycle)
- #26 Flam Paradiddle-Diddle (adds flams and an extra diddle)
- Compound rudiments that combine paradiddle sticking with other ornaments

### Teaching Notes

The double paradiddle extends the single paradiddle by adding two more alternating single strokes before the diddle. This creates a six-note grouping that naturally fits compound meters (6/8, 12/8) when played as eighth notes. The **hand-lead switching** property is preserved: RLRLRR ends on R, so L must lead the next half; LRLRLL ends on L, so R must lead the next half.

The "double" in the name refers to the two pairs of alternating singles (RL RL) before the diddle (RR), not to the number of times the pattern repeats. This naming convention extends through the paradiddle family: single = one pair of singles (RL), double = two pairs (RLRL), triple = three pairs (RLRLRL).

**Common mistakes:**
- Uneven spacing among the four single strokes — the alternating singles should be perfectly metronomic
- Rushing the diddle after four singles, or dragging the diddle to compensate
- Accent inconsistency between right-lead and left-lead halves
- Losing track of the 6-stroke grouping when playing in 4/4 (the accent shifts beat position each cycle)
- Weak d2 at the end of each half — the diddle must be as clean as in the single paradiddle

**Practice strategies:**
- Begin at slow tempos with a metronome, counting "1-e-&-a-2-e" for each 6-stroke half
- Practice each half-cycle separately: RLRLRR (right lead) and LRLRLL (left lead)
- Play in 6/8 time at first — this aligns the accent with the natural pulse and simplifies counting
- Then play in 4/4 to experience the polyrhythmic accent shifting
- Compare to the single paradiddle: play 4 bars of singles followed by 4 bars of doubles, feeling how the extra singles change the momentum
- Record and listen for accent balance and diddle quality between halves

**What to listen for:**
- Clear, consistent accent on the first stroke of each 6-stroke group
- Even volume and spacing across the four alternating single strokes
- Clean diddle with matched d1/d2 quality
- Equal accent strength from both hands
- Smooth transition between halves without hesitation

### Historical Context

The double paradiddle holds the distinction of being included in the NARD 13 Essential Rudiments, a core subset of the 26 Standard American Drum Rudiments designated by the National Association of Rudimental Drummers in 1933. This places it among the most fundamental patterns in the American rudimental tradition. Its inclusion in the Essential 13 (while the single paradiddle was in the broader 26 but not the Essential 13) reflects the double paradiddle's historical importance in military field drumming, where the six-note grouping is naturally suited to the compound meters common in marches and quicksteps.

The double paradiddle's six-stroke grouping makes it one of the most versatile paradiddle variants. In contemporary drum set playing, it is frequently used in 6/8 and 12/8 time signatures, Afro-Cuban-influenced grooves, and jazz waltz patterns. The natural polyrhythmic tension created when playing the six-note pattern in 4/4 time has been exploited by drummers from Buddy Rich to Vinnie Colaiuta.

The rudiment was retained as PAS #17 when the Percussive Arts Society expanded the list to 40 International Drum Rudiments in 1984.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>                          >
R  L  R  L  R R    L  R  L  R  L L
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1     e     &     a   | 2     e     &     a   | 3     e     &     a   |
Hand:    | R     L     R     L   | R     R     L     R   | L     R     L     L   |
Type:    | A     t     t     t   | d1    d2    A     t   | t     t     d1    d2  |
Accent:  | >                     |             >         |                       |
```
