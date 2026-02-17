# PAS #19: Paradiddle-Diddle

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 19 |
| **Name** | Paradiddle-Diddle |
| **Category** | diddle |
| **Family** | paradiddle |
| **Composed Of** | 2 single strokes + 2 diddles per half-cycle (a paradiddle with an additional diddle appended) |
| **Related** | #16 Single Paradiddle (2 singles + 1 diddle — the paradiddle-diddle adds a second diddle), #17 Double Paradiddle (shares the 6-stroke grouping), #6 Double Stroke Open Roll (the diddle component), #26 Flam Paradiddle-Diddle (adds flam ornaments to this pattern) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 (or 6/8 — the 6-stroke grouping fits compound meters naturally) |
| **Base Subdivision** | sextuplet (six notes per beat) |
| **Cycle Length** | 2 beats |
| **Strokes Per Cycle** | 12 |
| **Primary Strokes Per Cycle** | 12 |

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | 0 | 1 | R | A |
| 2 | 1/6 | 1la | L | t |
| 3 | 2/6 | 1li | R | d1 |
| 4 | 3/6 | 1& | R | d2 |
| 5 | 4/6 | 1la | L | d1 |
| 6 | 5/6 | 1li | L | d2 |
| 7 | 1 | 2 | L | A |
| 8 | 1 + 1/6 | 2la | R | t |
| 9 | 1 + 2/6 | 2li | L | d1 |
| 10 | 1 + 3/6 | 2& | L | d2 |
| 11 | 1 + 4/6 | 2la | R | d1 |
| 12 | 1 + 5/6 | 2li | R | d2 |

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>                    >
R  L  R R  L L | L  R  L L  R R
```

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | R | accent | none | n/a | Accented lead stroke, right-hand half begins |
| 2 | L | tap | none | n/a | Unaccented single, the second "para" stroke |
| 3 | R | d1 | none | n/a | First stroke of first diddle (right hand), wrist-initiated |
| 4 | R | d2 | none | n/a | Second stroke of first diddle (right hand), finger-controlled |
| 5 | L | d1 | none | n/a | First stroke of second diddle (left hand), wrist-initiated |
| 6 | L | d2 | none | n/a | Second stroke of second diddle (left hand), finger-controlled |
| 7 | L | accent | none | n/a | Accented lead stroke, left-hand half begins |
| 8 | R | tap | none | n/a | Unaccented single, the second "para" stroke |
| 9 | L | d1 | none | n/a | First stroke of first diddle (left hand), wrist-initiated |
| 10 | L | d2 | none | n/a | Second stroke of first diddle (left hand), finger-controlled |
| 11 | R | d1 | none | n/a | First stroke of second diddle (right hand), wrist-initiated |
| 12 | R | d2 | none | n/a | Second stroke of second diddle (right hand), finger-controlled |

### Ornament Timing

**Diddle Timing**:
- Each diddle pair occupies one sextuplet-note duration (1/3 beat)
- Each individual diddle stroke occupies 1/6 beat (one sextuplet note)
- Right-hand half: first diddle (R) on 1li-1&, second diddle (L) on 1la-1li
- Left-hand half: first diddle (L) on 2li-2&, second diddle (R) on 2la-2li
- The two consecutive diddles alternate hands (RR LL or LL RR), creating a brief excerpt of double-stroke roll texture
- At slow tempos: all four diddle strokes are distinct and clearly separated (open)
- At fast tempos: the diddle pairs blur into a smooth roll-like texture (closed)

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent | 1.0 | Full-stroke lead note at the top of each half-cycle |
| tap | 0.65 - 0.77 | Unaccented single stroke; must contrast clearly against the accent |
| d1 (first diddle) | 0.65 - 0.77 | First stroke of each diddle at tap level |
| d2 (first diddle) | 0.90 - 0.98 x d1 | Slight decay on the second stroke of the first diddle |
| d1 (second diddle) | 0.65 - 0.77 | First stroke of the second diddle at tap level |
| d2 (second diddle) | 0.90 - 0.98 x d1 | Slight decay on the second stroke of the second diddle |

The paradiddle-diddle features two consecutive diddles after the two singles. The first diddle follows the same accent-to-tap transition as in the single paradiddle. The second diddle occurs on the opposite hand, creating a brief double-stroke roll excerpt (RR LL or LL RR). Both diddles should be at the same dynamic level, with the d2 of each decaying slightly from its d1. The consecutive diddles create a subtle "rolling" dynamic texture that distinguishes this pattern from other paradiddles.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>  -  -  -  -  - | >  -  -  -  -  -
```

One accent per half-cycle, always on the lead stroke (stroke 1 and stroke 7). Unlike the first three paradiddles (which all use sixteenth-note subdivision), the paradiddle-diddle's sextuplet subdivision places 6 strokes per beat, so the accent falls on each beat. The lead hand alternates each half: R leads beat 1, L leads beat 2, R leads beat 3, and so on.

### Dynamic Contour

The paradiddle-diddle has a six-note dynamic contour: one strong accent, one tap, then four diddle strokes (two diddles of two strokes each). The two consecutive diddles create a "rolling" effect that distinguishes this rudiment from the other paradiddles, where only one diddle separates each accent. The dynamic shape is: accent peak, quick drop to tap level, then a sustained low-dynamic plateau through the four diddle strokes with subtle d2 decay within each pair. The two-diddle ending gives the pattern a compound feel that sits naturally in triplet-based and sextuplet-based rhythmic contexts (6/8, 12/8). When played at speed, the consecutive diddles produce a brief buzz-like texture that contrasts with the clean single strokes at the beginning of each half.

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
| 5 | L | d1 | low | wrist |
| 6 | L | d2 | tap | fingers |
| 7 | L | accent | full | wrist + arm |
| 8 | R | tap | low | wrist |
| 9 | L | d1 | low | wrist |
| 10 | L | d2 | tap | fingers |
| 11 | R | d1 | low | wrist |
| 12 | R | d2 | tap | fingers |

The accent is played from full height with combined wrist and arm motion. After the accent, the stick drops to low height for the tap and the first diddle d1. The first diddle d2 is generated by finger rebound. The second diddle begins on the opposite hand at low height (d1 from wrist, d2 from fingers). The hand that played the accent has a full beat (6 strokes) before it needs to play its next accent, which occurs when the pattern repeats after the mirror half. The two consecutive diddles (RR LL or LL RR) require the player to execute a double-stroke roll fragment at sextuplet speed, demanding solid finger control.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm accents with clear height contrast. Singles and diddle d1 strokes are controlled wrist motions. All four diddle strokes (two d1 + two d2) are individually articulated and clearly audible as four separate notes. Focus on maintaining even spacing across all six sextuplet positions and crisp hand transitions between the two consecutive diddles. |
| Moderate (100-150 BPM) | Accents employ Moeller whip technique. Diddles transition to wrist-finger technique (d1 wrist, d2 finger rebound). The consecutive diddles (RR LL) begin to feel like a brief double-stroke roll excerpt. The sextuplet subdivision produces notes at the same rate as sixteenth notes at 150% of the BPM, demanding substantial finger control at moderate tempos. |
| Fast (150-180 BPM) | All motions become compact. Accents are wrist-only from half height. All diddle strokes are finger-controlled at tap height. The two consecutive diddles merge into a smooth, roll-like texture. The sextuplet subdivision at 180 BPM produces 18 notes per second, requiring advanced finger technique. The accent-tap contrast narrows but must remain perceptible to preserve the pattern's identity. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Paradiddle-diddle in 6/8**: Play as eighth notes in 6/8 time — one complete half-cycle fills one measure, and the accent naturally falls on the downbeat of each bar
- **Paradiddle-diddle as sixteenth notes in 4/4**: Reinterpret the 6-stroke group in straight sixteenth notes for a 3-against-4 polyrhythmic effect (same approach as the double paradiddle)
- **Orchestrated around the kit**: Distribute accent on cymbal/tom and taps on snare; the consecutive diddles create a rapid fill between accent points
- **Paradiddle-diddle with hand reversal**: Practice leading with the non-dominant hand exclusively (LRLLRR) for balanced development
- **Double-time paradiddle-diddles**: Play at sextuplet speed over a half-time feel for advanced coordination

### Prerequisites

- #1 Single Stroke Roll — clean alternating singles
- #6 Double Stroke Open Roll — controlled diddles with matched d1/d2 quality, especially consecutive diddles alternating between hands (the RR LL fragment)
- #16 Single Paradiddle — the accent-singles-diddle concept (the paradiddle-diddle adds one more diddle)

### Builds Toward

- #26 Flam Paradiddle-Diddle (adds flam grace notes to this pattern's accents)
- Complex compound rudiments that combine paradiddle-diddle sticking with drag ornaments
- Advanced drum set applications in compound time signatures

### Teaching Notes

The paradiddle-diddle is structurally distinct from the other three paradiddles. While the single, double, and triple paradiddles all follow the pattern of N single strokes + 1 diddle, the paradiddle-diddle has 2 singles + 2 diddles. The second diddle is what sets it apart: the "diddle-diddle" suffix in the name refers to the two consecutive double strokes at the end of each half-cycle. This makes it a hybrid between a paradiddle and a double-stroke roll fragment.

A critical distinction: the paradiddle-diddle does **not** naturally alternate the lead hand when played as a single half-cycle (RLRRLL starts on R and ends on L — if repeated, R would lead again). To achieve the **hand-lead switching** that is characteristic of the paradiddle family, the full cycle must include both the right-lead half (RLRRLL) and the left-lead half (LRLLRR). The YAML definition for SOUSA includes both halves in a single 2-beat cycle to ensure this alternation.

The sextuplet subdivision is the standard interpretation for the paradiddle-diddle. The six-note grouping divides evenly into sextuplets (6 notes per beat), placing one complete half-cycle per beat. This differs from the other paradiddles, which all use sixteenth-note subdivision. The sextuplet feel gives the paradiddle-diddle a distinctly different rhythmic character: flowing and triplet-based rather than square and sixteenth-note-based.

**Common mistakes:**
- Uneven diddle quality between the first and second diddle — both must match
- The transition between the two consecutive diddles (R-to-L or L-to-R hand switch within the diddle pair) creating a "gap" or "flam"
- Treating the sextuplets as sixteenth notes (4 per beat instead of 6), losing the compound feel
- One hand's diddles consistently weaker than the other
- Losing the accent-tap contrast within the fast sextuplet subdivision

**Practice strategies:**
- Begin by playing the double-stroke roll (RRLL RRLL) continuously, then add the accent and single stroke at the beginning of each group
- Practice at slow tempos in 6/8 time to internalize the compound meter feel
- Isolate the two-diddle fragment (RRLL) as a double-stroke roll exercise
- Practice right-lead (RLRRLL) and left-lead (LRLLRR) halves separately, then combine
- Use a metronome set to dotted quarter notes in 6/8 to anchor the accent pattern
- Record and listen for the smoothness of the hand transition between consecutive diddles

**What to listen for:**
- Clear, consistent accent on the first stroke of each 6-note group
- Even, matched diddle quality across both consecutive diddles
- Smooth hand transition between the first and second diddle (no gap or flam)
- Correct sextuplet feel — the six notes should be evenly spaced within each beat
- Equal accent and diddle quality from both right-lead and left-lead halves

### Historical Context

The paradiddle-diddle was included in the NARD (National Association of Rudimental Drummers) original 26 rudiments when NARD was established in 1933. It is sometimes referred to as the "single paradiddle-diddle" to distinguish it from potential compound forms, though PAS simply calls it "Paradiddle-Diddle."

The paradiddle-diddle's six-note grouping makes it a natural fit for compound time signatures (6/8, 9/8, 12/8), and it has been a staple of rudimental marching percussion in these meters for centuries. In the drum corps tradition, paradiddle-diddles are frequently used in 6/8 street beats and roll-offs. The two consecutive diddles give the pattern a "rolling" quality distinct from other paradiddles.

In modern drum set playing, the paradiddle-diddle is widely used in shuffle and swing contexts where the triplet-based subdivision aligns with the rhythmic feel. It also appears frequently in Afro-Cuban 6/8 patterns, jazz waltz grooves, and any musical context that calls for compound-meter fluency. The rudiment was retained as PAS #19 when the Percussive Arts Society expanded the list to 40 International Drum Rudiments in 1984 under Jay Wanamaker's committee, positioned as the final rudiment in the diddle category.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>                       >
R  L  R R  L L    L  R  L L  R R
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1     la    li    &     la    li  | 2     la    li    &     la    li  |
Hand:    | R     L     R     R     L     L   | L     R     L     L     R     R   |
Type:    | A     t     d1    d2    d1    d2  | A     t     d1    d2    d1    d2  |
Accent:  | >                                 | >                                 |
```
