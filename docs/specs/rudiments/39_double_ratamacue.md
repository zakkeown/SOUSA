# PAS #39: Double Ratamacue

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 39 |
| **Name** | Double Ratamacue |
| **Category** | drag |
| **Family** | ratamacue |
| **Composed Of** | Two drags (#31) + three single strokes ending with an accent; the single ratamacue (#38) with an additional drag prepended |
| **Related** | #38 Single Ratamacue (one drag + singles + accent), #40 Triple Ratamacue (three drags + singles + accent), #31 Drag (the primitive drag ornament), #33 Double Drag Tap (two drags + tap, analogous "double" extension in the basic drag family) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 3 beats |
| **Strokes Per Cycle** | 18 (10 primary strokes + 8 grace notes) |
| **Primary Strokes Per Cycle** | 10 |

The double ratamacue extends the single ratamacue (#38) by adding a second drag at the beginning of each group. Where the single ratamacue has one drag followed by three single strokes ending with an accent (4 primary strokes per group), the double ratamacue has two drags followed by three single strokes ending with an accent (5 primary strokes per group). Each group spans 1.25 beats (5 sixteenth notes), and two groups fill the 3-beat cycle (5 + 5 = 10 sixteenth notes across 2.5 beats of primary strokes, with the remaining half-beat accommodated by the drag grace notes and the pickup into the next cycle). The two consecutive drags at the beginning of each group create a longer, more elaborate ruff sequence before the pattern resolves on the closing accent, intensifying the "build-and-release" character of the ratamacue.

The five primary strokes per group fall on consecutive sixteenth-note positions: two drag primaries (each preceded by grace notes), then three single strokes with the accent on the last. The groups alternate lead hands: the first group is right-lead, the second is left-lead.

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/16 | (1st grace before 1) | L | g |
| 2 | -1/32 | (2nd grace before 1) | L | g |
| 3 | 0 | 1 | R | t |
| 4 | 1/4 - 1/16 | (1st grace before 1e) | R | g |
| 5 | 1/4 - 1/32 | (2nd grace before 1e) | R | g |
| 6 | 1/4 | 1e | L | t |
| 7 | 1/2 | 1& | R | t |
| 8 | 3/4 | 1a | L | t |
| 9 | 1 | 2 | R | A |
| 10 | 1 + 1/4 - 1/16 | (1st grace before 2e) | R | g |
| 11 | 1 + 1/4 - 1/32 | (2nd grace before 2e) | R | g |
| 12 | 1 + 1/4 | 2e | L | t |
| 13 | 1 + 1/2 - 1/16 | (1st grace before 2&) | L | g |
| 14 | 1 + 1/2 - 1/32 | (2nd grace before 2&) | L | g |
| 15 | 1 + 1/2 | 2& | R | t |
| 16 | 1 + 3/4 | 2a | L | t |
| 17 | 2 | 3 | R | t |
| 18 | 2 + 1/4 | 3e | L | A |

Note: The first group (strokes #1-9) spans from beat 1 through beat 2: two drags on 1 and 1e, then taps on 1&, 1a, and accent on 2. The second group (strokes #10-18) spans from beat 2e through 3e: two drags on 2e and 2&, then taps on 2a, 3, and accent on 3e. The lead hand alternates between groups.

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
                        >  |                          >
llR  rrL  R  L  R      |  rrL  llR  L  R  L
```

Each group consists of two drags (alternating lead hand for each drag) followed by three single strokes with the accent on the last. The first group begins with a right-hand drag primary, the second with a left-hand drag primary. See [#38 Single Ratamacue](./38_single_ratamacue.md) for the shared "ra-ta-ma-cue" accent pattern that concludes each group.

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | drag grace 1 | #3 | First grace, first drag of group 1 |
| 2 | L | grace | drag grace 2 | #3 | Second grace, first drag of group 1 |
| 3 | R | tap | primary | n/a | First drag primary (unaccented), group 1 |
| 4 | R | grace | drag grace 1 | #6 | First grace, second drag of group 1 |
| 5 | R | grace | drag grace 2 | #6 | Second grace, second drag of group 1 |
| 6 | L | tap | primary | n/a | Second drag primary (unaccented), group 1 |
| 7 | R | tap | none | n/a | First single stroke; the "ta" |
| 8 | L | tap | none | n/a | Second single stroke; the "ma" |
| 9 | R | accent | none | n/a | Closing accent; the "cue" |
| 10 | R | grace | drag grace 1 | #12 | First grace, first drag of group 2 |
| 11 | R | grace | drag grace 2 | #12 | Second grace, first drag of group 2 |
| 12 | L | tap | primary | n/a | First drag primary (unaccented), group 2 |
| 13 | L | grace | drag grace 1 | #15 | First grace, second drag of group 2 |
| 14 | L | grace | drag grace 2 | #15 | Second grace, second drag of group 2 |
| 15 | R | tap | primary | n/a | Second drag primary (unaccented), group 2 |
| 16 | L | tap | none | n/a | First single stroke; the "ta" |
| 17 | R | tap | none | n/a | Second single stroke; the "ma" |
| 18 | L | accent | none | n/a | Closing accent; the "cue" |

### Ornament Timing

**Drag Grace Notes:**
- First grace offset from primary: **-1/16 beat** (the earlier of the two drag grace notes)
- Second grace offset from primary: **-1/32 beat** (the later of the two, closer to the primary)
- Both grace notes played by the **same hand** (opposite from the primary stroke)
- Each drag follows the standard ornament timing from #31 Drag
- There are four drags per cycle, each with its own pair of grace notes:
  - Strokes #1-2 (L) precede #3 (R); strokes #4-5 (R) precede #6 (L)
  - Strokes #10-11 (R) precede #12 (L); strokes #13-14 (L) precede #15 (R)
- Critical transition: Between the first and second drag within each group, the hand that just played a drag primary (#3 or #12) must immediately produce two grace notes (#4-5 or #13-14) for the next drag. This drag-primary-to-drag-grace transition on the same hand is the same technical challenge as in the double drag tap (#33).
- As in the single ratamacue (#38), the drag primaries are **taps** (unaccented), not accents. The accent is reserved for the final stroke of each group.

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent ("cue") | 1.0 | Closing accent, the loudest stroke in each group |
| tap (drag primaries + singles) | 0.65 - 0.77 | All drag primaries and middle taps are unaccented |
| grace (drag) | 0.45 - 0.65 | Each of the eight drag grace notes |

The same three-level dynamic hierarchy as the single ratamacue (#38): grace (softest), tap (medium), accent (loudest). The two consecutive drags at the beginning of each group extend the soft-to-medium buildup, making the closing accent feel even more emphatic by contrast.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
-     -     -  -  >  |  -     -     -  -  >
llR   rrL   R  L  R  |  rrL   llR   L  R  L
```

One accent per group (on the final stroke), with all preceding strokes unaccented. The accent placement creates the same off-beat emphasis as the single ratamacue.

### Dynamic Contour

The double ratamacue extends the single ratamacue's "build-and-release" contour by adding a longer preparation phase. Each group begins with two drag-preceded taps (four grace notes and two primaries), continues through two more taps, and resolves on a strong accent. The extended ruff-ruff-tap-tap-ACCENT shape creates a more dramatic buildup than the single ratamacue's single-ruff version. The two consecutive drags at the beginning produce a sense of gathering momentum -- a double "buzz" that propels the pattern forward through the taps into the accent resolution. The overall effect is a more intense, more dramatic version of the ratamacue's characteristic crescendo-to-accent shape.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace (drag 1) | tap (< 2") | fingers |
| 2 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 3 | R | tap (drag primary) | low (2-6") | wrist |
| 4 | R | grace (drag 1) | tap (< 2") | fingers |
| 5 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 6 | L | tap (drag primary) | low (2-6") | wrist |
| 7 | R | tap | low (2-6") | wrist |
| 8 | L | tap | low (2-6") | wrist |
| 9 | R | accent | full (12"+) | wrist + arm |
| 10 | R | grace (drag 1) | tap (< 2") | fingers |
| 11 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 12 | L | tap (drag primary) | low (2-6") | wrist |
| 13 | L | grace (drag 1) | tap (< 2") | fingers |
| 14 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 15 | R | tap (drag primary) | low (2-6") | wrist |
| 16 | L | tap | low (2-6") | wrist |
| 17 | R | tap | low (2-6") | wrist |
| 18 | L | accent | full (12"+) | wrist + arm |

The key physical challenge of the double ratamacue is the drag-primary-to-drag-grace transition between the first and second drags. Consider the right hand in group 1: it plays a tap-height drag primary (#3), then must immediately drop to produce two finger-controlled grace notes (#4-5) for the left hand's drag. This tap-to-grace transition on the same hand requires rapidly switching from a wrist-driven tap to a finger-controlled double bounce. This challenge is shared with the double drag tap (#33), but in the ratamacue context the drag primaries are taps rather than accents, making the height transition slightly less extreme.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accents. Grace notes individually articulated as open drags. All five primary strokes per group are clearly separated. The two consecutive drags are audible as distinct ruff events. The drag-primary-to-grace transition is manageable due to wider spacing. Each stroke is individually controllable. |
| Moderate (100-120 BPM) | Wrist-driven strokes. Drag grace notes close up to standard ruff sound. The two drags flow together as a connected pair of ruffs. The drag-primary-to-grace transition tightens, requiring efficient bounce technique. Standard grace offsets at -1/16 and -1/32. The five primary strokes and accent flow as a connected group. |
| Fast (120-140 BPM) | Finger control for grace notes and taps. Accent drops to half height class. Drag grace notes compress toward -1/32 and -1/64 offsets. The two drags at the beginning blur into a continuous buzzing preparation before the single strokes and accent. The pattern approaches a stream of sixteenths with embedded ruffs and an off-beat accent. |

---

## 6. Variations & Pedagogy

### Common Variations

- **6/8 interpretation**: Some traditions notate the double ratamacue in 6/8 time rather than 4/4, grouping the strokes differently while maintaining the same sound.
- **No-ornament preparation**: Practice as five alternating single strokes with the accent on the last (R L R L >R | L R L R >L) to isolate the rhythm before adding drags.
- **Continuous double ratamacues**: Linking multiple patterns end-to-end for extended rudimental passages.
- **Ratamacue progression**: Practicing single, double, and triple ratamacues in sequence as a graduated exercise.

### Prerequisites

- #38 Single Ratamacue -- the simpler ratamacue pattern must be solid before adding a second drag
- #31 Drag -- consistent drag ornament on both hands
- #33 Double Drag Tap -- the drag-primary-to-drag-grace transition is the same technical challenge
- #6 Double Stroke Open Roll -- controlled double bounce technique

### Builds Toward

- #40 Triple Ratamacue (adds a third drag at the beginning)
- Extended rudimental solo passages combining ratamacue variants
- Drum corps and marching percussion literature

### Teaching Notes

The double ratamacue adds a second drag to the beginning of the single ratamacue pattern. The primary new challenge is the drag-primary-to-drag-grace transition between the two consecutive drags, which is the same skill required in the double drag tap (#33). See [#38 Single Ratamacue](./38_single_ratamacue.md) for foundational teaching guidance on the ratamacue's accent placement and dynamic contour.

**Common mistakes:**
- Accenting the drag primaries: As in the single ratamacue, the drag primaries are taps, not accents. Only the final stroke of each group should be accented.
- Collapsing the second drag: At speed, the second drag often loses one or both grace notes, reducing it to a flam or plain tap.
- Uneven drags: Both drags within each group should have identical ruff quality. Students often execute the first drag well but rush or compress the second.
- Losing count: With five primary strokes per group plus grace notes, students can lose track of position. Counting "1-e-&-a-2" for the primary strokes helps maintain orientation.
- Rushing into the accent: The taps leading into the closing accent must maintain even sixteenth-note spacing.
- Inconsistent drag quality across the four drags per cycle.

**Practice strategies:**
- Master the single ratamacue (#38) before adding the second drag.
- Practice the drag-primary-to-drag-grace transition in isolation (as in #33 Double Drag Tap).
- Build up incrementally: practice with only the first drag (single ratamacue), then add the second drag.
- Count the primary strokes: "1-e-&-a-2" for group 1, aligning with the sixteenth-note grid.
- Practice each group separately (right-lead, then left-lead) before alternating.
- Compare all four drags per cycle for equal quality.

**What to listen for:**
- The extended "build-and-release" contour: two ruffs leading through taps to a strong accent.
- All four drags per cycle should have identical ruff quality.
- Clear dynamic separation between taps and the closing accent.
- Even sixteenth-note spacing across all five primary strokes per group.
- Both groups (right-lead and left-lead) should sound identical.

### Historical Context

The double ratamacue was included in the original NARD 26 rudiments and has been practiced in American rudimental drumming traditions since at least the early 19th century. It was retained as PAS #39 when the Percussive Arts Society expanded the rudiment list to 40 in 1984. The relationship between the single (#38) and double (#39) ratamacue mirrors the progressive "single then double" pattern seen throughout the PAS rudiments (cf. #32 Single Drag Tap to #33 Double Drag Tap, #16 Single Paradiddle to #17 Double Paradiddle). The additional drag at the beginning creates a more elaborate and dramatic pattern that features prominently in drum corps snare drum solos and rudimental contest pieces, where its extended buildup provides dramatic phrasing and forward momentum.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
                           >  |                             >
llR   rrL   R   L   R     |  rrL   llR   L   R   L
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1        e        &     a   | 2        e        &     a   | 3     e        |
Hand:    | (ll)R    (rr)L    R     L   | R        (rr)L    (ll)R  L  | R     L        |
Type:    |  gg  t    gg  t   t     t   | A         gg  t    gg  t  t | t     A        |
Accent:  |                             | >                           |       >        |
```

Grace notes shown in doubled parentheses indicate two same-hand grace notes falling just before the grid position. The first grace is at -1/16 beat and the second at -1/32 beat from the primary. The first group occupies beats 1 through 2 (drags on 1 and 1e, taps on 1& and 1a, accent on 2). The second group occupies beats 2e through 3e (drags on 2e and 2&, taps on 2a and 3, accent on 3e).
