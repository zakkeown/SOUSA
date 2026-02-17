# PAS #32: Single Drag Tap

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 32 |
| **Name** | Single Drag Tap |
| **Category** | drag |
| **Family** | basic drag |
| **Composed Of** | Drag (#31) + tap; each group is a drag (two grace notes + accented primary) followed by a single tap on the opposite hand |
| **Related** | #31 Drag (the primitive drag ornament), #33 Double Drag Tap (two drags + tap), #22 Flam Tap (analogous structure using flams instead of drags), #34 Lesson 25 (drag-based pattern with different accent structure) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 2 beats |
| **Strokes Per Cycle** | 8 (4 primary strokes + 4 grace notes) |
| **Primary Strokes Per Cycle** | 4 |

The single drag tap places one drag followed by one tap in each beat, alternating the lead hand every beat. The drag primary falls on the downbeat of each beat and the tap falls on the "&" (halfway through), creating a steady eighth-note pulse of primary strokes. The two grace notes precede each drag primary, filling the space just before the beat. The pattern spans 2 beats to complete one full left-right alternation.

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/16 | (1st grace before 1) | L | g |
| 2 | -1/32 | (2nd grace before 1) | L | g |
| 3 | 0 | 1 | R | A |
| 4 | 1/2 | 1& | L | t |
| 5 | 1 - 1/16 | (1st grace before 2) | R | g |
| 6 | 1 - 1/32 | (2nd grace before 2) | R | g |
| 7 | 1 | 2 | L | A |
| 8 | 1 + 1/2 | 2& | R | t |

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>        -     |  >        -
llR      L     |  rrL      R
```

Each beat contains a drag (two grace notes + accented primary) followed by a single tap on the opposite hand. The lead hand alternates every beat.

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | drag grace 1 | #3 | First drag grace note, same hand as #2 |
| 2 | L | grace | drag grace 2 | #3 | Second drag grace note, closer to primary |
| 3 | R | accent | primary | n/a | Right-hand drag primary (accented) |
| 4 | L | tap | none | n/a | Left-hand tap following the drag |
| 5 | R | grace | drag grace 1 | #7 | First drag grace note, same hand as #6 |
| 6 | R | grace | drag grace 2 | #7 | Second drag grace note, closer to primary |
| 7 | L | accent | primary | n/a | Left-hand drag primary (accented) |
| 8 | R | tap | none | n/a | Right-hand tap following the drag |

### Ornament Timing

**Drag Grace Notes:**
- First grace offset from primary: **-1/16 beat** (the earlier of the two drag grace notes)
- Second grace offset from primary: **-1/32 beat** (the later of the two, closer to the primary)
- Both grace notes played by the **same hand** (opposite from the primary stroke)
- Strokes #1-2 (both L) precede stroke #3 (R primary); strokes #5-6 (both R) precede stroke #7 (L primary)
- The two grace notes form a quick diddle (same-hand double bounce) leading into the primary
- Important: the tap (stroke #4 or #8) and the following drag grace pair (strokes #5-6 or #1-2 of the next cycle) are played by **different hands**. Unlike the flam tap (#22), where the tap-to-grace transition occurs on the same hand, the single drag tap's tap-to-grace transition crosses hands, making it somewhat more forgiving technically.

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (drag primary) | 1.0 | Accented primary strokes preceded by drag grace notes |
| tap | 0.65 - 0.77 | Unaccented single stroke following each drag |
| grace (drag) | 0.45 - 0.65 | Each of the two drag grace notes |

The single drag tap has a three-level dynamic hierarchy: grace notes (softest), tap (medium), accent (loudest). The grace notes create a soft ruff leading into the full accent, and the tap provides a dynamic valley after it.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>     -     |  >     -
llR   L     |  rrL   R
```

One accent per beat (on the drag primary), with the tap unaccented between.

### Dynamic Contour

The single drag tap produces a repeating "ruff-accent-valley" pattern at the beat level: the two grace notes create a soft onset, the drag primary lands as a full accent, and the tap drops to an unaccented level. This "buzz-LOUD-soft" shape repeats each beat, alternating hands. The overall feel is a steady eighth-note pulse with the on-beat strokes emphasized by the drag ornament and the off-beat taps providing rhythmic continuity at a lower dynamic. Compared to the flam tap (#22), which has a similar accent-tap structure, the single drag tap's drag ornament adds more rhythmic "fill" before the accent, giving it a slightly busier, more textured sound.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace (drag 1) | tap (< 2") | fingers |
| 2 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 3 | R | accent | full (12"+) | wrist + arm |
| 4 | L | tap | low (2-6") | wrist |
| 5 | R | grace (drag 1) | tap (< 2") | fingers |
| 6 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 7 | L | accent | full (12"+) | wrist + arm |
| 8 | R | tap | low (2-6") | wrist |

Consider the left hand's path through the first beat: it plays two grace notes at tap height (#1-2, fingers/rebound), then after the right hand plays the accent (#3), the left hand plays a tap at low height (#4, wrist). The grace-to-tap transition on the same hand requires shifting from finger-controlled bounces to a wrist-driven tap stroke with slightly more height. In the second beat, the roles reverse. The right hand's path follows the same sequence: grace-grace (before beat 2), then tap (on 2&).

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accented drag primaries. Grace notes are individually articulated as two distinct soft strokes (open drag). Clear separation between grace notes, primary, and tap. The double bounce is controlled with active finger strokes for both graces. The overall pattern feels like four distinct events per beat: bounce-bounce-ACCENT-tap. |
| Moderate (100-140 BPM) | Wrist-driven strokes. Drag grace notes close up, with the second relying on rebound. The drag sounds more like a ruff than two separate notes. Grace offsets at standard -1/16 and -1/32. The pattern flows as a continuous alternating sequence. The tap should be clearly softer than the accent but still rhythmically placed. |
| Fast (140-160 BPM) | Finger control for grace notes and taps. Accents drop to half height class. Drag grace notes compress toward -1/32 and -1/64 offsets, approaching a crushed ruff. The tap provides rhythmic continuity while the drags become more textural (buzz-like) than articulate. The rudiment begins to feel like a driving eighth-note pattern with buzz accents on the beats. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Double drag tap (#33)**: Two drags before the tap, extending the pattern to 3 beats per group
- **Unaccented drag**: Playing the drag primary as a tap for a softer, more flowing variation
- **No-ornament preparation**: Practice as accent-tap (>R L | >L R) to isolate the rhythm and accent pattern before adding drag grace notes
- **Single drag tap in triplets**: Some interpretations swing the pattern, placing the drag and tap in triplet subdivision rather than sixteenth notes

### Prerequisites

- #31 Drag -- the basic drag ornament must be consistent on both hands
- #1 Single Stroke Roll -- basic hand alternation
- #6 Double Stroke Open Roll -- controlled double bounce technique for the grace notes
- Ability to transition from a tap-height double bounce to a full accent on the opposite hand

### Builds Toward

- #33 Double Drag Tap (adds a second drag before the tap)
- #34 Lesson 25 (drag combined with single strokes in a longer pattern)
- #35 Single Dragadiddle (drag combined with paradiddle)
- #30 Flam Drag (flam + drag combination; the single drag tap develops relevant coordination)
- Application in drum corps, concert, and marching percussion literature

### Teaching Notes

The single drag tap is the first compound drag rudiment, combining the basic drag (#31) with a simple tap. It is the drag-category analog of the flam tap (#22) and introduces the concept of integrating drag ornaments into a rhythmic pattern.

**Common mistakes:**
- Grace notes too loud: The ruff overpowers the accent, producing a triplet feel rather than a decorated accent
- Missing or collapsed drag: Only one grace note sounds, converting the drag into a flam. Ensure both grace notes are present
- Uneven beat spacing: The drag primary and tap should be evenly spaced (each occupying an eighth-note position). Students often rush the tap or delay it
- Inconsistent drag quality between hands: Left-hand drags (rrL) should sound identical to right-hand drags (llR)
- Tap and drag grace at the same volume: The tap should be clearly softer than the accent but louder than the grace notes
- Rushing into the drag: The grace notes should lead smoothly into the primary, not be jammed against it

**Practice strategies:**
- Start with the accent-tap pattern without ornaments: >R L | >L R, using a metronome
- Practice drags separately on each hand to solidify the double bounce
- Add grace notes to the accent-tap pattern one hand at a time: llR L first, then add rrL R
- Focus on the dynamic hierarchy: grace (soft) < tap (medium) < accent (loud)
- Practice slowly and gradually increase tempo, listening for consistent drag quality on both hands
- Record and compare the left-lead and right-lead halves for balance

**What to listen for:**
- Clear three-level dynamic hierarchy: grace (soft), tap (medium), accent (loud)
- Both grace notes present in each drag (not collapsed to a flam)
- Even eighth-note spacing between the drag primary and the tap
- Identical drag quality on both hands
- The ruff effect should color the accent without overpowering it
- Smooth alternation between the two halves of the cycle

### Historical Context

The single drag tap was included in the original NARD 26 rudiments and has been a standard in rudimental drumming since the earliest codification of American field drumming techniques. The rudiment was retained as PAS #32 in the 1984 expansion to 40 rudiments. Its relationship to the drag (#31) parallels the flam tap's (#22) relationship to the flam (#20): both take the basic ornament and place it in a simple accent-tap rhythmic framework. The single drag tap appears frequently in military field drum music, drum corps snare drum literature, and rudimental solos. In John S. Pratt's influential *26 Standard American Drum Rudiments* method and in the Percussive Arts Society's instructional materials, the single drag tap is presented immediately after the basic drag as the natural first step in building drag-based vocabulary.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>           >
llR   L  |  rrL   R
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1           &     | 2           &     |
Hand:    | (ll)R       L     | (rr)L       R     |
Type:    |  gg  A      t     |  gg  A      t     |
Accent:  |      >            |      >            |
```

Grace notes shown in doubled parentheses indicate two same-hand grace notes falling just before the grid position. The first grace is at -1/16 beat and the second at -1/32 beat from the primary. Drag primaries land on beats 1 and 2, while taps fall on the "&" positions.
