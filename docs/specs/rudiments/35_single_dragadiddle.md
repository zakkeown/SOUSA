# PAS #35: Single Dragadiddle

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 35 |
| **Name** | Single Dragadiddle |
| **Category** | drag |
| **Family** | compound drag |
| **Composed Of** | Drag (#31) + tap + diddle; a portmanteau of "drag" and "diddle" -- the accented lead stroke is dragged, followed by a tap on the opposite hand, then a diddle on the lead hand |
| **Related** | #31 Drag (the primitive drag ornament), #16 Single Paradiddle (analogous structure with accent instead of drag), #36 Drag Paradiddle #1 (adds a second tap before the diddle), #37 Drag Paradiddle #2 (adds a second drag before the paradiddle), #6 Double Stroke Open Roll (the diddle component) |
| **NARD Original** | No (added when PAS expanded to 40 in 1984) |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 3 beats |
| **Strokes Per Cycle** | 12 (8 primary strokes + 4 grace notes) |
| **Primary Strokes Per Cycle** | 8 |

The single dragadiddle is the simplest fusion of drag and paradiddle-family elements. Each group spans 1.5 beats (6 sixteenth-note slots) and contains one dragged accent, one tap, and one diddle (two strokes). The drag replaces what would be a plain accented lead stroke in a paradiddle, creating the portmanteau name "drag-a-diddle." The pattern is hand-to-hand: each group alternates the lead hand, with the diddle at the end forcing the opposite hand to lead the next group. Two groups (right-lead and left-lead) form one complete 3-beat cycle.

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/16 | (1st drag grace before 1) | L | g |
| 2 | -1/32 | (2nd drag grace before 1) | L | g |
| 3 | 0 | 1 | R | A |
| 4 | 1/4 | 1e | L | t |
| 5 | 1/2 | 1& | R | d1 |
| 6 | 3/4 | 1a | R | d2 |
| 7 | 1 + 1/2 - 1/16 | (1st drag grace before 2&) | R | g |
| 8 | 1 + 1/2 - 1/32 | (2nd drag grace before 2&) | R | g |
| 9 | 1 + 1/2 | 2& | L | A |
| 10 | 1 + 3/4 | 2a | R | t |
| 11 | 2 | 3 | L | d1 |
| 12 | 2 + 1/4 | 3e | L | d2 |

Note: Each group occupies 4 sixteenth-note positions: the dragged accent on the first, the tap on the second, and the diddle pair on the third and fourth. The first group begins on beat 1; the second group begins on beat 2&. The remaining sixteenth-note positions (beat 2 and 2e) serve as transition space between groups.

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>              |  >
llR   L  R R   |  rrL   R  L L
```

Each group is a drag (two grace notes + accented primary) followed by a tap on the opposite hand, then a diddle on the lead hand. The name "dragadiddle" captures this structure: drag + (a) + diddle.

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | drag grace 1 | #3 | First drag grace note, group 1 |
| 2 | L | grace | drag grace 2 | #3 | Second drag grace note, group 1 |
| 3 | R | accent | primary | n/a | Right-hand drag primary (accented lead stroke) |
| 4 | L | tap | none | n/a | Left-hand tap following the drag |
| 5 | R | d1 | none | n/a | First stroke of right-hand diddle |
| 6 | R | d2 | none | n/a | Second stroke of right-hand diddle |
| 7 | R | grace | drag grace 1 | #9 | First drag grace note, group 2 |
| 8 | R | grace | drag grace 2 | #9 | Second drag grace note, group 2 |
| 9 | L | accent | primary | n/a | Left-hand drag primary (accented lead stroke) |
| 10 | R | tap | none | n/a | Right-hand tap following the drag |
| 11 | L | d1 | none | n/a | First stroke of left-hand diddle |
| 12 | L | d2 | none | n/a | Second stroke of left-hand diddle |

### Ornament Timing

**Drag Grace Notes:**
- First grace offset from primary: **-1/16 beat** (the earlier of the two drag grace notes)
- Second grace offset from primary: **-1/32 beat** (the later of the two, closer to the primary)
- Both grace notes played by the **same hand** (opposite from the primary stroke)
- Strokes #1-2 (both L) precede stroke #3 (R primary); strokes #7-8 (both R) precede stroke #9 (L primary)
- The diddle-to-grace transition between groups (strokes #5-6 R diddle, then #7-8 R grace) keeps the same hand active across the group boundary. After completing the diddle, the hand must transition from a low/tap height diddle d2 to producing drag grace notes. Since both the diddle d2 and drag grace notes are at tap height, this transition is physically natural.

**Diddle Timing:**
- Each diddle pair occupies one eighth-note duration (1/2 beat)
- Each individual diddle stroke occupies 1/4 beat (one sixteenth note)
- d1 falls on the "&" of the beat; d2 falls on the "a" of the beat
- At slow tempos: both strokes are distinct and clearly separated (open)
- At fast tempos: strokes closely spaced, nearly indistinguishable (closed)

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (drag primary) | 1.0 | Accented lead stroke preceded by drag grace notes |
| tap | 0.65 - 0.77 | Unaccented single stroke following the drag |
| d1 | 0.65 - 0.77 | First diddle stroke at tap level |
| d2 | 0.90 - 0.98 x d1 | Slight decay on second bounce of diddle |
| grace (drag) | 0.45 - 0.65 | Each of the four drag grace notes |

The single dragadiddle has a four-level dynamic hierarchy: grace notes (softest), d2 (slightly softer than d1), tap and d1 (medium), accent (loudest). The drag ornament creates a soft ruff onset before the loud accent, the tap provides a medium-level stroke, and the diddle tapers off at the end of each group. The overall shape is a "buzz-LOUD-medium-fade" contour per group.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>     -  - -   |  >     -  - -
llR   L  R R   |  rrL   R  L L
```

One accent per group (on the dragged lead stroke). The tap and diddle strokes are all unaccented. This matches the single paradiddle's accent pattern with the drag ornament replacing the plain accent.

### Dynamic Contour

The single dragadiddle produces a repeating contour within each 1.5-beat group: the two grace notes create a soft "ruff" onset, the drag primary lands as the peak accent, the tap drops to medium level, and the diddle pair tapers with a slight decay from d1 to d2. The overall shape is "buzz-PEAK-drop-taper." Compared to the single paradiddle (#16), the dragadiddle's drag ornament adds more textural fill before the accent, giving it a busier, more decorated lead-in. Compared to the single drag tap (#32), the dragadiddle extends the post-accent tail with a diddle rather than just a single tap, creating a longer phrase that provides more rhythmic momentum.

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
| 5 | R | d1 | low (2-6") | wrist |
| 6 | R | d2 | tap (< 2") | fingers / rebound |
| 7 | R | grace (drag 1) | tap (< 2") | fingers |
| 8 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 9 | L | accent | full (12"+) | wrist + arm |
| 10 | R | tap | low (2-6") | wrist |
| 11 | L | d1 | low (2-6") | wrist |
| 12 | L | d2 | tap (< 2") | fingers / rebound |

The physical demands of the single dragadiddle combine drag and diddle techniques. The right hand's path through the first group illustrates the key transitions: it plays a full-height drag primary (#3), then drops to low height for the diddle d1 (#5), followed by a finger-controlled d2 (#6). The d2-to-drag-grace transition (strokes #6-7, same hand) is physically natural since both are at tap height -- the hand simply continues its low-stick-height work from the diddle rebound directly into the drag grace notes for the next group. The left hand transitions from drag grace notes (#1-2) to tap (#4), requiring a shift from finger-controlled bounces to a slightly higher wrist stroke.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accented drag primaries. Grace notes individually articulated as two distinct soft strokes (open drag). Clear separation between all four primary strokes per group. Diddles are open with two distinct wrist strokes. The group structure is clearly audible: ruff-ACCENT-tap-bounce-bounce. |
| Moderate (100-140 BPM) | Wrist-driven strokes. Drag grace notes close up to standard ruff sound. Diddle shifts to wrist-finger technique (d1 wrist, d2 finger rebound). The d2-to-drag-grace transition becomes smoother as the finger rebound from d2 naturally feeds into the grace note production. The overall phrase flows as a continuous alternating pattern. |
| Fast (140-160 BPM) | Finger control for grace notes, taps, and diddle d2. Accents drop to half height class. Drag grace notes compress toward -1/32 and -1/64. Diddles become nearly closed. The rudiment begins to feel like a continuous stream of decorated paradiddle-like phrases. The drag's ruff blurs into a textural buzz before each accent. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Dragadiddle with open drags**: Exaggerated drag spacing for slow practice and ornament clarity
- **Dragadiddle without drag**: Practice as accent-tap-diddle (R L R R | L R L L) -- this is equivalent to the single paradiddle (#16), isolating the rhythmic structure
- **Double dragadiddle**: Adding a second tap between the drag accent and the diddle (not a standard PAS rudiment, but a logical extension analogous to single vs. double paradiddle)
- **Dragadiddle with accent on the diddle**: Shifting the accent to the d1 for coordination development (non-standard exercise)

### Prerequisites

- #31 Drag -- the basic drag ornament must be consistent on both hands
- #16 Single Paradiddle -- the accent-tap-diddle sticking pattern forms the underlying structure
- #6 Double Stroke Open Roll -- controlled diddles at various tempos
- #32 Single Drag Tap -- combining drag with a following tap develops the drag-to-tap transition

### Builds Toward

- #36 Drag Paradiddle #1 (adds a second tap, extending the paradiddle portion)
- #37 Drag Paradiddle #2 (adds a second drag, creating a more complex pattern)
- Application in drum corps and marching percussion literature where drag-decorated paradiddle patterns provide textural variety

### Teaching Notes

The single dragadiddle is the most natural entry point into the compound drag rudiments. It is simply a single paradiddle with a drag on the accented lead stroke -- if a student already plays the single paradiddle and the basic drag, combining them into the dragadiddle is conceptually straightforward. The name itself -- a portmanteau of "drag" and "diddle" -- captures the essence of the rudiment.

**Common mistakes:**
- Treating the drag as a flam: the drag uses TWO same-hand grace notes (a double bounce), not one opposite-hand grace note. This is the most common error for students coming from flam paradiddle (#24) practice
- Uneven diddle: d2 significantly softer or rushed relative to d1. The diddle should be even and controlled
- Losing the accent-tap contrast: the drag accent should be clearly louder than the following tap and diddle
- Rushing the diddle: students often compress the diddle to get to the next group's drag sooner. The diddle should occupy its full rhythmic value
- Inconsistent drag quality between hands: right-hand drags (llR) should sound identical to left-hand drags (rrL)
- Grace notes too loud: the drag ruff should be a subtle ornament, not three equal strokes

**Practice strategies:**
- Start with single paradiddles (R L R R | L R L L) with a strong accent on the lead stroke
- Add drag grace notes to the accented stroke one hand at a time
- Practice the d2-to-drag-grace transition in isolation: play a diddle (R R), then immediately produce two grace notes for the next drag
- Use a metronome with the click on the drag accent (every 1.5 beats)
- Practice slowly, listening for four distinct dynamic levels: grace (soft), tap/d1 (medium), d2 (slight decay), accent (full)
- Compare right-lead and left-lead groups for consistency

**What to listen for:**
- Clean drag ornament (two soft grace notes) before each accent
- Clear accent-tap dynamic contrast
- Even diddle with natural d2 decay
- Smooth transition between groups (d2 to drag grace on the same hand)
- Identical quality between right-lead and left-lead groups
- The overall feel of a paradiddle with a decorated lead stroke

### Historical Context

The single dragadiddle was not part of the original NARD 26 rudiments. It was added when the Percussive Arts Society expanded the list to 40 international drum rudiments in 1984. The rudiment fills a logical gap in the drag category: where the flam paradiddle (#24) combines a flam with the paradiddle sticking, the single dragadiddle combines a drag with a simplified paradiddle structure. The name "dragadiddle" is a portmanteau coined to describe the fusion of drag and diddle elements. Despite its relatively recent formal recognition, the pattern almost certainly existed in rudimental practice long before its codification -- any drummer practicing drags and paradiddles would naturally encounter this combination. The dragadiddle serves as a stepping stone to the more complex drag paradiddles (#36 and #37), which extend the pattern with additional strokes.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>                  >
llR   L   R R  |   rrL   R   L L
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1     e     &     a   | 2                 | 2&    2a    3     3e  |
Hand:    | (ll)R L     R     R   |                   | (rr)L R     L     L   |
Type:    |  gg A t     d1    d2  |                   |  gg A t     d1    d2  |
Accent:  |     >                 |                   |      >                |
```

Grace notes shown in doubled parentheses indicate two same-hand grace notes (drag) falling just before the grid position. The first grace is at -1/16 beat and the second at -1/32 beat from the primary. The first group occupies beat 1 (4 sixteenth-note slots). The second group occupies beats 2& through 3e (4 sixteenth-note slots). Beat 2 and 2e are transition space between groups.
