# PAS #36: Drag Paradiddle #1

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 36 |
| **Name** | Drag Paradiddle #1 |
| **Category** | drag |
| **Family** | compound drag |
| **Composed Of** | Drag (#31) + Single Paradiddle (#16); a single paradiddle whose accented lead stroke is preceded by drag grace notes -- structurally equivalent to replacing the paradiddle's plain accent with a drag |
| **Related** | #31 Drag (the primitive drag ornament), #16 Single Paradiddle (the underlying paradiddle sticking), #24 Flam Paradiddle (analogous structure using flam instead of drag), #35 Single Dragadiddle (simpler drag + diddle combination with one fewer tap), #37 Drag Paradiddle #2 (extends with a second drag) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 3.5 beats |
| **Strokes Per Cycle** | 14 (10 primary strokes + 4 grace notes) |
| **Primary Strokes Per Cycle** | 10 |

Drag Paradiddle #1 is the drag-category counterpart of the flam paradiddle (#24). Each group contains a full single paradiddle sticking (accent + two taps + diddle = 5 primary strokes) with the accented lead stroke preceded by drag grace notes. The pattern spans 3.5 beats for one full left-right alternation (two groups of 5 primary strokes plus 2 drag grace notes each). The 10 primary strokes fall on the sixteenth-note grid, distributed as two groups of 5 strokes occupying 5 sixteenth-note positions each. The three remaining sixteenth-note slots (in a 3.5-beat / 14-slot framework) provide rest or transition space between groups.

The critical difference from the single dragadiddle (#35) is the addition of a second tap between the drag accent and the diddle, creating the full paradiddle sticking (accent-tap-tap-diddle) rather than the shortened dragadiddle sticking (accent-tap-diddle).

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/16 | (1st drag grace before 1) | L | g |
| 2 | -1/32 | (2nd drag grace before 1) | L | g |
| 3 | 0 | 1 | R | A |
| 4 | 1/4 | 1e | L | t |
| 5 | 1/2 | 1& | R | t |
| 6 | 3/4 | 1a | L | d1 |
| 7 | 1 | 2 | L | d2 |
| 8 | 1 + 3/4 - 1/16 | (1st drag grace before 2a) | R | g |
| 9 | 1 + 3/4 - 1/32 | (2nd drag grace before 2a) | R | g |
| 10 | 1 + 3/4 | 2a | L | A |
| 11 | 2 | 3 | R | t |
| 12 | 2 + 1/4 | 3e | L | t |
| 13 | 2 + 1/2 | 3& | R | d1 |
| 14 | 2 + 3/4 | 3a | R | d2 |

Note: Each group occupies 5 sixteenth-note positions for its primary strokes (accent + tap + tap + d1 + d2) plus 2 grace notes. The first group spans beats 1 through 2 (positions 1, 1e, 1&, 1a, 2). The second group spans beats 2a through 3a (positions 2a, 3, 3e, 3&, 3a). The remaining positions serve as transition space. The diddle at the end of each group forces the opposite hand to lead the next group, maintaining the hand-switching property inherited from the paradiddle.

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>                    |  >
llR   L   R  L L     |  rrL   R   L  R R
```

Each group is a full single paradiddle sticking with a drag on the accented lead stroke. The sticking within each group is: dragged accent + tap (opposite hand) + tap (lead hand) + diddle (opposite hand). The lead hand alternates between groups.

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | drag grace 1 | #3 | First drag grace note, group 1 |
| 2 | L | grace | drag grace 2 | #3 | Second drag grace note, group 1 |
| 3 | R | accent | primary | n/a | Right-hand drag primary (accented paradiddle lead) |
| 4 | L | tap | none | n/a | First tap -- "para" alternation stroke |
| 5 | R | tap | none | n/a | Second tap -- returns to lead hand |
| 6 | L | d1 | none | n/a | First stroke of left-hand diddle |
| 7 | L | d2 | none | n/a | Second stroke of left-hand diddle |
| 8 | R | grace | drag grace 1 | #10 | First drag grace note, group 2 |
| 9 | R | grace | drag grace 2 | #10 | Second drag grace note, group 2 |
| 10 | L | accent | primary | n/a | Left-hand drag primary (accented paradiddle lead) |
| 11 | R | tap | none | n/a | First tap -- "para" alternation stroke |
| 12 | L | tap | none | n/a | Second tap -- returns to lead hand |
| 13 | R | d1 | none | n/a | First stroke of right-hand diddle |
| 14 | R | d2 | none | n/a | Second stroke of right-hand diddle |

### Ornament Timing

**Drag Grace Notes:**
- First grace offset from primary: **-1/16 beat** (the earlier of the two drag grace notes)
- Second grace offset from primary: **-1/32 beat** (the later of the two, closer to the primary)
- Both grace notes played by the **same hand** (opposite from the primary stroke)
- Strokes #1-2 (both L) precede stroke #3 (R primary); strokes #8-9 (both R) precede stroke #10 (L primary)
- The diddle-to-grace transition between groups (strokes #6-7 L diddle, then #8-9 R grace) crosses hands -- the diddle is on one hand while the drag grace notes are on the opposite hand. This means the hand producing the grace notes has been resting during the diddle, giving it time to prepare. This makes the transition easier than in the single dragadiddle (#35), where the same hand plays both the diddle and the following drag grace notes.

**Diddle Timing:**
- Each diddle pair occupies one eighth-note duration (1/2 beat)
- Each individual diddle stroke occupies 1/4 beat (one sixteenth note)
- d1 falls on the "a" of one beat; d2 falls on the downbeat of the next beat
- At slow tempos: both strokes are distinct and clearly separated (open)
- At fast tempos: strokes closely spaced, nearly indistinguishable (closed)

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (drag primary) | 1.0 | Accented lead stroke preceded by drag grace notes |
| tap | 0.65 - 0.77 | Unaccented single strokes (2 per group) |
| d1 | 0.65 - 0.77 | First diddle stroke at tap level |
| d2 | 0.90 - 0.98 x d1 | Slight decay on second bounce of diddle |
| grace (drag) | 0.45 - 0.65 | Each of the four drag grace notes |

The Drag Paradiddle #1 shares the same four-level dynamic hierarchy as the single dragadiddle, but with an additional tap stroke that extends the medium-dynamic "body" of each group. The overall shape is: grace notes (soft ruff onset), accent (peak), two taps (medium plateau), diddle (medium with slight decay).

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>     -  -  - -   |  >     -  -  - -
llR   L  R  L L   |  rrL   R  L  R R
```

One accent per group (on the dragged lead stroke). The two taps and the diddle are all unaccented. This is the standard single paradiddle accent pattern with the lead accent decorated by a drag.

### Dynamic Contour

The Drag Paradiddle #1 produces a repeating "ruff-PEAK-plateau-taper" contour within each group: the drag grace notes create a soft buzz onset, the primary accent is the dynamic peak, the two taps form a medium-level plateau, and the diddle pair tapers with slight decay from d1 to d2. Compared to the single dragadiddle (#35), the additional tap extends the plateau phase, giving the group a longer "body" after the accent. Compared to the flam paradiddle (#24), the drag's double-bounce onset creates a busier lead-in than the flam's single grace note, giving the Drag Paradiddle #1 a more textured, ornamental character at the start of each phrase. The overall feel is a weighted, decorated paradiddle with clear phrasing.

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
| 5 | R | tap | low (2-6") | wrist |
| 6 | L | d1 | low (2-6") | wrist |
| 7 | L | d2 | tap (< 2") | fingers / rebound |
| 8 | R | grace (drag 1) | tap (< 2") | fingers |
| 9 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 10 | L | accent | full (12"+) | wrist + arm |
| 11 | R | tap | low (2-6") | wrist |
| 12 | L | tap | low (2-6") | wrist |
| 13 | R | d1 | low (2-6") | wrist |
| 14 | R | d2 | tap (< 2") | fingers / rebound |

The physical flow of the Drag Paradiddle #1 is smoother than the single dragadiddle because the cross-hand transition between groups is less demanding. The left hand's path through the first group: it plays two drag grace notes at tap height (#1-2, fingers), then rests while the right hand accents (#3), then plays a tap at low height (#4, wrist), rests again while the right hand taps (#5), then plays the diddle (#6-7, wrist to fingers). The right hand meanwhile transitions from full-height accent (#3) through a low-height tap (#5) to drag grace notes (#8-9) for the next group. The accent-to-tap drop-down (#3 to #5 on the right hand) is the familiar downstroke technique needed in all paradiddle-family rudiments.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accented drag primaries. Grace notes individually articulated as open drags. All five primary strokes per group are clearly separated. Diddles are open with two distinct wrist strokes. The paradiddle sticking is easily identifiable within each group. |
| Moderate (100-140 BPM) | Wrist-driven strokes. Drag grace notes close up to standard ruff. Diddles shift to wrist-finger technique. The accent uses a controlled downstroke, catching the stick low for the following taps. The pattern begins to flow as a continuous alternating phrase. The drag adds a subtle "buzz" before each group's lead accent. |
| Fast (140-160 BPM) | Finger control for grace notes, taps, and diddle d2. Accents drop to half height class. Drag grace notes compress toward -1/32 and -1/64. Diddles become nearly closed. The rudiment feels like a paradiddle stream with buzz-accented lead strokes. Moeller technique aids the accent-to-tap transition. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Drag paradiddle without drag**: Practice as a plain accented single paradiddle (>R L R L L | >L R L R R) to solidify the underlying sticking before adding the drag ornament
- **Open drag paradiddle**: Exaggerated drag spacing for slow practice and ornament clarity
- **Alternative sticking**: Some sources notate the pattern with a leading tap before the drag (R llR L R R | L rrL R L L), creating a 6-primary-stroke group. The YAML definition used in this project follows the direct-drag-lead version (llR L R LL | rrL R L RR).
- **Drag paradiddle with inverted diddle**: Moving the diddle to the beginning of the group (non-standard exercise)

### Prerequisites

- #31 Drag -- the basic drag ornament must be consistent on both hands
- #16 Single Paradiddle -- the accent-tap-tap-diddle sticking pattern must be secure
- #35 Single Dragadiddle -- the simpler drag + diddle combination develops the drag-in-paradiddle context
- #6 Double Stroke Open Roll -- controlled diddles at various tempos
- Downstroke control -- ability to play a full drag accent and stop the stick low for the following tap

### Builds Toward

- #37 Drag Paradiddle #2 (extends with a second drag, creating a longer pattern)
- Application in drum corps snare drum parts and rudimental solos where drag-decorated paradiddle patterns are common
- Development of the drag-within-paradiddle coordination applicable to musical contexts

### Teaching Notes

Drag Paradiddle #1 is the drag-category counterpart of the flam paradiddle (#24). If a student can play both the single paradiddle and the basic drag, combining them into the Drag Paradiddle #1 is conceptually straightforward -- simply replace the paradiddle's accented lead stroke with a dragged accent.

**Common mistakes:**
- Treating the drag as a flam: the drag uses TWO same-hand grace notes (double bounce), not one opposite-hand grace note. Students who have recently practiced the flam paradiddle (#24) may inadvertently produce a flam instead of a drag
- Losing the paradiddle sticking: the underlying R L R L L | L R L R R pattern must remain intact. The drag ornament decorates the accent but should not alter the sticking sequence
- Uneven taps: both taps within each group should be at the same dynamic level
- Rushing through the paradiddle body: the two taps and diddle after the accent should maintain even sixteenth-note spacing
- Inconsistent drag quality between right-lead and left-lead groups
- Diddle decay too extreme: d2 should be only slightly softer than d1, not dramatically quieter

**Practice strategies:**
- Start with accented single paradiddles (>R L R L L | >L R L R R) at a comfortable tempo
- Add drag grace notes one hand at a time: first practice llR L R LL only, then add rrL R L RR
- Practice the accent-to-tap downstroke transition: play a full accent, then immediately produce a low tap
- Use a metronome with the click on the drag accent
- Compare the quality of the drag ornament in isolation with its quality within the full pattern
- Record and listen for balanced groups and consistent drag quality on both hands

**What to listen for:**
- Clean drag ornament (two soft grace notes forming a brief ruff) before each accent
- Clear accent-tap contrast (the drag accent should be noticeably louder than all following strokes)
- Even spacing of all five primary strokes per group on the sixteenth-note grid
- Balanced, even diddle with natural d2 decay
- Identical quality between right-lead and left-lead groups
- The paradiddle's characteristic hand-switching feel with the added texture of the drag ornament

### Historical Context

Drag Paradiddle #1 was included in the original NARD 26 rudiments when NARD was established in 1933, with its origins in early American and European field drumming traditions. The concept of combining a drag with paradiddle sticking is a natural extension of both fundamental techniques, and patterns resembling the drag paradiddle appear in military drumming manuals from the 19th century. The rudiment was retained as PAS #36 in the 1984 expansion to 40 rudiments. The "#1" designation distinguishes it from Drag Paradiddle #2 (#37), which extends the pattern by adding a second drag. The relationship between these two drag paradiddles mirrors the single-to-double progression seen throughout the PAS rudiments (e.g., single paradiddle to double paradiddle, single drag tap to double drag tap). In drum corps and rudimental solo literature, the drag paradiddles appear frequently where a composer wants the driving feel of a paradiddle with the added textural weight of drag ornamentation.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>                        >
llR   L   R   L L    |   rrL   R   L   R R
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1     e     &     a   | 2                       2a    | 3     e     &     a   |
Hand:    | (ll)R L     R     L   | L                       (rr)L | R     L     R     R   |
Type:    |  gg A t     t     d1  | d2                       gg A | t     t     d1    d2  |
Accent:  |     >                 |                              > |                       |
```

Grace notes shown in doubled parentheses indicate two same-hand grace notes (drag) falling just before the grid position. The first grace is at -1/16 beat and the second at -1/32 beat from the primary. The first group occupies beats 1 through 2 (5 sixteenth-note positions: 1, 1e, 1&, 1a, 2). The second group occupies beats 2a through 3a (5 sixteenth-note positions: 2a, 3, 3e, 3&, 3a). Drag primaries land on beat 1 and beat 2a respectively.
