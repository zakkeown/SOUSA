# PAS #33: Double Drag Tap

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 33 |
| **Name** | Double Drag Tap |
| **Category** | drag |
| **Family** | basic drag |
| **Composed Of** | Two Drags (#31) + tap; each group has two consecutive drags followed by a single tap |
| **Related** | #31 Drag (the primitive drag ornament), #32 Single Drag Tap (one drag + tap), #34 Lesson 25 (another multi-drag pattern), #33 is to #32 as #17 Double Paradiddle is to #16 Single Paradiddle (extended version of the same idea) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 3 beats |
| **Strokes Per Cycle** | 14 (6 primary strokes + 8 grace notes) |
| **Primary Strokes Per Cycle** | 6 |

The double drag tap extends the single drag tap by placing two consecutive drags before the tap. Each group of three primary strokes spans 1.5 beats: two dragged accents on successive eighth-note positions, followed by a tap. The pattern requires 3 beats to complete one full left-right alternation (two groups of three primary strokes each). The six primary strokes fall on the eighth-note grid, with the drag grace notes filling the space just before each accented primary.

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/16 | (1st grace before 1) | L | g |
| 2 | -1/32 | (2nd grace before 1) | L | g |
| 3 | 0 | 1 | R | A |
| 4 | 1/2 - 1/16 | (1st grace before 1&) | R | g |
| 5 | 1/2 - 1/32 | (2nd grace before 1&) | R | g |
| 6 | 1/2 | 1& | L | A |
| 7 | 1 | 2 | R | t |
| 8 | 1 + 1/2 - 1/16 | (1st grace before 2&) | R | g |
| 9 | 1 + 1/2 - 1/32 | (2nd grace before 2&) | R | g |
| 10 | 1 + 1/2 | 2& | L | A |
| 11 | 2 - 1/16 | (1st grace before 3) | L | g |
| 12 | 2 - 1/32 | (2nd grace before 3) | L | g |
| 13 | 2 | 3 | R | A |
| 14 | 2 + 1/2 | 3& | L | t |

Note: The first group (strokes #1-7) spans beats 1 to 2: drag on 1, drag on 1&, tap on 2. The second group (strokes #8-14) spans beats 2& to 3&: drag on 2&, drag on 3, tap on 3&. The lead hand alternates between groups. Within each group, the two drags alternate hands (the grace notes switch sides for each drag), and the tap is on the same hand as the second drag's primary.

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>        >        -     |  >        >        -
llR      rrL      R     |  rrL      llR      L
```

Each group consists of two drags (alternating lead hand) followed by a tap. The first group begins with a right-hand drag primary, the second with a left-hand drag primary.

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | drag grace 1 | #3 | First grace, first drag of group 1 |
| 2 | L | grace | drag grace 2 | #3 | Second grace, first drag of group 1 |
| 3 | R | accent | primary | n/a | First drag primary, group 1 |
| 4 | R | grace | drag grace 1 | #6 | First grace, second drag of group 1 |
| 5 | R | grace | drag grace 2 | #6 | Second grace, second drag of group 1 |
| 6 | L | accent | primary | n/a | Second drag primary, group 1 |
| 7 | R | tap | none | n/a | Tap closing group 1 |
| 8 | R | grace | drag grace 1 | #10 | First grace, first drag of group 2 |
| 9 | R | grace | drag grace 2 | #10 | Second grace, first drag of group 2 |
| 10 | L | accent | primary | n/a | First drag primary, group 2 |
| 11 | L | grace | drag grace 1 | #13 | First grace, second drag of group 2 |
| 12 | L | grace | drag grace 2 | #13 | Second grace, second drag of group 2 |
| 13 | R | accent | primary | n/a | Second drag primary, group 2 |
| 14 | L | tap | none | n/a | Tap closing group 2 |

### Ornament Timing

**Drag Grace Notes:**
- First grace offset from primary: **-1/16 beat** (the earlier of the two drag grace notes)
- Second grace offset from primary: **-1/32 beat** (the later of the two, closer to the primary)
- Both grace notes played by the **same hand** (opposite from the primary stroke)
- Each drag follows the standard ornament timing from #31 Drag
- There are four drags per cycle, each with its own pair of grace notes:
  - Strokes #1-2 (L) precede #3 (R); strokes #4-5 (R) precede #6 (L)
  - Strokes #8-9 (R) precede #10 (L); strokes #11-12 (L) precede #13 (R)
- Critical transition: Between the first and second drag within each group, the hand that just played a drag primary (#3 or #10) must immediately produce two grace notes (#4-5 or #11-12) for the next drag. This drag-primary-to-drag-grace transition on the same hand is the defining technical challenge of the double drag tap.

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (drag primary) | 1.0 | Accented primary strokes for all four drags |
| tap | 0.65 - 0.77 | Unaccented stroke closing each group |
| grace (drag) | 0.45 - 0.65 | Each of the eight drag grace notes |

The double drag tap has the same three-level dynamic hierarchy as the single drag tap: grace (softest), tap (medium), accent (loudest). However, with two consecutive accented drags before each tap, the pattern has a stronger emphasis on the accented strokes, producing a more driving and intense dynamic shape.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>        >        -     |  >        >        -
llR      rrL      R     |  rrL      llR      L
```

Two accents per group (both on drag primaries), with the tap unaccented. The pattern creates a "strong-strong-weak" grouping within each 1.5-beat phrase.

### Dynamic Contour

The double drag tap produces a distinctive "ruff-accent-ruff-accent-valley" pattern within each 1.5-beat group. The two consecutive drags create a pair of buzz-accented strokes that build momentum, and the tap provides a dynamic release. Compared to the single drag tap (#32), which alternates accent-tap evenly, the double drag tap's two consecutive accents create a more intense, forward-driving feel. The "strong-strong-weak" grouping produces a natural sense of phrasing that leads the ear from one group to the next. The brief tap at the end of each group functions as a pickup into the next group's first drag.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace (drag 1) | tap (< 2") | fingers |
| 2 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 3 | R | accent | full (12"+) | wrist + arm |
| 4 | R | grace (drag 1) | tap (< 2") | fingers |
| 5 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 6 | L | accent | full (12"+) | wrist + arm |
| 7 | R | tap | low (2-6") | wrist |
| 8 | R | grace (drag 1) | tap (< 2") | fingers |
| 9 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 10 | L | accent | full (12"+) | wrist + arm |
| 11 | L | grace (drag 1) | tap (< 2") | fingers |
| 12 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 13 | R | accent | full (12"+) | wrist + arm |
| 14 | L | tap | low (2-6") | wrist |

The defining physical challenge of the double drag tap is the drag-primary-to-drag-grace transition. Consider the right hand in the first group: it plays a full-height accent (#3), then must immediately drop to tap height to produce two finger-controlled grace notes (#4-5) for the opposite hand's drag. This full-to-tap height transition on the same hand requires a rapid downstroke-to-bounce motion: the stick comes from full height for the accent and must be caught near the head to initiate the double bounce for the next drag's grace notes. This is more demanding than the single drag tap, where the same hand plays a drag primary and then rests while the other hand taps.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accented drag primaries. Grace notes individually articulated (open drags). Clear separation between each drag and the tap. The accent-to-grace transition within each hand is manageable due to the wider spacing. Each group's three events (drag-drag-tap) are clearly distinguishable. |
| Moderate (100-140 BPM) | Wrist-driven accents. Drag grace notes close up to standard ruff sound. The accent-to-grace transition tightens, requiring efficient downstroke technique: the accent must be caught low to prepare for the subsequent grace notes. The two drags per group begin to flow together, with the tap providing a rhythmic break. |
| Fast (140-160 BPM) | Finger control for grace notes and taps. Accents drop to half height class. The accent-to-grace transition becomes the critical bottleneck -- the hand must convert the accent's impact energy into a controlled double bounce nearly instantaneously. Drag grace notes compress toward -1/32 and -1/64 offsets. The pattern feels like a continuous stream of ruff-accents with brief tap punctuation. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Triple drag tap**: Three drags before the tap (not a standard PAS rudiment, but a logical extension used in exercises)
- **Unaccented double drag tap**: Playing all strokes at a uniform dynamic for evenness practice
- **No-ornament preparation**: Practice as accent-accent-tap (>R >L R | >L >R L) to isolate the rhythm and phrasing before adding grace notes
- **Double drag tap as written exercise**: Often practiced with different sticking patterns to develop flexibility

### Prerequisites

- #31 Drag -- the basic drag ornament must be solid on both hands
- #32 Single Drag Tap -- the simpler drag + tap pattern should be comfortable before adding a second drag
- #6 Double Stroke Open Roll -- controlled double bounce technique
- Downstroke control -- the ability to play a full accent and immediately catch the stick near the head for the following grace notes

### Builds Toward

- #34 Lesson 25 (complex multi-drag pattern)
- #39 Double Ratamacue (two drags in a ratamacue context)
- Application in drum corps and marching percussion passages requiring extended drag phrases
- Development of the accent-to-grace transition skill needed in all compound drag rudiments

### Teaching Notes

The double drag tap extends the single drag tap by adding a second drag before each tap, introducing the challenging accent-to-grace transition within the same hand. It is an important stepping stone between the simpler basic drag patterns and the more complex compound drag rudiments.

**Common mistakes:**
- Collapsing the second drag: The second drag in each group often loses one or both grace notes at speed, reducing it to a flam or plain accent
- Uneven drags: The first and second drag within each group should be identical in quality. Students often execute the first drag well but rush or compress the second
- Rushing the tap: The tap at the end of each group is often cut short as the player hurries to the next group's first drag
- Losing the accent-to-grace transition: After the first drag's accent (#3 or #10), the same hand must immediately produce two grace notes. This transition is often too slow, causing the second drag to arrive late
- Inconsistent group spacing: Each 1.5-beat group should be evenly spaced. Students often compress the second half (drag-tap) relative to the first drag
- Different drag quality between the first and second drag: Both drags within a group should have identical ruff character

**Practice strategies:**
- Start with the accent-accent-tap pattern without ornaments: >R >L R | >L >R L
- Practice the accent-to-grace transition in isolation: play a full accent, then immediately produce two grace notes on the same hand
- Add drags one at a time: first only the first drag (llR L R | rrL R L), then add the second (llR rrL R | rrL llR L)
- Use a metronome and count: "1 & 2" for the first group, "& 3 &" for the second (primary strokes on the eighth-note grid)
- Practice slowly to ensure both drags in each group are rhythmically even and dynamically matched
- Compare the two groups (right-lead vs. left-lead) for consistency
- Record and listen for equal drag quality across all four drags per cycle

**What to listen for:**
- All four drags per cycle should have identical ruff quality
- Clear three-level dynamics: grace (soft) < tap (medium) < accent (loud)
- Even eighth-note spacing of the three primary strokes within each group
- The tap should be clearly softer than the two preceding accented drags
- Both groups (right-lead and left-lead) should sound identical
- The "strong-strong-weak" phrasing should create a natural musical grouping

### Historical Context

The double drag tap was included in the original NARD 26 rudiments and has been a standard in rudimental drumming since the earliest American field drumming traditions. It was retained as PAS #33 in the 1984 expansion to 40 rudiments. The relationship between the single drag tap (#32) and the double drag tap (#33) mirrors the progressive structure seen throughout the PAS rudiments: begin with the simplest form and extend it. This "single then double" pattern is a fundamental pedagogical approach in rudimental drumming (cf. single paradiddle #16 to double paradiddle #17, single ratamacue #38 to double ratamacue #39). The double drag tap appears in military drumming manuals, drum corps snare drum literature, and contest solos where the extended drag phrase provides rhythmic drive and textural interest. Its two-drag grouping creates a recognizable rhythmic motif that has been a fixture of American rudimental style since at least the early 19th century.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>          >          -     |  >          >          -
llR        rrL        R     |  rrL        llR        L
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1           &         | 2           &         | 3           &         |
Hand:    | (ll)R       (rr)L     | R           (rr)L     | (ll)R       L         |
Type:    |  gg  A       gg  A    | t            gg  A    |  gg  A      t         |
Accent:  |      >           >    |                  >    |      >                |
```

Grace notes shown in doubled parentheses indicate two same-hand grace notes falling just before the grid position. The first grace is at -1/16 beat and the second at -1/32 beat from the primary. The first group occupies beats 1 through 2 (drag on 1, drag on 1&, tap on 2). The second group occupies beats 2& through 3& (drag on 2&, drag on 3, tap on 3&).
