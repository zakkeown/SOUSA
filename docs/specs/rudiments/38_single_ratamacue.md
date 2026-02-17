# PAS #38: Single Ratamacue

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 38 |
| **Name** | Single Ratamacue |
| **Category** | drag |
| **Family** | ratamacue |
| **Composed Of** | Drag (#31) + three single strokes ending with an accent; equivalently, a drag followed by a single stroke four (#2) where the first stroke is the drag primary and the last stroke is accented |
| **Related** | #31 Drag (the primitive drag ornament), #39 Double Ratamacue (two drags + singles + accent), #40 Triple Ratamacue (three drags + singles + accent), #32 Single Drag Tap (drag + tap, simpler drag compound), #2 Single Stroke Four (the underlying four-note pattern) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 2 beats |
| **Strokes Per Cycle** | 12 (8 primary strokes + 4 grace notes) |
| **Primary Strokes Per Cycle** | 8 |

The single ratamacue is the foundational pattern of the ratamacue family (PAS #38-40). Its name is onomatopoeic: "ra-ta-ma-cue" maps to the four primary strokes of each group -- "ra" represents the drag grace notes and their primary, "ta" and "ma" are the two middle taps, and "cue" is the closing accent. The pattern alternates between right-lead and left-lead groups, each spanning one beat. The drag primary and subsequent three strokes are evenly spaced on the sixteenth-note grid, with the two grace notes falling just before the first primary stroke of each group. Each group of four primary strokes fills one beat of sixteenth notes: the drag primary on the downbeat, two taps on the "e" and "&", and the accent on the "a". The accent on the final stroke provides a strong resolution that propels the pattern into the next group.

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/16 | (1st grace before 1) | L | g |
| 2 | -1/32 | (2nd grace before 1) | L | g |
| 3 | 0 | 1 | R | t |
| 4 | 1/4 | 1e | L | t |
| 5 | 1/2 | 1& | R | t |
| 6 | 3/4 | 1a | L | A |
| 7 | 1 - 1/16 | (1st grace before 2) | R | g |
| 8 | 1 - 1/32 | (2nd grace before 2) | R | g |
| 9 | 1 | 2 | L | t |
| 10 | 1 + 1/4 | 2e | R | t |
| 11 | 1 + 1/2 | 2& | L | t |
| 12 | 1 + 3/4 | 2a | R | A |

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
            >  |              >
llR  L  R  L   |  rrL  R  L  R
```

Each beat contains one complete ratamacue: a drag (two grace notes + tap primary) followed by two taps and a closing accent. The lead hand alternates every beat.

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | drag grace 1 | #3 | First drag grace note, same hand as #2 |
| 2 | L | grace | drag grace 2 | #3 | Second drag grace note, closer to primary |
| 3 | R | tap | primary | n/a | Drag primary (unaccented); the "ra" |
| 4 | L | tap | none | n/a | First tap after drag; the "ta" |
| 5 | R | tap | none | n/a | Second tap; the "ma" |
| 6 | L | accent | none | n/a | Closing accent; the "cue" |
| 7 | R | grace | drag grace 1 | #9 | First drag grace note, group 2 |
| 8 | R | grace | drag grace 2 | #9 | Second drag grace note, group 2 |
| 9 | L | tap | primary | n/a | Drag primary (unaccented); the "ra" |
| 10 | R | tap | none | n/a | First tap after drag; the "ta" |
| 11 | L | tap | none | n/a | Second tap; the "ma" |
| 12 | R | accent | none | n/a | Closing accent; the "cue" |

### Ornament Timing

**Drag Grace Notes:**
- First grace offset from primary: **-1/16 beat** (the earlier of the two drag grace notes)
- Second grace offset from primary: **-1/32 beat** (the later of the two, closer to the primary)
- Both grace notes played by the **same hand** (opposite from the primary stroke)
- Strokes #1-2 (both L) precede stroke #3 (R primary); strokes #7-8 (both R) precede stroke #9 (L primary)
- The two grace notes form a quick diddle (same-hand double bounce) leading into the drag primary
- Important: the drag primary in the ratamacue is a **tap** (unaccented), not an accent as in the basic drag (#31). The accent is reserved for the final stroke of the group ("cue"). This distinguishes the ratamacue from most other drag-category rudiments where the drag primary is accented.
- The grace notes should produce a soft ruff leading into the first tap, not a distinct two-note event

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent ("cue") | 1.0 | Closing accent, the loudest stroke in each group |
| tap (drag primary + singles) | 0.65 - 0.77 | The drag primary and middle taps are all unaccented |
| grace (drag) | 0.45 - 0.65 | Each of the two drag grace notes |

The ratamacue has a distinctive three-level dynamic hierarchy, but unlike most drag rudiments, the drag primary is not accented. The hierarchy is: grace notes (softest) leading into the drag primary tap, two more taps at the same level, and finally the closing accent (loudest). This creates a characteristic "soft-medium-medium-medium-LOUD" contour within each group that gives the ratamacue its propulsive quality.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
-  -  -  >  |  -  -  -  >
llR  L  R  L  |  rrL  R  L  R
```

One accent per beat, always on the final sixteenth note ("a" position). The three preceding strokes (including the drag primary) are all unaccented.

### Dynamic Contour

The single ratamacue produces a repeating crescendo-to-accent shape: each beat begins with the soft drag grace notes, continues through three evenly voiced taps, and resolves on a strong accent. This "build-and-release" contour is the defining musical characteristic of the ratamacue family. The accent on the "a" of each beat creates an off-beat emphasis that gives the pattern a syncopated, forward-driving feel. When repeated, the accents land on weak subdivisions (the "a" of each beat), producing a rhythmic tension against the underlying pulse that resolves when the pattern ends. This off-beat accent placement distinguishes the ratamacue from most other drag rudiments, where accents typically fall on strong beats.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace (drag 1) | tap (< 2") | fingers |
| 2 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 3 | R | tap (drag primary) | low (2-6") | wrist |
| 4 | L | tap | low (2-6") | wrist |
| 5 | R | tap | low (2-6") | wrist |
| 6 | L | accent | full (12"+) | wrist + arm |
| 7 | R | grace (drag 1) | tap (< 2") | fingers |
| 8 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 9 | L | tap (drag primary) | low (2-6") | wrist |
| 10 | R | tap | low (2-6") | wrist |
| 11 | L | tap | low (2-6") | wrist |
| 12 | R | accent | full (12"+) | wrist + arm |

The physical flow of the single ratamacue involves two contrasting motions within each beat. The left hand in group 1 follows this path: two grace notes at tap height (finger control), then a tap at low height (#4), then a full accent (#6). This creates a gradual upward height progression (tap -> low -> full) within the same hand, which naturally supports the crescendo into the accent. The right hand plays the drag primary tap (#3), a tap (#5), then immediately switches to drag grace notes (#7-8) at the start of the next group. The accent-to-grace transition across the group boundary (stroke #6 accent to stroke #7 grace on the opposite hand) is manageable because different hands are involved.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accents. Grace notes individually articulated as two distinct soft strokes (open drag). All four primary strokes per beat are clearly separated with full low-to-full height progression visible. The onomatopoeic syllables "ra-ta-ma-cue" can be spoken in rhythm at this tempo. Each grace note uses active finger strokes. |
| Moderate (100-140 BPM) | Wrist-driven strokes for taps. Drag grace notes close up, with the second relying on rebound. The ruff becomes a single ornamental event rather than two distinct notes. The three taps and accent flow as a connected sixteenth-note group. Standard grace offsets at -1/16 and -1/32. The accent remains at full height to maintain dynamic contrast. |
| Fast (140-160 BPM) | Finger control for grace notes and taps. Accent drops to half height class. Drag grace notes compress toward -1/32 and -1/64 offsets, approaching a crushed ruff. The four primary strokes per beat merge into a rapid sixteenth-note burst with accent emphasis. The ratamacue begins to sound more like a continuous flow of sixteenths with off-beat accents than four distinct syllables. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Triplet interpretation**: Some traditions play the ratamacue in a triplet feel rather than strict sixteenth notes, especially in older rudimental styles. The four primary strokes are grouped as a triplet + downbeat rather than four even sixteenths.
- **No-ornament preparation**: Practice as four alternating single strokes with the accent on the last (R L R >L | L R L >R) to isolate the accent placement before adding drag grace notes.
- **Ratamacue as fill**: In drum set contexts, the ratamacue is frequently used as a fill pattern, with the accent landing on a crash cymbal or different drum.
- **Continuous ratamacues**: Linking multiple ratamacues end-to-end, where the accent of one group flows directly into the drag of the next (the accent hand plays the drag grace notes for the following group).

### Prerequisites

- #31 Drag -- the basic drag ornament must be consistent on both hands
- #1 Single Stroke Roll -- basic hand alternation at sixteenth-note speed
- #2 Single Stroke Four -- the four-note single stroke pattern that forms the primary strokes
- #6 Double Stroke Open Roll -- controlled double bounce technique for the grace notes
- Ability to place accents on any position within a sixteenth-note group

### Builds Toward

- #39 Double Ratamacue (adds a second drag at the beginning)
- #40 Triple Ratamacue (adds two more drags at the beginning)
- Application in drum corps, concert, and marching percussion literature
- Drum set fills and solo vocabulary

### Teaching Notes

The single ratamacue is the first of the three ratamacue rudiments and the canonical pattern of the ratamacue family. It combines the drag ornament with a four-note single stroke group, placing the accent on the final stroke rather than on the drag primary. This accent placement is unusual among drag rudiments and gives the ratamacue its distinctive character.

**Common mistakes:**
- Accenting the drag primary: In most drag rudiments (#31-37), the drag primary is accented. In the ratamacue, the drag primary is a tap. Students often carry the habit of accenting the drag primary, which destroys the ratamacue's characteristic "build to the cue" shape.
- Grace notes too loud: The ruff overpowers the tap primary, making the pattern sound cluttered at the beginning of each group.
- Missing the accent: The closing accent ("cue") is the most important stroke. It must be clearly louder than the preceding taps to create the ratamacue's signature dynamic contour.
- Uneven sixteenth-note spacing: The four primary strokes should be evenly spaced within the beat. Students often rush the taps leading into the accent.
- Inconsistent drag quality between hands: Left-lead drags (rrL) should sound identical to right-lead drags (llR).
- Treating the drag as a flam: Both grace notes must be present. A collapsed drag (single grace) converts the ratamacue into a flamacue-like pattern (#23).

**Practice strategies:**
- Speak the syllables "ra-ta-ma-CUE" in rhythm before playing, emphasizing the final syllable.
- Start without grace notes: practice R L R >L | L R L >R to establish the accent pattern.
- Add grace notes: ll(R L R >L) | rr(L R L >R), ensuring the drag does not disrupt the sixteenth-note flow.
- Practice each hand's lead separately before alternating.
- Use a metronome with the click on the beat; notice that the accent falls on the "a" (the last sixteenth of each beat), creating an off-beat emphasis.
- Compare the sound of right-lead and left-lead groups for consistency.

**What to listen for:**
- The "ra-ta-ma-CUE" shape: three soft-to-medium strokes building to a strong accent.
- Clear dynamic separation between the taps and the closing accent.
- Drag grace notes that color the first tap without overpowering it.
- Even sixteenth-note spacing across all four primary strokes.
- Identical quality on both right-lead and left-lead groups.
- A propulsive, forward-driving feel created by the off-beat accent.

### Historical Context

The ratamacue is one of the oldest named drum rudiments, included in the original NARD (National Association of Rudimental Drummers) standard 26 rudiments when NARD was founded in 1933. The name "ratamacue" is onomatopoeic, derived from the sound the pattern makes when played: "ra-ta-ma-cue" approximates the quick ruff, two taps, and accented final stroke. This phonetic naming convention is characteristic of the oral tradition through which rudimental drumming was transmitted in early American military field drumming. The ratamacue appears in Charles Stewart Ashworth's *A New, Useful and Complete System of Drum Beating* (1812) and other early American drumming manuals, establishing it as one of the foundational patterns of the rudimental tradition. The Percussive Arts Society retained the single ratamacue as PAS #38 and added the double (#39) and triple (#40) variants when the rudiment list was expanded from 26 to 40 in 1984, though all three forms were practiced long before formal codification. In drum corps and marching percussion, the ratamacue remains a fundamental vocabulary element, often used for fills, transitions, and solo passages where its characteristic crescendo-to-accent shape provides natural phrasing and forward motion.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
               >  |                 >
llR   L   R   L   |  rrL   R   L   R
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1        e     &     a   | 2        e     &     a   |
Hand:    | (ll)R    L     R     L   | (rr)L    R     L     R   |
Type:    |  gg  t   t     t     A   |  gg  t   t     t     A   |
Accent:  |                     >    |                     >    |
```

Grace notes shown in doubled parentheses indicate two same-hand grace notes falling just before the grid position. The first grace is at -1/16 beat and the second at -1/32 beat from the primary. The drag primaries land on beats 1 and 2 (unaccented), and the accents fall on the "a" positions (1a and 2a). The four primary strokes within each beat fill the complete sixteenth-note grid: 1, 1e, 1&, 1a for the first group and 2, 2e, 2&, 2a for the second.
