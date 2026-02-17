# PAS #21: Flam Accent

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 21 |
| **Name** | Flam Accent |
| **Category** | flam |
| **Family** | basic flam |
| **Composed Of** | Flam (#20) + single stroke taps in triplet subdivision |
| **Related** | #20 Flam (the primitive flam ornament), #22 Flam Tap (flam + tap in eighth notes), #28 Swiss Army Triplet (another triplet-based flam rudiment, but with diddles) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | triplet |
| **Cycle Length** | 2 beats |
| **Strokes Per Cycle** | 8 (6 primary strokes + 2 grace notes) |
| **Primary Strokes Per Cycle** | 6 |

The flam accent is one of the few standard rudiments written in **triplet** subdivision (3 notes per beat rather than 4). Each beat contains one triplet group of 3 primary strokes, with a flam on the first note of each group. The triplet feel gives this rudiment its characteristic flowing, rolling quality that distinguishes it from the more angular sixteenth-note-based flam rudiments.

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/32 | (grace before 1) | L | g |
| 2 | 0 | 1 | R | A |
| 3 | 1/3 | 1& | L | t |
| 4 | 2/3 | 1a | R | t |
| 5 | 1 - 1/32 | (grace before 2) | R | g |
| 6 | 1 | 2 | L | A |
| 7 | 1 + 1/3 | 2& | R | t |
| 8 | 1 + 2/3 | 2a | L | t |

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>              >
lR   L   R  |  rL   R   L
```

Each group of three is one beat of triplets. The flam falls on beat 1 of each triplet group, followed by two single-stroke taps that complete the triplet.

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | grace | #2 | Grace note preceding right-hand flam |
| 2 | R | accent | primary | n/a | Accented primary -- beat 1 of first triplet |
| 3 | L | tap | none | n/a | Unaccented tap -- beat 2 of first triplet |
| 4 | R | tap | none | n/a | Unaccented tap -- beat 3 of first triplet |
| 5 | R | grace | grace | #6 | Grace note preceding left-hand flam |
| 6 | L | accent | primary | n/a | Accented primary -- beat 1 of second triplet |
| 7 | R | tap | none | n/a | Unaccented tap -- beat 2 of second triplet |
| 8 | L | tap | none | n/a | Unaccented tap -- beat 3 of second triplet |

### Ornament Timing

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- Stroke #1 (L grace) precedes stroke #2 (R primary); stroke #5 (R grace) precedes stroke #6 (L primary)
- The grace note falls before the triplet downbeat, not on a triplet subdivision position
- Because the triplet grid is wider than sixteenth grids (1/3 beat between positions vs 1/4), the grace note offset of -1/32 is well clear of the preceding triplet note

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (primary) | 1.0 | Flam primary strokes on beat 1 of each triplet |
| tap | 0.65 - 0.77 | The two taps following each flam |
| grace (flam) | 0.50 - 0.70 | Soft grace note preceding each primary |

The flam accent has three distinct dynamic levels: the loud primary (accent), the moderate taps, and the soft grace notes. This three-level dynamic hierarchy is central to the rudiment's musical character.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>     -     -  |  >     -     -
lR    L     R  |  rL    R     L
```

One accent per beat, always on the first triplet note (which is the flam). The two taps that follow are unaccented.

### Dynamic Contour

The flam accent produces a repeating "strong-weak-weak" pattern within each beat, mirroring the natural stress pattern of triple meter. The accented flam provides a clear rhythmic anchor, while the two following taps create a sense of motion and flow toward the next accent. This accent-tap-tap contour gives the flam accent its characteristic lilting, waltz-like feel. The alternating lead hand (R on beat 1, L on beat 2) ensures the pattern feels balanced and naturally cyclic. The overall contour across the 2-beat cycle is symmetrical: each beat has the same strong-weak-weak shape, but with opposite hand leading.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace | tap (< 2") | fingers |
| 2 | R | accent | full (12"+) | wrist + arm |
| 3 | L | tap | low (2-6") | wrist |
| 4 | R | tap | low (2-6") | wrist |
| 5 | R | grace | tap (< 2") | fingers |
| 6 | L | accent | full (12"+) | wrist + arm |
| 7 | R | tap | low (2-6") | wrist |
| 8 | L | tap | low (2-6") | wrist |

The physical challenge of the flam accent lies in the transition sequence within each hand. Consider the right hand's path through one full cycle: it plays a full-height primary (stroke #2), immediately drops to low-height tap (stroke #4), then must prepare as a grace note (stroke #5) before the left hand's primary. This full-to-low-to-tap height sequence requires smooth, controlled stick management.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-90 BPM) | Each triplet note is distinctly articulated. Full wrist + arm motion for accented flams, deliberate wrist strokes for taps. Grace notes clearly audible with wide spacing from primary. The triplet subdivision is felt as three distinct events per beat. Practice focus: accent-tap dynamic contrast and flam quality. |
| Moderate (90-130 BPM) | Standard performance tempo. Wrist-driven strokes predominate. The accent stroke uses a controlled downstroke (accent, then stop the stick low for the following tap). The two taps flow naturally with wrist motion. Grace note spacing tightens to standard -1/32 offset. The rolling triplet feel becomes apparent. |
| Fast (130-160 BPM) | Finger control becomes primary for taps. The accent stroke transitions to a Moeller whip to maintain dynamic contrast at speed. Stick heights compress: accent drops to half class, taps to tap class. Grace note spacing approaches -1/64. The rudiment begins to feel like a continuous flowing motion rather than discrete triplet groups. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Flam accent in 6/8**: The same pattern fits naturally into compound time, with two flam accent groups per bar of 6/8
- **Inverted flam accent**: The flam is placed on the third triplet note instead of the first, creating a pickup-style feel
- **Accented taps variation**: Adding accents to the second or third triplet notes for dynamic independence training
- **Flam accent with diddles**: Replacing taps with diddles for a more complex variation (related to flam paradiddle-diddle)
- **No-flam preparation**: Practice the accent-tap-tap pattern as plain triplets (without grace notes) to isolate the accent pattern from the ornament

### Prerequisites

- #20 Flam -- the basic flam ornament must be solid before adding the triplet context
- #1 Single Stroke Roll -- alternating single strokes form the tap portions
- Comfort with triplet subdivision -- the player must be able to feel and count in groups of three

### Builds Toward

- #28 Swiss Army Triplet (triplet-based flam with diddle instead of two taps)
- #23 Flamacue (more complex accent pattern with flams)
- #30 Flam Drag (flam combined with drag ornament)
- Application in jazz comping, waltz-time fills, and compound-meter passages

### Teaching Notes

The flam accent is one of the most musical of the 40 rudiments, and its triplet feel makes it immediately applicable to jazz, 6/8 time, and Afro-Cuban styles. It is often the first triplet-based rudiment students encounter.

**Common mistakes:**
- Playing in sixteenth notes instead of triplets -- the flam accent is NOT in 4/4 sixteenth-note subdivision; it is in triplet subdivision (3 evenly spaced notes per beat)
- Even dynamics across all three triplet notes -- the accent on the flam must be noticeably louder than the two taps
- Rushing the two taps after the accent -- the taps should be evenly spaced within the triplet, not crammed toward the next beat
- Flat flams on the accent -- the grace note must be clearly softer and earlier than the primary
- Losing the alternation -- the lead hand must switch every beat (R-lead, then L-lead)
- Uneven triplet spacing -- all three notes in each triplet group should be equidistant in time

**Practice strategies:**
- Begin by playing accented triplets (no flams): >R L R | >L R L, focusing on the accent pattern
- Add the grace notes only after the accent-tap-tap pattern is comfortable
- Practice with a metronome set to the triplet subdivision to lock in spacing
- Use the counting "1-&-a, 2-&-a" for each beat to internalize the triplet grid
- Practice each hand's lead separately before alternating

**What to listen for:**
- Clear three-level dynamic hierarchy: grace (soft) < tap (medium) < accent (loud)
- Even triplet spacing -- no rushing or dragging within the group
- Identical sound quality regardless of which hand leads
- The flam on beat 1 of each group should sound unified (grace + primary as one event)
- A smooth, flowing feel that does not sound choppy or angular

### Historical Context

The flam accent was included in the original NARD 26 rudiments and has deep roots in American and European field drumming traditions. Its triplet feel connects it to the Swiss and Basel drumming traditions, where triple-meter patterns are fundamental. The rudiment was retained as PAS #21 in the 1984 expansion to 40 rudiments. The flam accent is particularly important in jazz drumming, where the triplet feel aligns naturally with swing rhythm. Legendary jazz drummer Joe Morello was known for his impeccable flam accents, and the rudiment appears frequently in George Lawrence Stone's *Stick Control* (1935) and Charles Wilcoxon's rudimental method books. In the drum corps tradition, the flam accent is a staple of snare drum ensemble writing, often used in compound-meter passages and transitions.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>              >
lR   L   R  |  rL   R   L
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. Triplet subdivision: 3 notes per beat. -->

```
Beat:    | 1        &        a     | 2        &        a     |
Hand:    | (l)R     L        R     | (r)L     R        L     |
Type:    |  g  A    t        t     |  g  A    t        t     |
Accent:  |     >                   |     >                   |
```

Grace notes shown in parentheses indicate they fall just before the grid position (offset -1/32 beat). The triplet grid divides each beat into three equal parts at positions 0, 1/3, and 2/3 of each beat.
