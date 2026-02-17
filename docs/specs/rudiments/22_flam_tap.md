# PAS #22: Flam Tap

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 22 |
| **Name** | Flam Tap |
| **Category** | flam |
| **Family** | basic flam |
| **Composed Of** | Flam (#20) + tap, built on a double stroke roll (#6) framework with the first of each double flammed |
| **Related** | #20 Flam (the primitive flam ornament), #21 Flam Accent (flam + taps in triplets), #6 Double Stroke Open Roll (same RR LL hand pattern without grace notes), #29 Inverted Flam Tap (flam on the second stroke of each double instead of the first) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | eighth |
| **Cycle Length** | 1 beat |
| **Strokes Per Cycle** | 6 (4 primary strokes + 2 grace notes) |
| **Primary Strokes Per Cycle** | 4 |

The flam tap is structured as a double stroke roll (RR LL) with a flam added to the first stroke of each double. Each hand plays three consecutive events: grace note (opposite hand), accented primary, then tap. The pattern alternates between hands every half beat.

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/32 | (grace before 1) | L | g |
| 2 | 0 | 1 | R | A |
| 3 | 1/4 | 1e | R | t |
| 4 | 1/2 - 1/32 | (grace before 1&) | R | g |
| 5 | 1/2 | 1& | L | A |
| 6 | 3/4 | 1a | L | t |

Note: Although the base subdivision is eighth notes (flams land on eighth-note positions 1 and &), the tap strokes subdivide each half beat, placing them on the sixteenth-note "e" and "a" positions. The flam tap thus occupies a sixteenth-note grid in practice, but its primary rhythmic anchors are the eighth-note flam placements.

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>        >
lR  R  | rL  L
```

Each group consists of a flam (grace + accented primary) followed by a tap on the same hand as the primary. The lead hand alternates every half beat.

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | grace | #2 | Grace note preceding right-hand flam |
| 2 | R | accent | primary | n/a | Right-hand flam primary stroke (accented) |
| 3 | R | tap | none | n/a | Right-hand tap following the flam |
| 4 | R | grace | grace | #5 | Grace note preceding left-hand flam |
| 5 | L | accent | primary | n/a | Left-hand flam primary stroke (accented) |
| 6 | L | tap | none | n/a | Left-hand tap following the flam |

### Ornament Timing

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- Stroke #1 (L grace) precedes stroke #2 (R primary); stroke #4 (R grace) precedes stroke #5 (L primary)
- The grace note and primary are heard as a single unified note
- Important: the grace note for the second flam (stroke #4) is played by the right hand, which just played the tap (stroke #3). This rapid transition from tap to grace on the same hand is a key technical challenge.

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (primary) | 1.0 | Flam primary strokes on the eighth-note positions |
| tap | 0.65 - 0.77 | The tap following each flam, on the same hand |
| grace (flam) | 0.50 - 0.70 | Soft grace note preceding each primary |

The flam tap features the same three-level dynamic hierarchy as the flam accent: grace (soft), tap (medium), accent (loud). However, the dynamic contour differs because the tap follows the accent on the **same hand**, creating an accent-to-tap transition within a single hand rather than between hands.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>     -     >     -
lR    R     rL    L
```

Accents fall on every other primary stroke (the flammed notes), with taps between.

### Dynamic Contour

The flam tap produces a repeating "strong-weak" pattern at the eighth-note level: accented flam, then unaccented tap, alternating between hands. This creates a steady pulse of accented strokes on the eighth-note grid, with the taps filling in the sixteenth-note subdivisions between. The overall feel is more driving and rhythmic than the flam accent's flowing triplet character. The accent-tap pair on each hand mimics the feel of a double stroke roll, but with the added weight of the flam on the first stroke.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace | tap (< 2") | fingers |
| 2 | R | accent | full (12"+) | wrist + arm |
| 3 | R | tap | low (2-6") | wrist |
| 4 | R | grace | tap (< 2") | fingers |
| 5 | L | accent | full (12"+) | wrist + arm |
| 6 | L | tap | low (2-6") | wrist |

The primary physical challenge of the flam tap is the rapid height transition within each hand across the cycle. Consider the right hand's path: it plays a full-height accented primary (#2), immediately drops to a low-height tap (#3), then must transition to a tap-height grace note (#4). This full-to-low-to-tap height sequence happens within a single beat and requires a controlled "downstroke-tap-upstroke" stick motion. The accent is a downstroke (start high, stop low), the tap stays low, and the grace requires the stick to be near the head for the next hand's primary.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accented flams. Clear separation between the flam and the following tap. The double-stroke nature (RR, LL) is clearly audible. Grace notes have wide spacing from primary (-1/16 range). Focus on matching left-hand and right-hand flam quality. |
| Moderate (100-140 BPM) | Wrist-driven strokes. The accent uses a controlled downstroke technique: the stick comes from full height for the accent and is caught low for the tap. Grace note spacing at standard -1/32. The rudiment begins to flow as a continuous alternating pattern. The transition from tap to grace (same hand) becomes the critical control point. |
| Fast (140-180 BPM) | Finger control for taps and grace notes. The accent drops to half height class. The double-stroke feel increases as the pattern approaches a flammed double stroke roll. Grace note spacing tightens to -1/64. Moeller technique (whip motion) may be employed for the accent strokes to maintain dynamic contrast. The stick path becomes a continuous, circular motion. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Inverted flam tap (#29)**: The flam is placed on the second stroke of each double instead of the first (R lR | L rL), creating a distinctly different accent pattern and hand coordination challenge
- **Flam tap with buzz**: Replace the tap with a buzz stroke for a more sustained sound (less common, but used in some corps-style literature)
- **Double flam tap**: Two flam taps followed by a different rudiment or pattern break
- **No-flam preparation**: Practice as a double stroke roll with accents on the first of each double (>R R >L L) to isolate the accent-tap hand motion before adding grace notes
- **Swiss-style flam tap**: Played with wider, more open flam spacing in the Basel drumming tradition

### Prerequisites

- #20 Flam -- the basic flam ornament must be consistent on both hands
- #6 Double Stroke Open Roll -- the RR LL alternation pattern is the structural foundation
- #1 Single Stroke Roll -- basic hand alternation
- Downstroke control -- the ability to play an accent and stop the stick low for the following tap

### Builds Toward

- #29 Inverted Flam Tap (inverted version with flam on the second double)
- #24 Flam Paradiddle (flam combined with paradiddle sticking)
- #27 Pataflafla (complex compound flam pattern)
- #25 Single Flammed Mill (continuous same-hand flam + tap pattern)
- Application in rudimental solos, drum corps snare parts, and drumset fills

### Teaching Notes

The flam tap is one of the most practical and widely used flam rudiments. Its connection to the double stroke roll makes it an excellent vehicle for developing stick control, downstroke technique, and hand independence.

**Common mistakes:**
- Flat flams: Both sticks striking simultaneously, especially at faster tempos where the grace-to-primary spacing collapses
- Accent and tap at the same volume: The dynamic contrast between the flam (accent) and the following tap must be clear
- Uneven doubles: The spacing between the accent and the tap should be even (each occupying a sixteenth-note position), not rushed or lagging
- Grace note on the wrong hand: The grace is always played by the opposite hand from the primary -- left grace before right primary, right grace before left primary
- "Galloping" rhythm: The accent-tap pairs should be evenly spaced, not lopsided
- Losing the tap-to-grace transition: The hand that just played a tap must immediately prepare as a grace note for the other hand's flam. This is the hardest coordination point and often breaks down at speed.

**Practice strategies:**
- Start with accented doubles (no flams): >R R >L L, focusing on the downstroke-tap stick motion
- Add grace notes only after the accented double pattern is even and controlled
- Practice slowly with a metronome on the eighth-note pulse (flams on the click)
- Isolate the "tap to grace" transition: play the tap, pause, then execute the grace + opposite primary
- Practice at a wide range of tempos to develop both the open (slow) and closed (fast) versions
- Compare the sound of your right-lead flam (lR) with your left-lead flam (rL) and work to match them

**What to listen for:**
- Clear dynamic separation: grace (soft) < tap (medium) < accent (loud)
- Even spacing between accent and tap within each double
- Identical flam quality on both hands (left-lead and right-lead flams should sound the same)
- Smooth alternation -- the transition from one hand's tap to the other hand's grace should be seamless
- The overall pattern should have a steady, driving eighth-note pulse from the accented flams

### Historical Context

The flam tap was included in the original NARD 26 rudiments and is one of the most commonly encountered rudiments in both rudimental and orchestral percussion. Its direct relationship to the double stroke roll makes it a natural extension of fundamental technique. The rudiment was retained as PAS #22 in the 1984 expansion to 40 rudiments. The flam tap is ubiquitous in drum corps snare drum literature, where it serves as a building block for more complex passages. Its inverted form (#29, Inverted Flam Tap) was one of the 14 new rudiments added in the 1984 PAS expansion, demonstrating the flam tap's importance as a pattern worthy of inversion and further study. In John S. Pratt's *14 Modern Contest Solos* (1959), the flam tap features prominently and is considered essential repertoire for competitive rudimental snare drumming.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>           >
lR   R   |  rL   L
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1     e     &     a   |
Hand:    | (l)R  R     (r)L  L   |
Type:    |  g A  t      g A  t   |
Accent:  |    >            >     |
```

Grace notes shown in parentheses indicate they fall just before the grid position (offset -1/32 beat). Primary strokes land on eighth-note positions (1 and &), while taps fall on the sixteenth-note subdivisions (e and a).
