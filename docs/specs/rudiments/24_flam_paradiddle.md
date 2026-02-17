# PAS #24: Flam Paradiddle

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 24 |
| **Name** | Flam Paradiddle |
| **Category** | flam |
| **Family** | compound flam |
| **Composed Of** | Flam (#20) + Single Paradiddle (#16) -- a paradiddle with the accented lead stroke flammed |
| **Related** | #20 Flam (the primitive flam ornament), #16 Single Paradiddle (the underlying sticking pattern), #22 Flam Tap (another flam + double-stroke combination), #26 Flam Paradiddle-Diddle (extends this pattern with a second diddle), #6 Double Stroke Open Roll (the diddle component) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 2.5 beats |
| **Strokes Per Cycle** | 10 (8 primary strokes + 2 grace notes) |
| **Primary Strokes Per Cycle** | 8 |

The flam paradiddle is structurally identical to the single paradiddle (R L R R | L R L L) with a flam added to the accented lead stroke of each half-cycle. Each half has 4 primary strokes (1 accent + 1 tap + 2 diddle strokes) plus 1 grace note. The 2.5-beat cycle distributes 8 primary strokes across 10 sixteenth-note slots, with the fifth slot of each half being the diddle d2 landing on the "a" of the beat. The hand-lead switching property of the paradiddle is preserved: the diddle at the end of each half forces the opposite hand to lead the next half.

### Stroke Grid

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/32 | (grace before 1) | L | g |
| 2 | 0 | 1 | R | A |
| 3 | 1/4 | 1e | L | t |
| 4 | 1/2 | 1& | R | d1 |
| 5 | 3/4 | 1a | R | d2 |
| 6 | 1 + 1/4 - 1/32 | (grace before 2e) | R | g |
| 7 | 1 + 1/4 | 2e | L | A |
| 8 | 1 + 1/2 | 2& | R | t |
| 9 | 1 + 3/4 | 2a | L | d1 |
| 10 | 2 | 3 | L | d2 |

Note: The YAML defines beats_per_cycle as 2.5, which provides 10 sixteenth-note slots for 8 primary strokes plus 2 grace notes. The first half occupies beat 1 (slots 1, 1e, 1&, 1a) and the second half occupies beats 2e through 3 (slots 2e, 2&, 2a, 3), with beat 2 being a rest or transition between the two halves.

---

## 3. Sticking & Articulation

### Sticking Sequence

```
>                 >
lR   L   R R  |  rL   R   L L
```

Each group is a single paradiddle (accent, tap, diddle) with a flam on the accented lead stroke. The lead hand alternates each half-cycle.

### Stroke Detail Table

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | grace | #2 | Grace note preceding right-hand flam |
| 2 | R | accent | primary | n/a | Accented flam primary -- right-hand paradiddle lead |
| 3 | L | tap | none | n/a | Unaccented tap -- the "para" alternation stroke |
| 4 | R | d1 | none | n/a | First stroke of right-hand diddle, wrist-initiated |
| 5 | R | d2 | none | n/a | Second stroke of right-hand diddle, finger-controlled |
| 6 | R | grace | grace | #7 | Grace note preceding left-hand flam |
| 7 | L | accent | primary | n/a | Accented flam primary -- left-hand paradiddle lead |
| 8 | R | tap | none | n/a | Unaccented tap -- the "para" alternation stroke |
| 9 | L | d1 | none | n/a | First stroke of left-hand diddle, wrist-initiated |
| 10 | L | d2 | none | n/a | Second stroke of left-hand diddle, finger-controlled |

### Ornament Timing

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- Stroke #1 (L grace) precedes stroke #2 (R primary); stroke #6 (R grace) precedes stroke #7 (L primary)
- The grace note for the second flam (stroke #6) is played by the right hand, which just completed its diddle (strokes #4-5). This transition from diddle d2 to grace on the same hand is a key technical challenge.

**Diddle Timing:**
- Each diddle pair occupies one eighth-note duration (1/2 beat)
- Each individual diddle stroke occupies 1/4 beat (one sixteenth note)
- d1 falls on the "&" of the beat; d2 falls on the "a" of the beat
- At slow tempos: both strokes are distinct and clearly separated (open)
- At fast tempos: strokes closely spaced, nearly indistinguishable (closed)

---

## 4. Dynamics Model

### Velocity Ratios

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (flam primary) | 1.0 | Full-stroke flam primary at the top of each half-cycle |
| tap | 0.65 - 0.77 | Unaccented single stroke following the flam |
| d1 | 0.65 - 0.77 | First diddle stroke at tap level |
| d2 | 0.90 - 0.98 x d1 | Slight decay on the second bounce of the diddle |
| grace (flam) | 0.50 - 0.70 | Soft grace note preceding each primary |

The flam paradiddle combines the three-level dynamic hierarchy of the flam (grace-accent) with the accent-tap-diddle contour of the paradiddle. The flam adds weight and breadth to the paradiddle's accented lead stroke, making the accent-to-tap contrast even more pronounced than in the plain single paradiddle. The diddle strokes sit at the tap dynamic level, with d2 decaying slightly from d1.

### Accent Pattern

```
>              >
lR   L   RR | rL   R   LL
```

One accent per half-cycle, always on the flammed lead stroke. The tap and diddle strokes are all unaccented. This matches the single paradiddle's accent pattern with the added weight of the flam.

### Dynamic Contour

The flam paradiddle produces a repeating contour of one strong, flammed accent followed by three lighter strokes (tap + diddle). The flam grace note adds a soft onset before each accent, creating a "dip-then-peak" dynamic shape at the start of each half-cycle. The three unaccented strokes (tap, d1, d2) create a gradual dynamic decay toward the next accent. The diddle at the end provides a smooth rhythmic transition into the next half-cycle's flam. The overall feel is a strong, weighted pulse on each accent, with the flam adding breadth that makes each accent feel bigger than a plain paradiddle accent.

---

## 5. Physical / Kinesthetic

### Stroke Map

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace | tap (< 2") | fingers |
| 2 | R | accent | full (12"+) | wrist + arm |
| 3 | L | tap | low (2-6") | wrist |
| 4 | R | d1 | low (2-6") | wrist |
| 5 | R | d2 | tap (< 2") | fingers |
| 6 | R | grace | tap (< 2") | fingers |
| 7 | L | accent | full (12"+) | wrist + arm |
| 8 | R | tap | low (2-6") | wrist |
| 9 | L | d1 | low (2-6") | wrist |
| 10 | L | d2 | tap (< 2") | fingers |

The physical challenge of the flam paradiddle combines the demands of the flam and the paradiddle. The right hand's path through the first half-cycle illustrates the complexity: it plays a full-height accented primary (#2), drops to low-height for the diddle d1 (#4), finger-controls the d2 (#5), and then immediately transitions to a tap-height grace note (#6) for the opposite hand's flam. The diddle-d2-to-grace transition (strokes #5-6 on the same hand) requires the stick to remain near the head after the diddle rebound and produce a controlled grace note -- essentially staying at tap height across both strokes.

### Tempo-Dependent Behavior

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accented flams. Clear separation between grace and primary. The paradiddle sticking (accent, tap, diddle) is individually articulated. Diddles are open -- two distinct wrist strokes. Focus on matching flam quality between right-lead and left-lead halves. The diddle-to-grace transition can be executed deliberately. |
| Moderate (100-140 BPM) | Wrist-driven strokes. The accent uses a controlled downstroke, catching the stick low for the following tap. Diddles shift to wrist-finger technique (d1 wrist, d2 finger rebound). Grace note spacing at standard -1/32. The d2-to-grace transition becomes the critical control point -- the same hand must produce a finger-controlled diddle rebound and then immediately serve as a grace note. |
| Fast (140-180 BPM) | Finger control for taps, diddles, and grace notes. Accents drop to half height. The paradiddle flow becomes continuous, with the flam adding a slight rhythmic "push" at the start of each group. Grace note spacing tightens to -1/64. Moeller technique for accents to maintain dynamic contrast. The diddle-to-grace transition on the same hand becomes nearly automatic through the finger rebound. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Flamadiddle**: An alternate name for this rudiment used colloquially, emphasizing its flam + paradiddle construction
- **Flam paradiddle with inverted diddle**: Moving the diddle to the beginning of the group instead of the end
- **Double flam paradiddle**: Adding more single strokes before the diddle (analogous to the relationship between single and double paradiddle)
- **No-flam preparation**: Practice as a plain single paradiddle (R L R R | L R L L) with accents to isolate the accent-tap-diddle pattern before adding grace notes
- **Paradiddle with flam on different beats**: Moving the flam to the tap or diddle for coordination development (non-standard exercise)

### Prerequisites

- #20 Flam -- consistent flams on both hands
- #16 Single Paradiddle -- the accent-tap-diddle pattern must be secure
- #22 Flam Tap -- develops the flam-within-a-double-stroke-context skill
- Downstroke control -- the ability to play an accented flam and stop the stick low for the following tap

### Builds Toward

- #26 Flam Paradiddle-Diddle (extends with a second diddle)
- #27 Pataflafla (complex compound flam pattern)
- #36 Drag Paradiddle #1 (drag ornament version of the paradiddle)
- Application in drum corps snare drum parts, rudimental solos, and drum set fills

### Teaching Notes

The flam paradiddle is one of the most intuitive compound flam rudiments: simply add a flam to the existing single paradiddle accent. This makes it an excellent entry point into compound flam rudiments for students who already have solid paradiddle and flam fundamentals.

**Common mistakes:**
- Flat flam on the accented lead stroke: the grace must be clearly softer and earlier than the primary, especially at faster tempos where the added complexity of the paradiddle can distract from flam quality
- Losing paradiddle accent pattern: the flam should enhance the existing accent, not change the underlying accent-tap-diddle structure
- Uneven diddle: d2 significantly softer or rushed relative to d1
- Diddle-to-grace transition breakdown: the hand finishing its diddle must immediately prepare as a grace note for the opposite hand's flam -- this is the hardest coordination point
- One hand's flam consistently wider or tighter than the other

**Practice strategies:**
- Start with plain paradiddles with strong accents, then add the grace note
- Practice the diddle-to-grace transition in isolation: play RR (diddle), then immediately play the grace for the left-hand flam
- Use a metronome with the click on the paradiddle accents (every half-cycle)
- Practice each hand's lead separately before alternating
- Compare the sound of right-lead (lR L RR) and left-lead (rL R LL) halves for consistency

**What to listen for:**
- Clean, consistent flams on both right-lead and left-lead halves
- Clear accent-tap dynamic contrast (the flam accent should be noticeably louder than the tap and diddle)
- Even diddle quality with natural d2 decay
- Smooth flow through the diddle-to-grace transition
- The paradiddle's characteristic hand-switching feel with the added weight of the flam

### Historical Context

The flam paradiddle was included in the original NARD 26 rudiments when NARD was established in 1933, and its ancestry can be traced back through early American field drumming manuals. Charles Stewart Ashworth's 1812 *A New, Useful and Complete System of Drum Beating* includes patterns recognizable as flam paradiddles, and the rudiment appears in Strube's 1870 *Drum and Fife Instructor*. The flam paradiddle was retained as PAS #24 in the 1984 expansion to 40 rudiments, positioned as the second compound flam rudiment. In drum corps, the flam paradiddle is ubiquitous in snare drum book writing, where the flam adds rhythmic weight to the paradiddle's natural accent pattern. The rudiment is sometimes referred to colloquially as the "flamadiddle."

---

## 7. Notation

### Stick Notation

```
>                   >
lR   L   R R    |   rL   R   L L
```

### Grid Visualization

```
Beat:    | 1     e     &     a   | .     2e    2&    2a  | 3         |
Hand:    | (l)R  L     R     R   |       (r)L  R     L   | L         |
Type:    |  g A  t     d1    d2  |        g A  t     d1  | d2        |
Accent:  |    >                  |           >           |           |
```

Grace notes shown in parentheses indicate they fall just before the grid position (offset -1/32 beat). The first half occupies beat 1 (4 sixteenth slots). The second half begins on beat 2e and ends on beat 3. The diddle pairs occupy the "&" and "a" positions of each half's final beat.
