# PAS #27: Pataflafla

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 27 |
| **Name** | Pataflafla |
| **Category** | flam |
| **Family** | advanced flam |
| **Composed Of** | Flam (#20) + single stroke taps; a four-note group where the first and fourth notes are flammed, with two single-stroke taps between them |
| **Related** | #20 Flam (the primitive flam ornament), #22 Flam Tap (flam + tap in eighth-note doubles), #29 Inverted Flam Tap (tap-before-flam pattern), #1 Single Stroke Roll (underlying alternating singles) |
| **NARD Original** | No (added by PAS in 1984 expansion to 40 rudiments) |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 2 beats |
| **Strokes Per Cycle** | 12 (8 primary strokes + 4 grace notes) |
| **Primary Strokes Per Cycle** | 8 |

The name "pataflafla" is onomatopoeic, derived from French military drumming: "pa" (single stroke), "ta" (single stroke), "fla" (flam), "fla" (flam). Each half of the cycle contains four primary strokes arranged as flam-tap-flam-tap in sixteenth notes, with flams alternating between right-hand and left-hand leads. The full cycle of 2 beats contains two mirror-image halves, yielding 8 primary strokes filling all 8 sixteenth-note positions and 4 grace notes.

### Stroke Grid

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/32 | (grace before 1) | L | g |
| 2 | 0 | 1 | R | A |
| 3 | 1/4 | 1e | L | t |
| 4 | 1/2 - 1/32 | (grace before 1&) | R | g |
| 5 | 1/2 | 1& | L | A |
| 6 | 3/4 | 1a | R | t |
| 7 | 1 - 1/32 | (grace before 2) | R | g |
| 8 | 1 | 2 | L | A |
| 9 | 1 + 1/4 | 2e | R | t |
| 10 | 1 + 1/2 - 1/32 | (grace before 2&) | L | g |
| 11 | 1 + 1/2 | 2& | R | A |
| 12 | 1 + 3/4 | 2a | L | t |

Note: Every sixteenth-note position is occupied by a primary stroke. The grace notes sit just before their respective primary strokes. The flams alternate hands within each half: right-hand flam (lR) then left-hand flam (rL) in the first beat, then left-hand flam (rL) then right-hand flam (lR) in the second beat. This produces the distinctive back-to-back flams with alternating leads that define the pataflafla.

---

## 3. Sticking & Articulation

### Sticking Sequence

```
>        >        >        >
lR  L  rL  R  |  rL  R  lR  L
```

Each beat contains two flam-tap pairs. The lead hand alternates within each pair (right flam then left flam in beat 1, left flam then right flam in beat 2), creating the back-to-back flams with different leading hands that are the signature challenge of this rudiment.

### Stroke Detail Table

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | grace | #2 | Grace note preceding right-hand flam |
| 2 | R | accent | primary | n/a | Right-hand flam primary (accented) -- "pa" |
| 3 | L | tap | none | n/a | Left-hand tap -- "ta" |
| 4 | R | grace | grace | #5 | Grace note preceding left-hand flam |
| 5 | L | accent | primary | n/a | Left-hand flam primary (accented) -- first "fla" |
| 6 | R | tap | none | n/a | Right-hand tap -- second "fla" position tap |
| 7 | R | grace | grace | #8 | Grace note preceding left-hand flam |
| 8 | L | accent | primary | n/a | Left-hand flam primary (accented) |
| 9 | R | tap | none | n/a | Right-hand tap |
| 10 | L | grace | grace | #11 | Grace note preceding right-hand flam |
| 11 | R | accent | primary | n/a | Right-hand flam primary (accented) |
| 12 | L | tap | none | n/a | Left-hand tap |

### Ornament Timing

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- Four flams per cycle: strokes #1-2 (lR), #4-5 (rL), #7-8 (rL), #10-11 (lR)
- The critical challenge is the back-to-back flams: the tap hand (stroke #3 or #6) must immediately transition to a grace note (stroke #4 or #7) for the next flam on the opposite hand. This rapid tap-to-grace transition on the same hand, occurring twice per beat, is the defining technical difficulty of the pataflafla
- Grace notes must remain consistent in quality across all four flams despite the rapid hand transitions

---

## 4. Dynamics Model

### Velocity Ratios

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (flam primary) | 1.0 | All four flam primary strokes per cycle |
| tap | 0.65 - 0.77 | The two taps per beat between flams |
| grace (flam) | 0.50 - 0.70 | All four grace notes preceding flam primaries |

The pataflafla features the same three-level dynamic hierarchy as other flam rudiments (grace, tap, accent), but the density of flams -- four per two-beat cycle -- makes consistent dynamic control more demanding. The accent-tap alternation at the sixteenth-note level creates a driving, insistent rhythmic pulse.

### Accent Pattern

```
>     -     >     -  |  >     -     >     -
lR    L     rL    R  |  rL    R     lR    L
```

Accents fall on every other sixteenth note (the flammed notes), with taps between. This yields four accents per 2-beat cycle, creating a strong eighth-note pulse from the accented flams.

### Dynamic Contour

The pataflafla produces a rapid, alternating "strong-weak-strong-weak" pattern at the sixteenth-note level. Each pair of sixteenths contains one accented flam and one unaccented tap. The continuous alternation of accents and taps, combined with the alternating flam lead hand, creates a driving, propulsive feel. The back-to-back flams with different leading hands give the rudiment a characteristic "rolling" quality where no single hand dominates. The overall contour across the 2-beat cycle is symmetrical: beat 2 mirrors beat 1 with opposite hand leads.

---

## 5. Physical / Kinesthetic

### Stroke Map

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace | tap (< 2") | fingers |
| 2 | R | accent | full (12"+) | wrist + arm |
| 3 | L | tap | low (2-6") | wrist |
| 4 | R | grace | tap (< 2") | fingers |
| 5 | L | accent | full (12"+) | wrist + arm |
| 6 | R | tap | low (2-6") | wrist |
| 7 | R | grace | tap (< 2") | fingers |
| 8 | L | accent | full (12"+) | wrist + arm |
| 9 | R | tap | low (2-6") | wrist |
| 10 | L | grace | tap (< 2") | fingers |
| 11 | R | accent | full (12"+) | wrist + arm |
| 12 | L | tap | low (2-6") | wrist |

The primary physical challenge of the pataflafla is the rapid height transitions required by the back-to-back flams. Consider the right hand's path through one beat: it plays a full-height accented flam primary (#2), immediately drops to tap-height for a grace note (#4), then transitions to low-height for a tap (#6). Meanwhile, the left hand mirrors this: grace (#1), tap (#3), full-height accent (#5). Each hand cycles through three different height classes within a single beat, demanding exceptional stick control and the ability to execute rapid downstroke-to-grace transitions. The tap-to-grace transition (e.g., stroke #3 to #4, where the left hand plays a tap and the right hand must be in grace position) is the critical coordination point.

### Tempo-Dependent Behavior

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accented flams. Each of the four flams per cycle is clearly articulated with wide grace-to-primary spacing (-1/16 range). The taps between flams are distinct, deliberate wrist strokes. The back-to-back flam challenge is manageable at this tempo because there is sufficient time for the tap-to-grace transition. Focus on matching all four flams in quality. |
| Moderate (100-140 BPM) | Wrist-driven strokes predominate. The accent uses a controlled downstroke, and the tap-to-grace transition tightens. Grace note spacing at standard -1/32. The rudiment begins to flow as a continuous pattern rather than discrete flam-tap pairs. The Moeller whip motion may be introduced for the accented strokes to facilitate the rapid height changes. |
| Fast (140-160 BPM) | Finger control for taps and grace notes. All accents drop to half height class. Grace note spacing approaches -1/64. The back-to-back flams become the primary limiting factor -- the hands must execute two different flam orientations per beat with minimal separation. Moeller technique is essential. The stick path becomes a continuous, circular motion to keep up with the rapid height transitions. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Traditional "pa-ta-fla-fla" interpretation**: Some sources describe the pataflafla as two singles followed by two flams (R L lR rL) rather than alternating flam-tap pairs. This interpretation places the flams on notes 3 and 4 of each four-note group instead of notes 1 and 3
- **Single-lead pataflafla**: Playing all flams with the same lead hand (all lR or all rL) as a preparatory exercise
- **Pataflafla with accented taps**: Adding accents to the taps for an all-accent variation used in corps-style drumming
- **Extended pataflafla**: Combining pataflafla groups with other rudiments (paradiddles, drags) in rudimental solo passages

### Prerequisites

- #20 Flam -- the basic flam ornament must be rock-solid with both hands
- #22 Flam Tap -- the flam-tap hand coordination is the building block for this rudiment
- #1 Single Stroke Roll -- alternating single strokes at the sixteenth-note level
- Comfort with rapid tap-to-grace transitions on the same hand (developed through flam tap and flamacue practice)

### Builds Toward

- Advanced rudimental solo passages requiring rapid flam transitions
- Corps-style snare drum literature where flam density is high
- Development of ambidextrous flam control (four different flam orientations per cycle)

### Teaching Notes

The pataflafla is one of the most demanding flam rudiments due to its high density of flams -- four per two-beat cycle, with two per beat occurring back-to-back with different leading hands. This places extreme demands on the tap-to-grace transition skill.

**Common mistakes:**
- Flat flams: With four flams per cycle, maintaining consistent grace-to-primary separation is challenging. Back-to-back flams are especially prone to collapsing into flat flams
- Uneven flam quality: The four flams often differ in spacing and dynamic contrast, especially between dominant and non-dominant hand leads
- Rushing the taps: The taps between flams get compressed as players focus on the flam execution
- Losing the alternation: The back-to-back flams with alternating leads can cause hand confusion at speed
- Grace note too loud on back-to-back flams: The rapid transition from tap to grace sometimes produces an over-emphasized grace note

**Practice strategies:**
- Begin with the flam-tap pattern (alternating flams with taps) at very slow tempos, focusing on one beat at a time
- Practice the right-hand-lead beat and left-hand-lead beat separately before combining
- Isolate the tap-to-grace transition: play a tap, pause, then execute the grace + opposite-hand primary
- Use a metronome set to eighth notes, placing flams on every click to verify evenness
- Record and compare the sound of all four flams within the cycle -- they should be indistinguishable

**What to listen for:**
- All four flams should sound identical in grace-to-primary spacing and dynamic contrast
- Even sixteenth-note spacing across all 8 primary strokes
- Clear three-level dynamic hierarchy: grace (soft) < tap (medium) < accent (loud)
- The back-to-back flams should flow smoothly without hesitation or rhythmic hiccup
- Identical quality between the right-hand-lead beat and the left-hand-lead beat

### Historical Context

The pataflafla derives its name from onomatopoeic French military drumming terminology: "pa" (right stroke), "ta" (left stroke), "fla" (flam), "fla" (flam). Its roots lie in French and Swiss military field drumming traditions, particularly the Basel drumming school. The pattern was not included in the original NARD 26 rudiments established in 1933. It was added as PAS #27 when the Percussive Arts Society expanded the standard rudiments from 26 to 40 in 1984, recognizing the need for more advanced flam patterns in the canon. The pataflafla represents the culmination of flam-based technique: it requires all the coordination skills developed through the basic flam (#20), flam tap (#22), and compound flam rudiments, applied at a higher density. It appears frequently in competitive rudimental solos and drum corps snare drum parts, where its driving, flam-dense character adds intensity and complexity to musical passages.

---

## 7. Notation

### Stick Notation

```
>        >        >        >
lR  L  rL  R  |  rL  R  lR  L
```

### Grid Visualization

```
Beat:    | 1     e     &     a   | 2     e     &     a   |
Hand:    | (l)R  L     (r)L  R   | (r)L  R     (l)R  L   |
Type:    |  g A  t      g A  t   |  g A  t      g A  t   |
Accent:  |    >            >     |    >            >     |
```

Grace notes shown in parentheses indicate they fall just before the grid position (offset -1/32 beat). All eight sixteenth-note positions are occupied by primary strokes. The four flams land on positions 1, &, 2, and 2&, with taps on the e and a positions.
