# PAS #23: Flamacue

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 23 |
| **Name** | Flamacue |
| **Category** | flam |
| **Family** | compound flam |
| **Composed Of** | Flam (#20) + five-note single stroke pattern with accents on first and last primary strokes |
| **Related** | #20 Flam (the primitive flam ornament), #21 Flam Accent (flam + triplet taps), #1 Single Stroke Roll (the underlying alternating singles), #22 Flam Tap (another compound flam with accent-tap framework) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 3 beats |
| **Strokes Per Cycle** | 12 (10 primary strokes + 2 grace notes) |
| **Primary Strokes Per Cycle** | 10 |

The flamacue is built on a five-note single stroke pattern per half-cycle, bookended by two accented strokes: the first is a flam (grace + accented primary) and the fifth is a standalone accent. Three unaccented taps fill the space between. The full cycle contains two mirror-image halves (right-lead and left-lead), yielding 10 primary strokes across 3 beats (12 sixteenth-note slots). The distinctive feature of the flamacue is the accent at the END of each group rather than solely at the beginning, giving the rudiment a characteristic sense of resolution or "landing" at the end of each phrase.

### Stroke Grid

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/32 | (grace before 1) | L | g |
| 2 | 0 | 1 | R | A |
| 3 | 1/4 | 1e | L | t |
| 4 | 1/2 | 1& | R | t |
| 5 | 3/4 | 1a | L | t |
| 6 | 1 | 2 | R | A |
| 7 | 1 + 1/2 - 1/32 | (grace before 2&) | R | g |
| 8 | 1 + 1/2 | 2& | L | A |
| 9 | 1 + 3/4 | 2a | R | t |
| 10 | 2 | 3 | L | t |
| 11 | 2 + 1/4 | 3e | R | t |
| 12 | 2 + 1/2 | 3& | L | A |

Note: The first half (strokes 1-6) occupies beats 1-2 with the flam on beat 1 and the accent on beat 2. The second half (strokes 7-12) occupies beats 2&-3& with the flam on beat 2& and the accent on beat 3&. The two halves are separated by an eighth-note gap (beat 2e is empty), giving each group a pickup-like entry after the preceding accent.

---

## 3. Sticking & Articulation

### Sticking Sequence

```
>              >           >              >
lR   L   R   L   R    |   rL   R   L   R   L
```

Each group of five primary strokes begins with a flam (accented) and ends with a standalone accent. The three middle strokes are unaccented taps alternating hands.

### Stroke Detail Table

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | grace | #2 | Grace note preceding right-hand flam |
| 2 | R | accent | primary | n/a | Accented flam primary -- opening of right-lead half |
| 3 | L | tap | none | n/a | First unaccented tap |
| 4 | R | tap | none | n/a | Second unaccented tap |
| 5 | L | tap | none | n/a | Third unaccented tap |
| 6 | R | accent | none | n/a | Closing accent of right-lead half (no flam) |
| 7 | R | grace | grace | #8 | Grace note preceding left-hand flam |
| 8 | L | accent | primary | n/a | Accented flam primary -- opening of left-lead half |
| 9 | R | tap | none | n/a | First unaccented tap |
| 10 | L | tap | none | n/a | Second unaccented tap |
| 11 | R | tap | none | n/a | Third unaccented tap |
| 12 | L | accent | none | n/a | Closing accent of left-lead half (no flam) |

### Ornament Timing

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- Stroke #1 (L grace) precedes stroke #2 (R primary); stroke #7 (R grace) precedes stroke #8 (L primary)
- Only the FIRST primary stroke of each half-cycle receives a flam; the final accent (strokes #6 and #12) is a clean accent without a grace note

---

## 4. Dynamics Model

### Velocity Ratios

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (flam primary) | 1.0 | Accented primary of the opening flam |
| accent (closing) | 1.0 | Standalone accent at the end of each group |
| tap | 0.65 - 0.77 | The three unaccented taps between the accents |
| grace (flam) | 0.50 - 0.70 | Soft grace note preceding each flam primary |

The flamacue has a distinctive two-accent dynamic architecture: each half-cycle begins AND ends with a full-velocity accent, with three taps at reduced velocity in between. This "accent-tap-tap-tap-accent" contour creates a sense of departure and arrival that distinguishes the flamacue from other flam rudiments. The grace note on the opening flam adds weight to the first accent, while the closing accent is clean and sharp.

### Accent Pattern

```
>              >           >              >
lR   L   R   L   R    |   rL   R   L   R   L
```

Two accents per half-cycle: one on the opening flam and one on the closing stroke. The three taps in between are unaccented.

### Dynamic Contour

The flamacue produces a symmetrical "bookend" accent pattern within each 5-note group: a strong, flammed opening accent, three even taps at a reduced dynamic, and a strong closing accent. This creates a distinctive arc -- loud, soft, soft, soft, loud -- that gives the rudiment its unique musical character. The closing accent provides a sense of resolution that differentiates the flamacue from most other rudiments, where the accent typically falls only at the beginning of each group. The two halves of the cycle mirror each other with opposite hand leads, ensuring balanced development. The overall 3-beat cycle has a slightly asymmetric feel, as each half does not align with a simple 1-beat or 2-beat boundary.

---

## 5. Physical / Kinesthetic

### Stroke Map

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace | tap (< 2") | fingers |
| 2 | R | accent | full (12"+) | wrist + arm |
| 3 | L | tap | low (2-6") | wrist |
| 4 | R | tap | low (2-6") | wrist |
| 5 | L | tap | low (2-6") | wrist |
| 6 | R | accent | full (12"+) | wrist + arm |
| 7 | R | grace | tap (< 2") | fingers |
| 8 | L | accent | full (12"+) | wrist + arm |
| 9 | R | tap | low (2-6") | wrist |
| 10 | L | tap | low (2-6") | wrist |
| 11 | R | tap | low (2-6") | wrist |
| 12 | L | accent | full (12"+) | wrist + arm |

The primary physical challenge of the flamacue is the rapid height transitions within each hand. The right hand, for example, plays a full-height accented flam primary (#2), drops to low-height taps (#4), rises back to full-height for the closing accent (#6), then immediately transitions to a tap-height grace note (#7). This full-low-full-tap height sequence demands excellent stick control and downstroke/upstroke technique. The closing accent followed immediately by a grace note (same hand: R plays accent #6, then grace #7) is particularly challenging -- the hand must go from a full-height accent to a near-motionless grace position in a single sixteenth-note interval.

### Tempo-Dependent Behavior

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for both accents. Clear separation between grace and primary on the opening flam. The three taps are deliberate wrist strokes at low height. The accent-to-grace transition (strokes 6-7) can be executed with controlled upstroke motion. Each stroke is individually articulated. |
| Moderate (100-140 BPM) | Wrist-driven strokes predominate. The opening flam accent uses a controlled downstroke. Taps flow naturally between accents. The closing accent employs an upstroke preparation leading into the next group's grace note. Grace note spacing at standard -1/32. The overall five-note pattern begins to feel like a single musical gesture. |
| Fast (140-160 BPM) | Finger control for taps. Both accents drop to half height class. The grace note spacing tightens toward -1/64. The accent-to-grace transition requires Moeller whip technique to maintain the height differential. The five-note groups flow continuously with minimal separation between halves. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Flamacue with flam on the closing accent**: In some traditional interpretations, the final accent also receives a flam, making the last note of one group overlap with the first of the next (the shared-flam interpretation)
- **Flamacue as a pickup figure**: The five-note group can be treated as a pickup into a downbeat, placing the closing accent on a strong beat
- **Accented flamacue**: Adding dynamic shaping to the three taps (crescendo toward the closing accent) for musical phrasing
- **No-flam preparation**: Practice the five-note accent pattern (>R L R L R>) as alternating singles with accents to isolate the accent pattern from the ornament

### Prerequisites

- #20 Flam -- the basic flam ornament must be consistent on both hands
- #1 Single Stroke Roll -- alternating single strokes at various dynamics
- Downstroke and upstroke control -- the ability to play an accent and immediately transition to a grace note position (accent-to-grace on the same hand)

### Builds Toward

- #27 Pataflafla (more complex compound flam pattern)
- #30 Flam Drag (flam combined with drag)
- Application in rudimental solos where a strong phrase ending is needed
- The accent-to-grace transition skill developed here transfers to all advanced flam rudiments

### Teaching Notes

The flamacue is often underappreciated among the compound flam rudiments, but it develops a critical skill: the accent-to-grace transition on the same hand. This transition (closing accent immediately followed by a grace note for the next group) is one of the most demanding coordination challenges in rudimental drumming and transfers directly to many advanced patterns.

**Common mistakes:**
- Losing the closing accent: players often treat the last stroke as a tap, eliminating the distinctive "bookend" accent pattern
- Flat flam on the opening: the grace note must be clearly softer and earlier than the primary
- Uneven taps: the three taps between accents should be evenly spaced at the sixteenth-note grid positions
- Same dynamic for both accents: while both are at accent velocity, the opening flam should sound different from the closing clean accent due to the grace note adding breadth
- Rushing through the taps to reach the closing accent

**Practice strategies:**
- Begin with the five-note accent pattern without flams: >R L R L R> | >L R L R L>
- Add the grace note to the opening accent only after the accent pattern is comfortable
- Isolate the accent-to-grace transition: play the closing accent, pause, then execute the grace + opposite primary
- Practice at very slow tempos to develop the stick height transitions (full-low-low-low-full-tap)
- Count "1-e-&-a-2" for the first half to internalize the sixteenth-note grid positions

**What to listen for:**
- Clear two-accent bookend structure in each five-note group
- Even taps between the accents at a noticeably lower dynamic
- Clean flam on the opening (grace clearly preceding primary)
- Clean accent on the closing (no unintended grace or flam)
- Identical quality regardless of which hand leads

### Historical Context

The flamacue was included in the original NARD 26 rudiments when NARD was established in 1933, and its roots trace back to early American military drumming traditions. The name "flamacue" is likely a contraction of "flam" and "cue," reflecting its historical use as a signal or cue pattern in field drumming. Some historians attribute its codification to Bruce and Emmett's drumming manuals. The flamacue appears in Ashworth's *A New, Useful and Complete System of Drum Beating* (1812) in an early form. It was retained as PAS #23 in the 1984 expansion to 40 rudiments, positioned as the first compound flam rudiment. The flamacue's distinctive closing accent makes it one of the most recognizable rudiments in the flam family, and it remains a staple of competitive rudimental solos and drum corps snare drum parts.

---

## 7. Notation

### Stick Notation

```
>              >           >              >
lR   L   R   L   R    |   rL   R   L   R   L
```

### Grid Visualization

```
Beat:    | 1     e     &     a   | 2                 &     a   | 3     e     &         |
Hand:    | (l)R  L     R     L   | R                 (r)L  R   | L     R     L         |
Type:    |  g A  t     t     t   | A                  g A  t   | t     t     A         |
Accent:  |    >                  | >                    >      |                 >     |
```

Grace notes shown in parentheses indicate they fall just before the grid position (offset -1/32 beat). The first half occupies beats 1-2 (flam on 1, accent on 2). The second half occupies beats 2&-3& (flam on 2&, accent on 3&).
