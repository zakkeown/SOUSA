# PAS #29: Inverted Flam Tap

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 29 |
| **Name** | Inverted Flam Tap |
| **Category** | flam |
| **Family** | advanced flam |
| **Composed Of** | Flam (#20) + tap; the "inversion" of #22 Flam Tap, where the tap precedes the flam instead of following it |
| **Related** | #22 Flam Tap (the non-inverted version: flam then tap), #20 Flam (the primitive flam ornament), #6 Double Stroke Open Roll (the underlying LL RR hand pattern), #27 Pataflafla (another advanced flam with rapid flam transitions) |
| **NARD Original** | No (added by PAS in 1984 expansion to 40 rudiments) |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 1.5 beats |
| **Strokes Per Cycle** | 6 (4 primary strokes + 2 grace notes) |
| **Primary Strokes Per Cycle** | 4 |

The inverted flam tap is the mirror image of the flam tap (#22). Where the flam tap places the flam on the first stroke of each double (lR R | rL L), the inverted flam tap places the flam on the second stroke: the tap comes first, then the flam follows on the same hand (L lR | R rL). The "inversion" refers to the reversal of the tap-flam order within each double-stroke pair. The underlying hand pattern is an offset double stroke roll (LL RR) starting on the "and" of each count, rather than the standard double stroke roll (RR LL) starting on the downbeat. The YAML defines a 1.5-beat cycle to accommodate the three sixteenth-note positions per half-cycle.

### Stroke Grid

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | 0 | 1 | L | t |
| 2 | 1/4 - 1/32 | (grace before 1e) | L | g |
| 3 | 1/4 | 1e | R | A |
| 4 | 1/2 | 1& | R | t |
| 5 | 3/4 - 1/32 | (grace before 1a) | R | g |
| 6 | 3/4 | 1a | L | A |

Note: The primary strokes occupy sixteenth-note grid positions: 1, 1e, 1&, 1a. The pattern fills the beat completely. Each half of the cycle is a tap followed by a flam: L then lR (first half), R then rL (second half). The tap and the grace note for the following flam are played by the same hand, which is the distinctive coordination challenge of this rudiment.

---

## 3. Sticking & Articulation

### Sticking Sequence

```
      >        >
L   lR   |  R  rL
```

Each group consists of a tap followed by a flam where the grace note is played by the same hand that just played the tap. This is the "inversion" of the flam tap (#22), where the sequence is flam-then-tap instead of tap-then-flam.

**Comparison with Flam Tap (#22):**
- Flam Tap: `lR R | rL L` -- flam first, then tap on the same hand as the primary
- Inverted Flam Tap: `L lR | R rL` -- tap first, then flam where the grace is played by the hand that just tapped

### Stroke Detail Table

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | tap | none | n/a | Left-hand tap preceding right-hand flam |
| 2 | L | grace | grace | #3 | Left-hand grace note for right-hand flam (same hand as the preceding tap) |
| 3 | R | accent | primary | n/a | Right-hand flam primary (accented) |
| 4 | R | tap | none | n/a | Right-hand tap preceding left-hand flam |
| 5 | R | grace | grace | #6 | Right-hand grace note for left-hand flam (same hand as the preceding tap) |
| 6 | L | accent | primary | n/a | Left-hand flam primary (accented) |

### Ornament Timing

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- Stroke #2 (L grace) precedes stroke #3 (R primary); stroke #5 (R grace) precedes stroke #6 (L primary)
- Critical: the grace note hand is the **same hand** that just played the preceding tap. The left hand plays tap (#1) then immediately plays grace (#2) for the right-hand flam. This tap-to-grace transition on the same hand is the defining technical challenge
- The tap and grace are separated by a full sixteenth-note duration minus the grace offset (1/4 - 1/32 = 7/32 beat), but at fast tempos this interval becomes very short

---

## 4. Dynamics Model

### Velocity Ratios

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (flam primary) | 1.0 | Flam primary strokes on the "e" and "a" sixteenth positions |
| tap | 0.65 - 0.77 | The tap preceding each flam |
| grace (flam) | 0.50 - 0.70 | Soft grace note preceding each flam primary |

The inverted flam tap shares the same three-level dynamic hierarchy as the flam tap: grace (soft), tap (medium), accent (loud). However, the dynamic contour is reversed: where the flam tap goes accent-then-tap (loud-soft), the inverted flam tap goes tap-then-accent (soft-loud). This creates a pickup or anacrusis feel rather than the flam tap's downbeat-oriented character.

### Accent Pattern

```
-     >     -     >
L     lR    R     rL
```

Accents fall on the "e" and "a" sixteenth-note positions (the flam primaries), with taps on the "1" and "&" positions. This placement of accents on weak sixteenth-note positions gives the inverted flam tap its characteristic upbeat, syncopated feel.

### Dynamic Contour

The inverted flam tap produces a repeating "weak-strong" pattern at the eighth-note level: unaccented tap, then accented flam, alternating between hands. This creates an upbeat, pickup-oriented feel that contrasts sharply with the flam tap's "strong-weak" (accent-then-tap) character. The overall effect is one of constant forward motion, as each tap propels toward the following accented flam. The dynamic shape within each tap-flam pair is a small crescendo (soft tap rising to loud accent), giving the rudiment an energetic, driving quality.

---

## 5. Physical / Kinesthetic

### Stroke Map

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | tap | low (2-6") | wrist |
| 2 | L | grace | tap (< 2") | fingers |
| 3 | R | accent | full (12"+) | wrist + arm |
| 4 | R | tap | low (2-6") | wrist |
| 5 | R | grace | tap (< 2") | fingers |
| 6 | L | accent | full (12"+) | wrist + arm |

The primary physical challenge of the inverted flam tap is the upstroke required to transition from the tap to the accent. Consider the left hand's path: it plays a low-height tap (#1), immediately transitions to a tap-height grace (#2), and then must be ready high for its next accent (#6). The right hand mirrors: accent (#3), tap (#4), grace (#5). The critical moment is the tap-to-grace transition (#1 to #2, #4 to #5): the hand plays a low tap and must immediately drop even lower to grace height for the flam. This is followed by the opposite hand's full-height accent. The Moeller whip stroke is particularly valuable here -- the tap can serve as the preparatory "pickup" motion for the opposite hand's accented flam.

### Tempo-Dependent Behavior

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accented flams. The tap-to-grace transition has ample time. Each tap and flam are distinctly articulated. Grace notes have wide spacing from primary (-1/16 range). Focus on the upstroke motion: getting from tap height up to accent height for the opposite hand's flam. |
| Moderate (100-140 BPM) | Wrist-driven strokes. The Moeller whip becomes useful for the accent strokes: the tap serves as the preparatory downward motion, and the grace note feeds naturally into the opposite hand's upward whip for the accent. Grace note spacing at standard -1/32. The tap-to-grace connection should feel like a single flowing motion. |
| Fast (140-180 BPM) | Finger control for taps and grace notes. Accents drop to half height class. Grace note spacing approaches -1/64. The Moeller whip is essential for maintaining dynamic contrast at speed. The tap and grace become nearly continuous -- the hand barely lifts between the low tap and the even lower grace. The rudiment begins to feel like a continuous rocking motion between hands. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Non-alternating inverted flam tap**: All flams led by the same hand (all L lR or all R rL), used as a preparatory exercise
- **Inverted flam tap with accent on the tap**: Adding an accent to the tap for a double-accent variation
- **Inverted flam tap combined with flam tap**: Alternating one beat of flam tap with one beat of inverted flam tap, an excellent coordination exercise
- **No-flam preparation**: Practice the offset double stroke pattern (L R | R L) with accents on the second of each double to isolate the accent placement from the ornament

### Prerequisites

- #22 Flam Tap -- understanding the standard flam tap is essential before learning its inversion; the two rudiments should be studied as a pair
- #20 Flam -- the basic flam ornament must be solid on both hands
- #6 Double Stroke Open Roll -- the offset double stroke pattern (starting on the "and") underlies the inverted flam tap
- Moeller upstroke technique -- the ability to generate an accent from a low stick position is critical

### Builds Toward

- #27 Pataflafla (combines back-to-back flams requiring similar tap-to-grace coordination)
- Advanced rudimental solo passages where flam tap and inverted flam tap alternate
- Development of Moeller technique and accent control from low stick positions
- Application in drumset fills and corps-style snare parts where the upbeat accent pattern creates forward motion

### Teaching Notes

The inverted flam tap is best understood as the "mirror" of the flam tap (#22). Where the flam tap places the flam first (strong beat) and the tap second (weak beat), the inverted flam tap reverses this: tap first (weak beat), then flam (strong beat). This inversion changes the musical character entirely, from a downbeat-oriented pattern to an upbeat, pickup-oriented pattern.

**Common mistakes:**
- Confusing the sticking with the flam tap: Students must clearly distinguish `L lR | R rL` (inverted) from `lR R | rL L` (standard)
- Flat flams: The tap-to-grace transition can cause the grace note to arrive too late, collapsing the flam into a flat (simultaneous) stroke
- Uneven dynamics: The tap should be noticeably softer than the accented flam, but the rapid tap-to-grace-to-accent sequence can cause the tap to creep up in volume
- Rhythmic delay before the accent: The challenge of lifting the stick from tap height to accent height can cause a hesitation or "hiccup" before the flam
- Losing the sixteenth-note evenness: All four primary strokes should be evenly spaced at the sixteenth-note grid, regardless of which are taps and which are accents

**Practice strategies:**
- Begin by playing the offset accented doubles without flams: L >R | R >L (accent on the second of each double)
- Add the grace notes only after the accent-on-second-double pattern is comfortable and even
- Practice the flam tap and inverted flam tap back-to-back at the same tempo to internalize the difference
- Use the Moeller whip stroke from the beginning -- this rudiment is one of the best vehicles for developing Moeller technique
- Isolate the tap-to-grace transition: play the tap, pause, then execute the grace + opposite primary
- Practice at very slow tempos with a metronome on the sixteenth-note grid to ensure even spacing

**What to listen for:**
- Even sixteenth-note spacing across all four primary strokes
- Clear dynamic hierarchy: grace (soft) < tap (medium) < accent (loud)
- The tap-flam pair should sound like a pickup into a downbeat, not two equal events
- No rhythmic hesitation before the accented flam
- Identical flam quality on both hands (lR and rL flams should sound the same)
- The overall pattern should have a forward-driving, upbeat character distinct from the flam tap's grounded feel

### Historical Context

The inverted flam tap was one of the 14 new rudiments added when the Percussive Arts Society expanded the standard rudiments from 26 to 40 in 1984. It was added as PAS #29, directly following the basic flam rudiments and compound flam rudiments in the numbering scheme. The inclusion of the inverted flam tap alongside the original flam tap (#22, an NARD original) demonstrates the PAS expansion's focus on exploring variations and inversions of established patterns. The concept of "inverting" a rudiment -- reversing the order of its component strokes -- is a common pedagogical technique in rudimental drumming, and the inverted flam tap is perhaps the clearest example of this principle in the PAS 40. Bill Bachman, author of *Stick Technique* and a Modern Drummer columnist, has cited the inverted flam tap as one of the ten most important rudiments for developing Moeller technique and accent control.

---

## 7. Notation

### Stick Notation

```
      >        >
L   lR   |  R  rL
```

### Grid Visualization

```
Beat:    | 1     e     &     a   |
Hand:    | L     (l)R  R     (r)L |
Type:    | t      g A  t      g A |
Accent:  |           >          > |
```

Grace notes shown in parentheses indicate they fall just before the grid position (offset -1/32 beat). Primary strokes land on all four sixteenth-note positions (1, e, &, a). Taps fall on the stronger metric positions (1 and &) while accented flams fall on the weaker positions (e and a), creating the characteristic syncopated, upbeat feel.
