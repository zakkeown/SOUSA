# PAS #28: Swiss Army Triplet

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 28 |
| **Name** | Swiss Army Triplet |
| **Category** | flam |
| **Family** | advanced flam |
| **Composed Of** | Flam (#20) + double stroke (same hand as flam primary) + single stroke (opposite hand); built on a double stroke roll framework in triplet subdivision |
| **Related** | #21 Flam Accent (same sound, different sticking -- flam accent uses alternating singles while Swiss army triplet uses a double stroke), #20 Flam (the primitive flam ornament), #6 Double Stroke Open Roll (the double stroke foundation), #22 Flam Tap (another flam + double stroke pattern in eighth notes) |
| **NARD Original** | No (added by PAS in 1984 expansion to 40 rudiments) |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | triplet |
| **Cycle Length** | 2 beats |
| **Strokes Per Cycle** | 8 (6 primary strokes + 2 grace notes) |
| **Primary Strokes Per Cycle** | 6 |

The Swiss army triplet is a triplet-subdivision flam rudiment built on a double stroke roll framework. Each beat contains one triplet group of 3 primary strokes: a flammed first note followed by a same-hand tap (forming a double), then a single stroke on the opposite hand. The sticking is RRL (right-lead) or LLR (left-lead), contrasting with the flam accent (#21) which uses the same rhythm but with alternating single strokes (RLR or LRL). This double-stroke sticking means the Swiss army triplet does not naturally alternate -- it loops on the same lead hand unless deliberately switched.

### Stroke Grid

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

Note: The YAML definition encodes an alternating pattern where the lead hand switches every beat (right-lead on beat 1, left-lead on beat 2). This alternating interpretation is used for dataset generation to ensure balanced hand development. The traditional Swiss army triplet is a non-alternating pattern (all right-lead or all left-lead), but the alternating version is also widely taught and performed.

---

## 3. Sticking & Articulation

### Sticking Sequence

```
>              >
lR   L   R  |  rL   R   L
```

Each triplet group begins with a flam (grace + accented primary), followed by a tap on the opposite hand, then a tap on the same hand as the flam primary. The sticking within each beat is: flammed primary (R), opposite tap (L), same-hand tap (R) -- giving the underlying RLR hand pattern but with the first note flammed. However, the YAML encodes the second and third notes as L-R (beat 1) and R-L (beat 2), making the underlying single-stroke pattern alternate naturally between beats.

### Stroke Detail Table

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | grace | #2 | Grace note preceding right-hand flam |
| 2 | R | accent | primary | n/a | Right-hand flam primary (accented) -- beat 1 of first triplet |
| 3 | L | tap | none | n/a | Opposite-hand tap -- beat 2 of first triplet |
| 4 | R | tap | none | n/a | Same-hand tap -- beat 3 of first triplet |
| 5 | R | grace | grace | #6 | Grace note preceding left-hand flam |
| 6 | L | accent | primary | n/a | Left-hand flam primary (accented) -- beat 1 of second triplet |
| 7 | R | tap | none | n/a | Opposite-hand tap -- beat 2 of second triplet |
| 8 | L | tap | none | n/a | Same-hand tap -- beat 3 of second triplet |

### Ornament Timing

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- Stroke #1 (L grace) precedes stroke #2 (R primary); stroke #5 (R grace) precedes stroke #6 (L primary)
- The grace note falls before the triplet downbeat, not on a triplet subdivision position
- Because the triplet grid is wider than sixteenth-note grids (1/3 beat between positions vs 1/4), the grace note offset of -1/32 is well clear of the preceding triplet note

---

## 4. Dynamics Model

### Velocity Ratios

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (flam primary) | 1.0 | Flam primary strokes on beat 1 of each triplet |
| tap | 0.65 - 0.77 | The two taps following each flam |
| grace (flam) | 0.50 - 0.70 | Soft grace note preceding each primary |

The Swiss army triplet shares the same three-level dynamic hierarchy as the flam accent: grace (soft), tap (medium), accent (loud). The two rudiments are rhythmically identical in sound -- the distinction is purely in the sticking, which affects the physical execution and hand coordination.

### Accent Pattern

```
>     -     -  |  >     -     -
lR    L     R  |  rL    R     L
```

One accent per beat, always on the first triplet note (the flam). The two taps that follow are unaccented. This is identical to the flam accent's accent pattern.

### Dynamic Contour

The Swiss army triplet produces the same "strong-weak-weak" pattern within each beat as the flam accent, mirroring the natural stress pattern of triple meter. The accented flam provides a clear rhythmic anchor while the two following taps create a sense of motion toward the next accent. Despite sharing the same sound as the flam accent, the double-stroke sticking affects how players naturally phrase the pattern -- the same-hand connection between the flam primary and the second tap can subtly influence the evenness of the triplet spacing if not carefully controlled.

---

## 5. Physical / Kinesthetic

### Stroke Map

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

The physical challenge of the Swiss army triplet differs significantly from the flam accent despite the identical sound. Consider the right hand's path through one cycle: it plays a full-height accented primary (#2), drops to a low-height tap (#4), then must transition to tap-height for a grace note (#5). This is an accent-tap-grace sequence on the same hand across the beat boundary. Meanwhile, the left hand mirrors: grace (#1), tap (#3), accent (#6), tap (#8). The key distinction from the flam accent is that the double stroke (accent + tap on the same hand) creates a downstroke-to-tap sequence within each beat, similar to the flam tap (#22). The tap-to-grace transition across beats (stroke #4 to #5 for the right hand) is the critical coordination point.

### Tempo-Dependent Behavior

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accented flams. Each triplet note is distinctly articulated. The double stroke (accent + tap on same hand) is played as two clearly separate strokes. Grace notes have wide spacing from primary (-1/16 range). Focus on even triplet spacing despite the double-stroke sticking. |
| Moderate (100-140 BPM) | Wrist-driven strokes. The accent uses a controlled downstroke, catching the stick low for the second tap on the same hand. Grace note spacing at standard -1/32. The triplet flow becomes apparent. The double-stroke connection within each beat should feel natural, like a controlled double stroke roll at triplet speed. |
| Fast (140-180 BPM) | Finger control for taps and grace notes. The accent drops to half height class. Grace note spacing approaches -1/64. The double-stroke aspect of the sticking becomes an advantage at speed, as the rebound from the accent naturally feeds into the second tap on the same hand. Moeller technique employed for accent strokes. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Non-alternating (traditional)**: The original Swiss army triplet loops on a single lead hand (all lR L R or all rL R L) without alternating. This is common in Swiss Basel drumming
- **Flam accent comparison**: Played side-by-side with the flam accent (#21) to develop awareness of the sticking difference and its effect on hand coordination
- **Swiss army triplet in 6/8**: The triplet pattern fits naturally into compound time
- **Swiss army triplet with accent on third note**: A variation used in some corps-style literature for a different dynamic contour
- **No-flam preparation**: Practice the double-stroke triplet pattern (R L R | L R L) without flams to isolate the sticking from the ornament

### Prerequisites

- #20 Flam -- the basic flam ornament must be solid on both hands
- #21 Flam Accent -- understanding the triplet-based flam pattern provides the rhythmic foundation; the Swiss army triplet is the double-stroke counterpart
- #6 Double Stroke Open Roll -- the double-stroke technique (accent + tap on the same hand) is fundamental to this rudiment
- Comfort with triplet subdivision

### Builds Toward

- Advanced Swiss drumming literature and Basel drumming technique
- Development of double-stroke control within triplet subdivision
- Application in jazz comping, shuffle patterns, and compound-meter passages where the double-stroke sticking provides a different physical approach to the flam accent rhythm

### Teaching Notes

The Swiss army triplet is often taught in direct comparison with the flam accent (#21) because they produce the same rhythmic sound but use different stickings. This comparison is one of the most valuable pedagogical exercises in rudimental drumming -- it teaches students that the same rhythm can require fundamentally different physical approaches.

**Common mistakes:**
- Uneven triplet spacing: The double stroke (accent + tap on the same hand) often causes students to rush the second tap, shortening the space between it and the following single
- Confusing the sticking with the flam accent: Students may unconsciously revert to alternating single strokes (the flam accent sticking) instead of maintaining the double-stroke pattern
- Flat flams: The rapid transition from the third triplet note to the grace note for the next beat's flam is a coordination challenge
- Unequal doubles: The accent and same-hand tap should be clearly differentiated in dynamics, not played as two equal strokes
- Inconsistent alternation: When playing the alternating version, the lead-hand switch between beats can disrupt the flow

**Practice strategies:**
- Begin by playing the double-stroke triplet pattern without flams (R L R | L R L or R R L | L L R) to internalize the sticking
- Compare directly with the flam accent: play one bar of flam accent, then one bar of Swiss army triplet at the same tempo to feel the sticking difference
- Practice at very slow tempos to develop even triplet spacing despite the double-stroke sticking
- Use a metronome set to the triplet subdivision to verify that all three notes per beat are evenly spaced
- Practice the non-alternating version first (all right-lead), then all left-lead, before combining into the alternating form

**What to listen for:**
- Even triplet spacing -- all three notes per beat equidistant in time, despite the double-stroke sticking
- The Swiss army triplet should sound identical to the flam accent; if it sounds different, the triplet spacing is uneven
- Clear dynamic hierarchy: grace (soft) < tap (medium) < accent (loud)
- Consistent flam quality across both hands
- No audible "seam" at the beat boundary where the lead hand switches (in the alternating version)

### Historical Context

The Swiss army triplet originates from Swiss military drumming traditions, particularly the Basel school of drumming. Called "Triolets" in French and "Triolen" in German, the pattern appears in Swiss and French military duty calls and ceremonial drumming. It was not included in the original NARD 26 rudiments established in 1933. The Swiss army triplet was added as PAS #28 when the Percussive Arts Society expanded the standard rudiments from 26 to 40 in 1984, explicitly recognizing the influence of Swiss drumming on the American rudimental tradition. The inclusion of this rudiment alongside the flam accent (#21) highlights an important pedagogical principle: two rudiments can produce identical sounds but require fundamentally different physical techniques. Swiss Basel drumming -- one of the world's oldest continuous drumming traditions, dating to the 15th century -- uses the double-stroke triplet pattern extensively, and the PAS adoption brought this technique into the mainstream American rudimental vocabulary.

---

## 7. Notation

### Stick Notation

```
>              >
lR   L   R  |  rL   R   L
```

### Grid Visualization

```
Beat:    | 1        &        a     | 2        &        a     |
Hand:    | (l)R     L        R     | (r)L     R        L     |
Type:    |  g  A    t        t     |  g  A    t        t     |
Accent:  |     >                   |     >                   |
```

Grace notes shown in parentheses indicate they fall just before the grid position (offset -1/32 beat). The triplet grid divides each beat into three equal parts at positions 0, 1/3, and 2/3 of each beat. Note that the grid visualization is identical to the flam accent (#21) -- the difference is in the sticking (double stroke vs single stroke), not in the rhythmic placement or accent pattern.
