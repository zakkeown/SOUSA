# PAS #25: Single Flammed Mill

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 25 |
| **Name** | Single Flammed Mill |
| **Category** | flam |
| **Family** | compound flam |
| **Composed Of** | Flam (#20) + diddle on the opposite hand -- a flam followed by a double stroke (diddle) on the hand opposite to the flam primary |
| **Related** | #20 Flam (the primitive flam ornament), #6 Double Stroke Open Roll (the diddle component), #22 Flam Tap (another flam + double-stroke combination), #24 Flam Paradiddle (flam + paradiddle, a related compound pattern), #16 Single Paradiddle (the mill is a reverse paradiddle variant) |
| **NARD Original** | No (added by PAS in 1984) |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 2 beats |
| **Strokes Per Cycle** | 8 (6 primary strokes + 2 grace notes) |
| **Primary Strokes Per Cycle** | 6 |

The single flammed mill consists of two mirror-image halves, each containing a flam (grace + accented primary) followed by a diddle on the opposite hand. Each half has 3 primary strokes (1 accent + 2 diddle strokes) plus 1 grace note, for a total of 6 primary strokes and 2 grace notes across the full 2-beat cycle. The pattern is compact: each half occupies 3 sixteenth-note slots (accent + d1 + d2), with the fourth slot in each beat left empty or serving as the start of the next half.

The term "mill" in drumming refers to a reverse paradiddle pattern (e.g., R R L R instead of R L R R). The single flammed mill takes the concept of doubling on the same hand and adds a flam to the lead stroke. In the SOUSA YAML definition, the pattern is simplified to flam + opposite-hand diddle, emphasizing the flam-then-diddle alternation.

### Stroke Grid

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/32 | (grace before 1) | L | g |
| 2 | 0 | 1 | R | A |
| 3 | 1/4 | 1e | L | d1 |
| 4 | 1/2 | 1& | L | d2 |
| 5 | 1 - 1/32 | (grace before 2) | R | g |
| 6 | 1 | 2 | L | A |
| 7 | 1 + 1/4 | 2e | R | d1 |
| 8 | 1 + 1/2 | 2& | R | d2 |

Note: Each half occupies 3 sixteenth-note positions (the accent on the beat, d1 on the "e", and d2 on the "&"), leaving the "a" position empty before the next half begins. This creates a characteristic rhythmic gap -- a dotted-eighth + sixteenth feel -- that distinguishes the single flammed mill from denser rudiments.

---

## 3. Sticking & Articulation

### Sticking Sequence

```
>           >
lR   L L |  rL   R R
```

Each group is a flam followed by a diddle on the opposite hand. The pattern alternates: right-hand flam with left-hand diddle, then left-hand flam with right-hand diddle.

### Stroke Detail Table

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | grace | #2 | Grace note preceding right-hand flam |
| 2 | R | accent | primary | n/a | Accented flam primary -- right-hand lead |
| 3 | L | d1 | none | n/a | First stroke of left-hand diddle, wrist-initiated |
| 4 | L | d2 | none | n/a | Second stroke of left-hand diddle, finger-controlled |
| 5 | R | grace | grace | #6 | Grace note preceding left-hand flam |
| 6 | L | accent | primary | n/a | Accented flam primary -- left-hand lead |
| 7 | R | d1 | none | n/a | First stroke of right-hand diddle, wrist-initiated |
| 8 | R | d2 | none | n/a | Second stroke of right-hand diddle, finger-controlled |

### Ornament Timing

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- Stroke #1 (L grace) precedes stroke #2 (R primary); stroke #5 (R grace) precedes stroke #6 (L primary)
- The grace note for the second flam (stroke #5) is played by the right hand, which is the same hand that played the flam primary in the first half (stroke #2). This means the right hand transitions from accent (#2) to grace (#5) over the course of one beat.

**Diddle Timing:**
- Each diddle pair occupies one eighth-note duration (1/2 beat)
- Each individual diddle stroke occupies 1/4 beat (one sixteenth note)
- d1 falls on the "e" of the beat; d2 falls on the "&" of the beat
- The diddle is on the **opposite hand** from the flam primary, which is a distinctive feature of this rudiment
- At slow tempos: both strokes are distinct and clearly separated (open)
- At fast tempos: strokes closely spaced, nearly indistinguishable (closed)

---

## 4. Dynamics Model

### Velocity Ratios

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (flam primary) | 1.0 | Full-stroke flam primary at the start of each half-cycle |
| d1 | 0.65 - 0.77 | First diddle stroke at tap level on the opposite hand |
| d2 | 0.90 - 0.98 x d1 | Slight decay on the second bounce of the diddle |
| grace (flam) | 0.50 - 0.70 | Soft grace note preceding each primary |

The single flammed mill has a simple two-level primary dynamic: accented flam, then unaccented diddle. Unlike the paradiddle family where the diddle is on the same hand as the accent, here the diddle is on the opposite hand, which affects the dynamic transition. The hand that plays the flam primary can fully relax after its accent, while the opposite hand executes the diddle at tap level. This creates a clear accent-then-decay contour with the dynamic weight concentrated at the beginning of each half-cycle.

### Accent Pattern

```
>           >
lR   LL  |  rL   RR
```

One accent per half-cycle, always on the flammed lead stroke. The diddle strokes are unaccented. The pattern is simple and regular: accent every beat.

### Dynamic Contour

The single flammed mill produces a clean, regular pulse: each beat begins with a strong, flammed accent, followed by a softer diddle that decays into a brief silence before the next flam. The grace note adds a soft onset before each accent, creating a "dip-peak-decay" shape per half-cycle. The rhythmic gap on the "a" of each beat (no stroke in that position) allows a moment of rest that makes each flam-diddle group feel like a distinct musical event. This gives the single flammed mill a more spacious, less dense feel compared to rudiments that fill all four sixteenth-note positions.

---

## 5. Physical / Kinesthetic

### Stroke Map

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace | tap (< 2") | fingers |
| 2 | R | accent | full (12"+) | wrist + arm |
| 3 | L | d1 | low (2-6") | wrist |
| 4 | L | d2 | tap (< 2") | fingers |
| 5 | R | grace | tap (< 2") | fingers |
| 6 | L | accent | full (12"+) | wrist + arm |
| 7 | R | d1 | low (2-6") | wrist |
| 8 | R | d2 | tap (< 2") | fingers |

The physical motion of the single flammed mill involves a distinctive coordination pattern. While one hand plays a full-height accented primary, the other hand is positioned low for the grace note it just played, then must immediately initiate a diddle. The grace-hand transitions directly from its grace note role (tap height, fingers) to d1 (low height, wrist) -- essentially upgrading its motion from a gentle grace touch to an active wrist-driven diddle stroke. Meanwhile, the accent hand descends from full height to tap height to prepare as the grace for the next flam. The pattern can be thought of as a continuous exchange: each hand alternates between "accent duty" (high, powerful) and "diddle duty" (low, controlled).

### Tempo-Dependent Behavior

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accented flams. Clear separation between grace and primary. Diddles are open -- two distinct wrist strokes on the opposite hand. The rhythmic gap on the "a" position is clearly audible. Focus on the grace-to-diddle transition on the same hand and matching flam quality between halves. |
| Moderate (100-140 BPM) | Wrist-driven strokes. The flam primary uses a downstroke to set up the opposite hand's diddle. Diddles shift to wrist-finger technique (d1 wrist, d2 finger rebound). Grace note spacing at standard -1/32. The accent-to-grace transition on the same hand becomes smoother. The rhythmic gap is less prominent but still present. |
| Fast (140-160 BPM) | Finger control for diddles and grace notes. Accents drop to half height. The flam-diddle groups flow into each other with minimal separation. Grace note spacing tightens to -1/64. The diddle rebound from d2 can feed directly into the grace note for the next flam (since d2 is on the same hand as the upcoming grace), creating a nearly continuous motion. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Full mill (reverse paradiddle) interpretation**: Some traditions interpret the single flammed mill as a flammed reverse paradiddle (lR R L R | rL L R L), with an additional single stroke after the diddle. The SOUSA YAML uses the simpler flam + diddle interpretation.
- **Double flammed mill**: Two flammed mill groups before switching hands (not a standard PAS rudiment)
- **No-flam preparation**: Practice as accented strokes followed by diddles (>R LL | >L RR) to isolate the accent-diddle pattern
- **Continuous mills**: Playing the pattern continuously at speed to develop the accent-to-grace-to-diddle hand exchange

### Prerequisites

- #20 Flam -- consistent flams on both hands
- #6 Double Stroke Open Roll -- controlled diddles with matched d1/d2 quality
- #22 Flam Tap -- develops the flam-within-a-double-stroke-context skill
- Understanding of the reverse paradiddle (mill) sticking concept

### Builds Toward

- #27 Pataflafla (complex compound flam pattern with multiple flams per group)
- #29 Inverted Flam Tap (related pattern with flam placement on the second stroke)
- Application in drum corps solo literature and advanced rudimental etudes

### Teaching Notes

The single flammed mill was one of the 14 new rudiments added when the PAS expanded the standard list from 26 to 40 in 1984. It is sometimes overlooked in favor of more familiar compound flam rudiments like the flam paradiddle, but it develops a unique coordination skill: the simultaneous management of a flam on one hand and a diddle on the other.

**Common mistakes:**
- Grace note and d1 colliding: the grace note (opposite hand from diddle) and the first diddle stroke (same hand as grace) happen in quick succession -- the grace must lead the primary clearly before the diddle hand begins
- Uneven diddle: d2 significantly softer or rushed relative to d1
- Losing the rhythmic gap: players may rush through the "a" position to reach the next flam, destroying the characteristic spacing
- Flat flams: the grace must remain clearly softer and earlier than the primary
- Inconsistent hand balance: one hand's accent-to-diddle transition consistently smoother than the other

**Practice strategies:**
- Begin with plain accented strokes followed by opposite-hand diddles: >R LL | >L RR (no flams)
- Add grace notes only after the accent-diddle pattern is comfortable and even
- Isolate the hand exchange: play the full right-lead group (lR LL), pause, then the left-lead group (rL RR)
- Practice the grace-to-d1 transition in isolation: play a grace note, then immediately play a wrist-controlled diddle on the same hand
- Use a metronome on the beat to anchor the flam placements

**What to listen for:**
- Clean, consistent flams on both right-lead and left-lead halves
- Even, matched diddle quality across both halves
- Clear rhythmic gap on the "a" position (the rudiment should not sound like continuous sixteenth notes)
- Smooth hand exchange between the accent-hand and the diddle-hand roles
- The flam should sound like a single weighted event, not two separate notes

### Historical Context

The single flammed mill was one of the 14 rudiments added to the PAS standard list in 1984 when the Percussive Arts Society expanded from the NARD 26 to 40 International Drum Rudiments. The term "mill" derives from "windmill" and refers to the reverse paradiddle sticking pattern (RRLR instead of RLRR). Adding a flam to the lead stroke creates the "flammed mill." Although not part of the original NARD 26, the concept of the mill pattern has roots in early American and European drumming traditions. The single flammed mill was categorized by PAS under the flam rudiment family rather than the diddle family, emphasizing the flam ornament as its primary defining feature. In competitive rudimental drumming, the single flammed mill appears in advanced solo literature and drum corps snare drum parts, where the alternating flam-diddle texture creates a distinctive rhythmic effect.

---

## 7. Notation

### Stick Notation

```
>               >
lR   L L    |   rL   R R
```

### Grid Visualization

```
Beat:    | 1     e     &     a   | 2     e     &     a   |
Hand:    | (l)R  L     L         | (r)L  R     R         |
Type:    |  g A  d1    d2   rest |  g A  d1    d2   rest |
Accent:  |    >                  |    >                  |
```

Grace notes shown in parentheses indicate they fall just before the grid position (offset -1/32 beat). Each half-cycle occupies 3 of the 4 sixteenth-note positions within a beat, leaving the "a" position empty. The diddle pairs occupy the "e" and "&" positions following each flam.
