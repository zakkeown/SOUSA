# PAS #26: Flam Paradiddle-Diddle

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 26 |
| **Name** | Flam Paradiddle-Diddle |
| **Category** | flam |
| **Family** | compound flam |
| **Composed Of** | Flam (#20) + Paradiddle-Diddle (#19) -- a paradiddle-diddle with the accented lead stroke flammed |
| **Related** | #20 Flam (the primitive flam ornament), #19 Paradiddle-Diddle (the underlying sticking pattern), #24 Flam Paradiddle (flam + single paradiddle, the simpler variant), #16 Single Paradiddle (root paradiddle pattern), #6 Double Stroke Open Roll (the diddle component) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 (or 6/8 -- the 6-stroke grouping fits compound meters naturally) |
| **Base Subdivision** | sextuplet (six notes per beat) |
| **Cycle Length** | 2 beats |
| **Strokes Per Cycle** | 14 (12 primary strokes + 2 grace notes) |
| **Primary Strokes Per Cycle** | 12 |

The flam paradiddle-diddle is structurally identical to the paradiddle-diddle (R L R R L L | L R L L R R) with a flam added to the accented lead stroke of each half-cycle. Each half has 6 primary strokes (1 accent + 1 tap + 4 diddle strokes in 2 diddle pairs) plus 1 grace note. The sextuplet subdivision places 6 primary strokes per beat, with one complete half-cycle fitting within a single beat. The two consecutive diddles at the end of each half (RR LL or LL RR) create a brief double-stroke roll excerpt, giving the rudiment its characteristic "rolling" quality. The flam on the lead stroke adds weight and breadth to the natural paradiddle-diddle accent.

### Stroke Grid

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/32 | (grace before 1) | L | g |
| 2 | 0 | 1 | R | A |
| 3 | 1/6 | 1la | L | t |
| 4 | 2/6 | 1li | R | d1 |
| 5 | 3/6 | 1& | R | d2 |
| 6 | 4/6 | 1la | L | d1 |
| 7 | 5/6 | 1li | L | d2 |
| 8 | 1 - 1/32 | (grace before 2) | R | g |
| 9 | 1 | 2 | L | A |
| 10 | 1 + 1/6 | 2la | R | t |
| 11 | 1 + 2/6 | 2li | L | d1 |
| 12 | 1 + 3/6 | 2& | L | d2 |
| 13 | 1 + 4/6 | 2la | R | d1 |
| 14 | 1 + 5/6 | 2li | R | d2 |

---

## 3. Sticking & Articulation

### Sticking Sequence

```
>                          >
lR   L   R R   L L    |   rL   R   L L   R R
```

Each group is a paradiddle-diddle (accent, tap, diddle, diddle) with a flam on the accented lead stroke. The lead hand alternates each half-cycle.

### Stroke Detail Table

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | grace | #2 | Grace note preceding right-hand flam |
| 2 | R | accent | primary | n/a | Accented flam primary -- right-hand lead |
| 3 | L | tap | none | n/a | Unaccented tap -- the "para" alternation stroke |
| 4 | R | d1 | none | n/a | First stroke of first diddle (right hand), wrist-initiated |
| 5 | R | d2 | none | n/a | Second stroke of first diddle (right hand), finger-controlled |
| 6 | L | d1 | none | n/a | First stroke of second diddle (left hand), wrist-initiated |
| 7 | L | d2 | none | n/a | Second stroke of second diddle (left hand), finger-controlled |
| 8 | R | grace | grace | #9 | Grace note preceding left-hand flam |
| 9 | L | accent | primary | n/a | Accented flam primary -- left-hand lead |
| 10 | R | tap | none | n/a | Unaccented tap -- the "para" alternation stroke |
| 11 | L | d1 | none | n/a | First stroke of first diddle (left hand), wrist-initiated |
| 12 | L | d2 | none | n/a | Second stroke of first diddle (left hand), finger-controlled |
| 13 | R | d1 | none | n/a | First stroke of second diddle (right hand), wrist-initiated |
| 14 | R | d2 | none | n/a | Second stroke of second diddle (right hand), finger-controlled |

### Ornament Timing

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- Stroke #1 (L grace) precedes stroke #2 (R primary); stroke #8 (R grace) precedes stroke #9 (L primary)
- The grace note for the second flam (stroke #8) is played by the right hand, which just completed its diddle d2 (stroke #5) three sextuplet positions earlier. There is time to prepare, but the hand must transition from a low diddle position back to grace-note readiness.
- Because the sextuplet grid has narrower spacing (1/6 beat between positions), the grace note offset of -1/32 is relatively large -- it falls well before the preceding sextuplet position, so there is no risk of collision with the previous note.

**Diddle Timing:**
- Each diddle pair occupies one sextuplet-note duration (1/3 beat)
- Each individual diddle stroke occupies 1/6 beat (one sextuplet note)
- The two consecutive diddles alternate hands (RR LL or LL RR), creating a brief double-stroke roll excerpt
- Right-hand half: first diddle (R) on 1li-1&, second diddle (L) on 1la-1li
- Left-hand half: first diddle (L) on 2li-2&, second diddle (R) on 2la-2li
- At slow tempos: all four diddle strokes are distinct and clearly separated (open)
- At fast tempos: the diddle pairs blur into a smooth roll-like texture (closed)

---

## 4. Dynamics Model

### Velocity Ratios

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (flam primary) | 1.0 | Full-stroke flam primary at the top of each half-cycle |
| tap | 0.65 - 0.77 | Unaccented single stroke following the flam |
| d1 (first diddle) | 0.65 - 0.77 | First stroke of each diddle at tap level |
| d2 (first diddle) | 0.90 - 0.98 x d1 | Slight decay on the second stroke of the first diddle |
| d1 (second diddle) | 0.65 - 0.77 | First stroke of the second diddle at tap level |
| d2 (second diddle) | 0.90 - 0.98 x d1 | Slight decay on the second stroke of the second diddle |
| grace (flam) | 0.50 - 0.70 | Soft grace note preceding each primary |

The flam paradiddle-diddle combines the three-level flam dynamic (grace-primary contrast) with the paradiddle-diddle's accent-tap-diddle-diddle contour. The flam adds weight and breadth to the paradiddle-diddle's accent, making it the most dynamically prominent rudiment in the compound flam family. The four consecutive diddle strokes (two diddles alternating hands) sit at tap level and create a sustained low-dynamic plateau that contrasts strongly with the flammed accent.

### Accent Pattern

```
>                       >
lR   L   RR  LL    |   rL   R   LL  RR
```

One accent per half-cycle (per beat), always on the flammed lead stroke. The tap and all four diddle strokes are unaccented. The accent falls on every beat, creating a clear rhythmic pulse.

### Dynamic Contour

The flam paradiddle-diddle produces a six-note dynamic contour: one strong, flammed accent, one tap, then four diddle strokes (two diddles of two strokes each). The flam grace note creates a soft onset before the peak accent, followed by a quick drop to tap level and a sustained low-dynamic plateau through the four diddle strokes. Each diddle pair has subtle internal decay (d2 slightly softer than d1). The two consecutive diddles at the end produce a "rolling" texture that distinguishes this rudiment from the flam paradiddle (which has only one diddle). The overall feel within each beat is: rise (grace), peak (accent), drop (tap), plateau (diddle-diddle). The alternating hand lead ensures symmetrical dynamic treatment across the cycle.

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
| 6 | L | d1 | low (2-6") | wrist |
| 7 | L | d2 | tap (< 2") | fingers |
| 8 | R | grace | tap (< 2") | fingers |
| 9 | L | accent | full (12"+) | wrist + arm |
| 10 | R | tap | low (2-6") | wrist |
| 11 | L | d1 | low (2-6") | wrist |
| 12 | L | d2 | tap (< 2") | fingers |
| 13 | R | d1 | low (2-6") | wrist |
| 14 | R | d2 | tap (< 2") | fingers |

The physical demands of the flam paradiddle-diddle combine flam technique with the two-consecutive-diddle challenge of the paradiddle-diddle. The right hand's path through the first half illustrates: full-height accent (#2), drop to low for diddle d1 (#4), finger-controlled d2 (#5), then prepare as grace note (#8) for the next flam. The hand must transition from a finger-controlled diddle rebound to a grace note position, which is facilitated by the d2 naturally leaving the stick near the head. The two consecutive diddles (RR LL or LL RR) require executing a double-stroke roll fragment at sextuplet speed, demanding strong finger control. The hand exchange between the two consecutive diddles (R-to-L or L-to-R) is the critical control point.

### Tempo-Dependent Behavior

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm accents with clear height contrast. Grace notes clearly audible with wide spacing. The sextuplet subdivision is felt as six distinct events per beat. All four diddle strokes are individually articulated. Focus on the hand exchange between the two consecutive diddles and on matching flam quality between halves. |
| Moderate (100-140 BPM) | Wrist-driven strokes. The accent employs Moeller whip technique. Diddles transition to wrist-finger technique (d1 wrist, d2 finger rebound). Grace note spacing at standard -1/32. The consecutive diddles begin to feel like a brief double-stroke roll excerpt. The sextuplet subdivision produces notes at the same rate as sixteenth notes at 150% of the BPM, demanding substantial finger control. |
| Fast (140-160 BPM) | All motions become compact. Accents are wrist-only from half height. All diddle strokes are finger-controlled at tap height. The two consecutive diddles merge into a smooth, roll-like texture. Grace note spacing tightens to -1/64. The flam paradiddle-diddle at fast tempos feels like a flowing, continuous pattern with accented pulses on each beat. The d2-to-grace transition becomes nearly automatic through finger rebound momentum. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Flamadiddle-diddle**: An alternate name for this rudiment used colloquially
- **Flam paradiddle-diddle in 6/8**: The six-note grouping fits naturally into compound time, with the flam accent on the downbeat of each bar
- **Flam paradiddle-diddle as triplets**: Reinterpreted in a triplet feel where each 6-stroke group spans one beat of dotted-quarter-note pulse
- **No-flam preparation**: Practice as a plain paradiddle-diddle (R L R R L L | L R L L R R) with accents to isolate the accent-tap-diddle-diddle pattern before adding grace notes
- **Orchestrated around the kit**: Distribute the flam accent on cymbal/tom and taps on snare, with the consecutive diddles creating a rapid fill

### Prerequisites

- #20 Flam -- consistent flams on both hands
- #19 Paradiddle-Diddle -- the accent-tap-diddle-diddle pattern must be comfortable in sextuplets
- #24 Flam Paradiddle -- develops the flam-within-a-paradiddle concept with one fewer diddle
- #6 Double Stroke Open Roll -- controlled diddles, especially the RR LL hand exchange

### Builds Toward

- Advanced compound rudiments that combine flams with extended paradiddle patterns
- Application in compound-meter drum corps literature and jazz waltz contexts
- Flam drag patterns (#30) that combine flams with drag ornaments

### Teaching Notes

The flam paradiddle-diddle is the most complex of the four compound flam rudiments, combining the flam ornament with the longest paradiddle variant. The critical insight for students is that this rudiment is simply a paradiddle-diddle with a flam on the accent -- if the paradiddle-diddle is already comfortable, adding the grace note is a relatively straightforward extension.

The sextuplet subdivision deserves special attention. Unlike most flam rudiments (which use eighth or sixteenth subdivision), the flam paradiddle-diddle inherits the sextuplet feel of the paradiddle-diddle. Students who are comfortable with the flam paradiddle (#24, sixteenth notes) may struggle when transitioning to the sextuplet grid, where notes occur at 1/6-beat intervals instead of 1/4-beat intervals.

**Common mistakes:**
- Playing in sixteenth notes instead of sextuplets -- the six-note group must be evenly spaced within each beat, not four-note groups
- Three consecutive taps problem: the second diddle d2 of one group, followed by the grace note for the next flam, followed by the tap after the flam, creates three notes in rapid succession on nearby hands. Keeping these steady requires careful practice.
- Uneven diddle quality between the first and second consecutive diddles
- Gap or flam at the hand exchange between the two consecutive diddles (R-to-L or L-to-R)
- Flat flams at tempo due to the sextuplet speed making it harder to maintain grace-to-primary separation
- Losing the accent-tap contrast within the fast sextuplet subdivision

**Practice strategies:**
- Begin with plain paradiddle-diddles with accents, then add grace notes
- Practice the double-stroke roll fragment (RRLL or LLRR) in isolation at sextuplet speed
- Isolate the hand exchange between consecutive diddles
- Use a metronome set to dotted quarter notes in 6/8 to anchor the accent pattern
- Practice right-lead (lR L RR LL) and left-lead (rL R LL RR) halves separately
- Compare flam quality between halves and work to match them

**What to listen for:**
- Clean, consistent flams on both right-lead and left-lead halves
- Correct sextuplet feel -- six evenly spaced notes per beat, not four
- Even, matched diddle quality across both consecutive diddles
- Smooth hand exchange between the first and second diddle (no gap or flam)
- Clear accent-tap dynamic contrast despite the fast sextuplet subdivision
- The flam should add weight to the accent without disrupting the flow

### Historical Context

The flam paradiddle-diddle was included in the original NARD 26 rudiments when NARD was established in 1933. Its roots trace back at least to Ashworth's 1812 *A New, Useful and Complete System of Drum Beating*, which depicts and names the pattern. Rumrille (1817) also includes the rudiment, though he calls it simply a "Flam Paradiddle." The flam paradiddle-diddle was retained as PAS #26 in the 1984 expansion to 40 rudiments, positioned as the final compound flam rudiment. In the drum corps tradition, the flam paradiddle-diddle is used in compound-meter passages (6/8, 12/8) where the sextuplet grouping aligns naturally with the rhythmic feel. Its combination of flam, single strokes, and two consecutive diddles makes it one of the most technically demanding rudiments in the flam family. The rudiment is sometimes referred to as the "flamadiddle-diddle" in colloquial usage.

---

## 7. Notation

### Stick Notation

```
>                          >
lR   L   R R   L L    |   rL   R   L L   R R
```

### Grid Visualization

```
Beat:    | 1     la    li    &     la    li  | 2     la    li    &     la    li  |
Hand:    | (l)R  L     R     R     L     L   | (r)L  R     L     L     R     R   |
Type:    |  g A  t     d1    d2    d1    d2  |  g A  t     d1    d2    d1    d2  |
Accent:  |    >                              |    >                              |
```

Grace notes shown in parentheses indicate they fall just before the grid position (offset -1/32 beat). The sextuplet grid divides each beat into six equal parts at positions 0, 1/6, 2/6, 3/6, 4/6, and 5/6 of each beat. Each half-cycle fills one complete beat with 6 primary strokes plus 1 grace note.
