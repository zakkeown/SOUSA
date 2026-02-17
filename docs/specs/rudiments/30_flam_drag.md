# PAS #30: Flam Drag

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 30 |
| **Name** | Flam Drag |
| **Category** | flam |
| **Family** | advanced flam |
| **Composed Of** | Flam (#20) + tap + Drag (#31); a cross-category rudiment combining the flam ornament (single grace note before a primary) with the drag ornament (two grace notes before a primary) |
| **Related** | #20 Flam (the primitive flam ornament), #31 Drag (the primitive drag ornament), #21 Flam Accent (similar triplet-based structure but with single-stroke taps instead of drags), #32 Single Drag Tap (drag + tap pattern), #22 Flam Tap (flam + tap pattern) |
| **NARD Original** | No (added by PAS in 1984 expansion to 40 rudiments) |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 3 beats |
| **Strokes Per Cycle** | 12 (6 primary strokes + 2 flam grace notes + 4 drag grace notes) |
| **Primary Strokes Per Cycle** | 6 |

The flam drag is the only rudiment in the PAS 40 that combines two different ornament types: the flam (single grace note) and the drag (double grace note). Each half of the 3-beat cycle contains three primary strokes arranged as: flammed accent, tap, then dragged accent. The flam and drag alternate between hands, creating a balanced pattern. The YAML encodes the pattern over 3 beats with sixteenth-note subdivision, giving 12 sixteenth-note slots for 6 primary strokes -- the remaining slots accommodate the ornament timing.

### Stroke Grid

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/32 | (grace before 1) | L | g (flam) |
| 2 | 0 | 1 | R | A |
| 3 | 1/4 | 1e | L | t |
| 4 | 1/2 - 1/16 | (1st drag grace before 1&) | L | g (drag) |
| 5 | 1/2 - 1/32 | (2nd drag grace before 1&) | L | g (drag) |
| 6 | 1/2 | 1& | R | A |
| 7 | 1 + 1/2 - 1/32 | (grace before 2&) | R | g (flam) |
| 8 | 1 + 1/2 | 2& | L | A |
| 9 | 1 + 3/4 | 2a | R | t |
| 10 | 2 - 1/16 | (1st drag grace before 3) | R | g (drag) |
| 11 | 2 - 1/32 | (2nd drag grace before 3) | R | g (drag) |
| 12 | 2 | 3 | L | A |

Note: Each half-cycle occupies 1.5 beats. The first half spans beats 1 to 1& (flam on 1, tap on 1e, drag on 1&). The second half spans beats 2& to 3 (flam on 2&, tap on 2a, drag on 3). The drag grace notes are played by the opposite hand from the drag primary, and they form a quick double-bounce: the first grace at -1/16 beat and the second at -1/32 beat before the primary.

---

## 3. Sticking & Articulation

### Sticking Sequence

```
>        >        >        >
lR  L  llR   |   rL  R  rrL
```

Each group of three primary strokes begins with a flam (single grace + accent), followed by a tap on the opposite hand, then a drag (double grace + accent) on the same hand as the flam primary. The lead hand alternates between halves.

### Stroke Detail Table

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | flam grace | #2 | Single grace note preceding right-hand flam |
| 2 | R | accent | primary | n/a | Right-hand flam primary (accented) |
| 3 | L | tap | none | n/a | Left-hand tap between flam and drag |
| 4 | L | grace | drag grace 1 | #6 | First drag grace note (same hand as second grace, opposite from drag primary) |
| 5 | L | grace | drag grace 2 | #6 | Second drag grace note (closer to primary than first grace) |
| 6 | R | accent | primary | n/a | Right-hand drag primary (accented) |
| 7 | R | grace | flam grace | #8 | Single grace note preceding left-hand flam |
| 8 | L | accent | primary | n/a | Left-hand flam primary (accented) |
| 9 | R | tap | none | n/a | Right-hand tap between flam and drag |
| 10 | R | grace | drag grace 1 | #12 | First drag grace note |
| 11 | R | grace | drag grace 2 | #12 | Second drag grace note |
| 12 | L | accent | primary | n/a | Left-hand drag primary (accented) |

### Ornament Timing

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- Stroke #1 (L grace) precedes stroke #2 (R primary); stroke #7 (R grace) precedes stroke #8 (L primary)
- Standard single-grace-note flam behavior identical to #20 Flam

**Drag Grace Notes:**
- First grace offset from primary: **-1/16 beat** (the earlier of the two drag grace notes)
- Second grace offset from primary: **-1/32 beat** (the later of the two, closer to the primary)
- Both grace notes played by the **same hand** (opposite from the drag primary)
- Strokes #4-5 (both L) precede stroke #6 (R primary); strokes #10-11 (both R) precede stroke #12 (L primary)
- The two drag grace notes form a quick diddle (same-hand double bounce) leading into the primary
- The drag grace notes should sound like a brief buzz or ruff before the primary, not two distinct taps
- Important: the tap (stroke #3 or #9) and the first drag grace (stroke #4 or #10) are played by the **same hand**. The tap-to-drag-grace transition is a key coordination challenge, as the hand must shift from a low-height tap to the first of two rapid grace strokes

---

## 4. Dynamics Model

### Velocity Ratios

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (flam primary) | 1.0 | Accented primary strokes preceded by flam grace |
| accent (drag primary) | 1.0 | Accented primary strokes preceded by drag graces |
| tap | 0.65 - 0.77 | The single tap between flam and drag |
| grace (flam) | 0.50 - 0.70 | Single grace note for flam ornament |
| grace (drag) | 0.45 - 0.65 | Each of the two drag grace notes; slightly softer than flam grace |

The flam drag has a four-level dynamic hierarchy: drag grace (softest), flam grace, tap, accent (loudest). Both the flam and drag primaries are at full accent velocity, creating two strong anchor points per half-cycle. The tap provides a dynamic valley between the two accented ornaments.

### Accent Pattern

```
>     -     >     |     >     -     >
lR    L     llR   |     rL    R     rrL
```

Two accents per half-cycle: one on the flam and one on the drag. The tap between them is unaccented. This dual-accent pattern is similar to the flamacue (#23) but with a drag replacing the closing clean accent.

### Dynamic Contour

The flam drag produces a distinctive "accent-valley-accent" pattern within each half-cycle. The flam provides a broad, full opening accent (enhanced by the single grace note), the tap creates a brief dynamic dip, and the drag provides a slightly different-textured closing accent (enhanced by the double grace note ruff). The contrast between the flam's single-grace breadth and the drag's double-grace buzz gives the two accents subtly different sonic characters, even though both are at full velocity. The overall shape within each half-cycle is a shallow "U" contour: strong-weak-strong, with the two strong beats colored differently by their ornaments.

---

## 5. Physical / Kinesthetic

### Stroke Map

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace (flam) | tap (< 2") | fingers |
| 2 | R | accent | full (12"+) | wrist + arm |
| 3 | L | tap | low (2-6") | wrist |
| 4 | L | grace (drag 1) | tap (< 2") | fingers |
| 5 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 6 | R | accent | full (12"+) | wrist + arm |
| 7 | R | grace (flam) | tap (< 2") | fingers |
| 8 | L | accent | full (12"+) | wrist + arm |
| 9 | R | tap | low (2-6") | wrist |
| 10 | R | grace (drag 1) | tap (< 2") | fingers |
| 11 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 12 | L | accent | full (12"+) | wrist + arm |

The flam drag presents a unique physical challenge because it requires two different ornament techniques within each half-cycle. Consider the left hand's path through the first half: it plays a flam grace (#1, tap height, fingers), then a tap (#3, low height, wrist), then two drag grace notes (#4-5, tap height, fingers/rebound). The hand must execute a grace-tap-grace-grace sequence, transitioning between different height classes and motion types in rapid succession. The tap-to-drag transition (stroke #3 to #4) is particularly demanding: the hand goes from a low-height wrist tap to an immediate pair of finger-controlled grace strokes, requiring a rapid switch in stick control technique. The drag grace pair itself requires a controlled double bounce at tap height, with the second grace benefiting from the rebound of the first.

### Tempo-Dependent Behavior

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for both accented primaries. Flam grace clearly audible with wide spacing (-1/16 range). Drag grace notes are individually articulated as two distinct soft strokes (open drag). The tap is a deliberate wrist stroke. All ornaments are clearly distinguishable. Focus on the contrast between flam (one grace) and drag (two graces) textures. |
| Moderate (100-140 BPM) | Wrist-driven strokes. Flam grace at standard -1/32 offset. Drag grace notes at -1/16 and -1/32 offsets, beginning to close up and sound more like a buzz. The tap-to-drag transition tightens. The two ornament types should be clearly different in character: the flam is a single "kiss" before the primary, while the drag is a brief "buzz" or ruff. |
| Fast (140-160 BPM) | Finger control for all grace notes and taps. Accents drop to half height class. Flam grace approaches -1/64 offset. Drag grace notes close to -1/32 and -1/64 offsets, sounding nearly as a single buzz. The tap-to-drag transition requires very quick finger preparation. Both ornament types compress but should remain distinguishable: the flam is tighter than the drag, which retains a slight "buzz" quality from its two grace notes. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Flam drag in triplets**: Some interpretations place the flam drag in triplet subdivision rather than sixteenth notes, with the flam on the first triplet note, tap on the second, and drag on the third
- **Flam drag with accented tap**: Adding an accent to the tap for a continuous-accent variation
- **Reversed flam drag**: Drag first, then tap, then flam -- reversing the ornament order within each group
- **No-ornament preparation**: Practice the accent-tap-accent pattern as clean strokes (R L R | L R L) to isolate the rhythm from the ornaments, then add the flam, then add the drag

### Prerequisites

- #20 Flam -- the flam ornament must be consistent on both hands
- #31 Drag -- the drag ornament (double grace + primary) must be comfortable on both hands
- #21 Flam Accent -- the triplet/three-note grouping provides structural familiarity
- #32 Single Drag Tap -- the drag-tap combination develops relevant coordination
- Ability to differentiate between flam and drag ornaments in rapid succession

### Builds Toward

- Cross-category rudiment combinations in advanced rudimental solos
- Development of the ability to rapidly switch between different ornament techniques
- Application in drumset and corps-style passages where flam and drag textures alternate for timbral variety

### Teaching Notes

The flam drag is the culminating rudiment of the flam category, sitting at PAS #30 as the last and most complex of the 11 flam rudiments. It is also the first rudiment in the PAS 40 that explicitly combines two different ornament types (flam and drag), making it a bridge between the flam category (PAS #20-30) and the drag category (PAS #31-40).

**Common mistakes:**
- Confusing the drag grace notes with flam grace notes: The drag uses two grace notes on the same hand, while the flam uses one grace note on the opposite hand. At speed, students often reduce the drag to a single grace, making it sound like another flam
- Uneven drag grace notes: The two drag grace notes should be a quick, even double bounce, not two separate taps and not one louder than the other
- Losing the tap between ornaments: The tap is squeezed between two ornaments and can be rushed or swallowed at speed
- Same texture for flam and drag: The two ornaments should have distinct sonic characters -- the flam is a "breadth" ornament (widening the primary), while the drag is a "ruff" ornament (buzzing into the primary)
- Inconsistent ornament quality between halves: The right-hand flam + left-hand drag (first half) should be a mirror image of the left-hand flam + right-hand drag (second half)

**Practice strategies:**
- Begin by practicing flams and drags separately, then in isolation within the three-note pattern
- Play the pattern without ornaments first (>R L >R | >L R >L) to internalize the accent pattern
- Add flams only: lR L R | rL R L (a simplified flam accent)
- Add drags only: R L llR | L R rrL (a drag pattern)
- Combine both ornaments only after each is comfortable individually
- Practice at very slow tempos to clearly hear the textural difference between the flam (one grace) and the drag (two graces)
- Record and listen to verify that the two ornament types are sonically distinguishable

**What to listen for:**
- Two distinct ornament textures within each half-cycle: the flam's single-grace "breadth" and the drag's double-grace "ruff"
- Even tap between the two ornaments at a clearly lower dynamic
- Consistent flam quality (identical grace-to-primary spacing across the cycle)
- Consistent drag quality (identical double-grace buzz across the cycle)
- The overall pattern should have a clear accent-valley-accent contour per half-cycle
- Balanced execution between halves (right-lead and left-lead groups sound identical)

### Historical Context

The flam drag was not included in the original NARD 26 rudiments. It was added as PAS #30 when the Percussive Arts Society expanded the standard rudiments from 26 to 40 in 1984. As the last rudiment in the flam category, it serves as a capstone that bridges the flam and drag categories by combining both ornament types in a single pattern. The concept of cross-category rudiments -- combining ornaments from different families -- was part of the PAS expansion's philosophy of recognizing the interconnected nature of rudimental technique. While the original NARD 26 kept flams and drags largely separate, the PAS expansion acknowledged that advanced players routinely combine these ornaments in performance. The flam drag codifies this practice as a standard rudiment, providing a formal vehicle for developing the ability to switch rapidly between different ornament types. It appears in advanced drum corps snare drum literature and competitive rudimental solos where timbral variety and ornamental contrast are desired.

---

## 7. Notation

### Stick Notation

```
>        >        >        >
lR  L  llR   |   rL  R  rrL
```

### Grid Visualization

```
Beat:    | 1     e           &         | ...   2&    a           3         |
Hand:    | (l)R  L     (ll)  R         | ...   (r)L  R     (rr)  L         |
Type:    |  g A  t      gg   A         |        g A  t      gg   A         |
Accent:  |    >              >         |           >              >         |
```

Grace notes shown in parentheses indicate they fall just before the grid position. Flam grace notes use a single parenthesized letter (offset -1/32 beat). Drag grace notes use a doubled parenthesized letter, with the first grace at -1/16 beat and the second at -1/32 beat before the primary. The first half spans beats 1-1& and the second half spans beats 2&-3.
