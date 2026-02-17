# PAS #31: Drag (Ruff)

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 31 |
| **Name** | Drag (Ruff) |
| **Category** | drag |
| **Family** | basic drag |
| **Composed Of** | None -- this is the primitive drag ornament, two grace notes (same hand) preceding a primary stroke |
| **Related** | #20 Flam (single grace note counterpart), #32 Single Drag Tap (drag + tap), #33 Double Drag Tap (two drags + tap), #30 Flam Drag (flam combined with drag) |
| **NARD Original** | Yes (as "Ruff") |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | eighth |
| **Cycle Length** | 1 beat |
| **Strokes Per Cycle** | 6 (4 grace notes + 2 primary strokes) |
| **Primary Strokes Per Cycle** | 2 |

The drag is the foundational ornament of the drag category, playing the same structural role for drag rudiments (PAS #31-40) that the flam (#20) plays for flam rudiments (PAS #20-30). Where the flam uses a single grace note on the opposite hand, the drag uses two grace notes on the same hand (opposite from the primary). The two grace notes form a quick double-bounce or controlled diddle, producing a brief "buzz" or "ruff" leading into the primary stroke. The pattern alternates between right-hand and left-hand drags on the eighth-note grid.

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/16 | (1st grace before 1) | L | g |
| 2 | -1/32 | (2nd grace before 1) | L | g |
| 3 | 0 | 1 | R | A |
| 4 | 1/2 - 1/16 | (1st grace before 1&) | R | g |
| 5 | 1/2 - 1/32 | (2nd grace before 1&) | R | g |
| 6 | 1/2 | 1& | L | A |

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>          >
llR        rrL
```

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | drag grace 1 | #3 | First grace note (earlier), same hand as #2 |
| 2 | L | grace | drag grace 2 | #3 | Second grace note (closer to primary), same hand as #1 |
| 3 | R | accent | primary | n/a | Right-hand drag primary stroke |
| 4 | R | grace | drag grace 1 | #6 | First grace note (earlier), same hand as #5 |
| 5 | R | grace | drag grace 2 | #6 | Second grace note (closer to primary), same hand as #4 |
| 6 | L | accent | primary | n/a | Left-hand drag primary stroke |

### Ornament Timing

**Drag Grace Notes:**
- First grace offset from primary: **-1/16 beat** (the earlier of the two drag grace notes)
- Second grace offset from primary: **-1/32 beat** (the later of the two, closer to the primary)
- Both grace notes played by the **same hand** (opposite from the primary stroke)
- Strokes #1-2 (both L) precede stroke #3 (R primary); strokes #4-5 (both R) precede stroke #6 (L primary)
- The two grace notes form a quick diddle (same-hand double bounce) leading into the primary
- The grace notes should sound like a brief "buzz" or "ruff" before the primary, not two distinct taps
- At moderate tempos the inter-grace interval (1/32 beat) is shorter than the second-grace-to-primary interval (also 1/32 beat), producing an accelerating onset into the primary
- Key physical difference from the flam (#20): the flam grace is played by the opposite hand (an offset from simultaneous two-hand motion), while the drag grace notes are played by the same hand (a controlled double bounce)

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (primary) | 1.0 | Full stroke, the dominant sound of the drag |
| grace (drag) | 0.45 - 0.65 | Each of the two drag grace notes; softer than flam grace notes |

The dynamic contrast between grace notes and primary is fundamental to the drag's character. If the grace notes are too loud, the ornament sounds like a triple stroke or triplet rather than a ruff. If the grace notes are too quiet, the buzz effect before the primary is lost and the drag sounds like a plain accent. The two grace notes should be at approximately equal velocity to each other, though the second grace may be very slightly louder due to the natural rebound mechanics of the controlled double bounce.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>         >
llR       rrL
```

Both primary strokes are accented. The grace notes are inherently soft and do not receive accent treatment.

### Dynamic Contour

The drag has a symmetrical dynamic contour: each half-beat contains one "buzz-loud" pair (grace-grace-primary). The pattern is identical on both halves but alternates between hands. The dynamic shape is best described as two matched impulses per beat, each preceded by a soft double-bounce onset. Unlike the flam's single-grace onset, the drag's two-grace onset creates a brief crescendo effect (soft-soft-loud) that gives the primary a sense of forward momentum. There is no crescendo or decrescendo across the cycle; the pattern repeats identically.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace (drag 1) | tap (< 2") | fingers |
| 2 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 3 | R | accent | full (12"+) | wrist + arm |
| 4 | R | grace (drag 1) | tap (< 2") | fingers |
| 5 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 6 | L | accent | full (12"+) | wrist + arm |

The physical essence of the drag is a controlled double bounce with the non-primary hand while the primary hand executes a full downstroke. Unlike the flam, where two hands begin their motion simultaneously from different heights, the drag requires one hand to produce two rapid, quiet strokes (a bounced diddle at tap height) while the other hand prepares for a full accent. The first grace note is an active finger stroke; the second grace note benefits from the rebound of the first, making it partially passive. This bounce-and-catch technique is the defining physical skill of all drag-category rudiments.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Wide separation between grace notes and primary. Both grace notes are individually articulated as two distinct soft strokes ("open drag"). The double bounce is controlled with active finger strokes for both graces. The first grace at -1/16 and second at -1/32 are clearly distinguishable. Full wrist + arm motion for primary. |
| Moderate (100-140 BPM) | Standard drag spacing. Grace notes begin to close up, with the second grace relying more on rebound from the first. The two graces sound more like a quick buzz than two separate strokes. Grace offsets at -1/16 and -1/32 from primary. Primary uses wrist motion. This is the "standard closed drag" sound. |
| Fast (140-160 BPM) | Tight drag spacing. Grace notes compress toward -1/32 and -1/64 offsets. The double bounce becomes nearly a single buzz. The primary drops to half height class. The grace notes are almost entirely rebound-driven, with only the first requiring an active finger initiation. The drag approaches a "crushed" sound where the ruff and primary are barely distinguishable as separate events. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Open drag**: Exaggerated spacing between grace notes and primary for stylistic effect or slow tempos. Both grace notes are clearly audible as distinct strokes.
- **Closed drag**: Very tight grace-to-primary spacing, used at fast tempos. The two graces blur into a single buzz before the primary.
- **Three-stroke ruff**: In orchestral contexts, a "ruff" may include three grace notes rather than two. This is distinct from the PAS drag, which specifically uses two grace notes.
- **Unison drag**: Both grace notes and primary played by the same hand (a triple stroke). This is a practice variation, not the standard rudiment.
- **Drag with tap primary**: Playing the primary as a tap (unaccented) rather than an accent for dynamic variation within musical contexts.

### Prerequisites

- #1 Single Stroke Roll -- basic hand alternation and stroke control
- #6 Double Stroke Open Roll -- the double bounce (diddle) technique is the physical foundation for the drag grace notes
- Rebound control -- ability to produce two rapid, quiet bounces at tap height

### Builds Toward

- #32 Single Drag Tap (drag + tap pattern)
- #33 Double Drag Tap (two drags + tap)
- #34 Lesson 25 (drag + single stroke combination)
- #35 Single Dragadiddle (drag + paradiddle)
- #36 Drag Paradiddle #1 (drag + paradiddle variation)
- #37 Drag Paradiddle #2 (drag + paradiddle variation)
- #38 Single Ratamacue (drag + triplet + accent)
- #39 Double Ratamacue (two drags + triplet + accent)
- #40 Triple Ratamacue (three drags + triplet + accent)
- #30 Flam Drag (flam combined with drag, cross-category)
- All drag-category rudiments (PAS #31-40) build directly on this primitive

### Teaching Notes

The drag is the most fundamental ornament of the drag category and the basis of all drag rudiments (PAS #31-40). It is the double-grace-note counterpart of the flam (#20), and its mastery is essential before progressing to any compound drag rudiment. Despite its apparent simplicity, producing consistent, well-controlled drags requires a specific technique (the controlled double bounce) that differs significantly from the flam technique.

**Common mistakes:**
- Grace notes too loud: The double bounce overpowers the primary, producing a triplet feel rather than an ornamental ruff
- Grace notes too quiet or absent: The stick is too low or the bounce too weak to produce audible grace notes, making the drag indistinguishable from a plain accent
- Uneven grace notes: The first grace is significantly louder or softer than the second, producing an unbalanced ruff. Both grace notes should be approximately equal in volume
- Single grace instead of double: Only one grace note sounds, converting the drag into a flam. This is especially common at fast tempos where the double bounce collapses
- Grace notes on the wrong hand: The drag grace notes must be played by the same hand (opposite from the primary). Playing them on the primary hand converts the pattern to a triple stroke
- Too much spacing between graces: The two grace notes should form a quick, connected double bounce, not two isolated taps with a gap between them
- Inconsistent spacing between hands: Left-hand drags (rrL) should sound identical to right-hand drags (llR)

**Practice strategies:**
- Start with the double bounce: practice tap-height double strokes (diddles) with each hand to develop the controlled bounce
- Play the two grace notes at mezzo-piano, then add the accented primary on the opposite hand
- Practice each hand's drag separately (all llR, then all rrL) before alternating
- Use a metronome and place the primary on the click; the grace notes should fall naturally just before
- Compare the sound of right-hand drags (llR) and left-hand drags (rrL) and work to match them
- Practice the drag at very slow tempos as three distinct strokes (open drag), then gradually close the spacing
- Record and listen back, checking that both hands produce identical drag quality

**What to listen for:**
- Each drag should sound like a brief "ruff" or buzz leading into the primary, not three separate notes
- Left-hand and right-hand drags should be indistinguishable in quality
- Consistent spacing from drag to drag
- Clear dynamic separation between grace notes (soft) and primary (full)
- The two grace notes should be approximately equal in volume and blend together as a unified ornament

### Historical Context

The drag (also known as the "ruff" in orchestral terminology) is one of the oldest drum ornaments, with roots predating formal rudiment codification. The term "drag" refers to the dragging or bouncing of the stick to produce two rapid grace notes before the primary stroke. The drag was included in the original NARD (National Association of Rudimental Drummers) standard 26 rudiments as the "Ruff" when NARD was founded in 1933, and its origins trace back to European military field drumming traditions. In Charles Stewart Ashworth's *A New, Useful and Complete System of Drum Beating* (1812), ruff patterns are among the foundational ornaments. The Percussive Arts Society retained it as PAS #31 (first of the drag rudiments) when the list was expanded to 40 in 1984. The drag serves as the primitive building block for all 10 drag-category rudiments (PAS #31-40), just as the flam (#20) is the primitive for all 11 flam-category rudiments (PAS #20-30). In orchestral percussion, the "ruff" (typically notated as three or more grace notes) is a closely related ornament, though the PAS drag specifically defines exactly two grace notes.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>              >
llR            rrL
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1              &            |
Hand:    | (ll)R          (rr)L        |
Type:    |  gg  A          gg  A       |
Accent:  |      >              >       |
```

Grace notes shown in doubled parentheses indicate two same-hand grace notes falling just before the grid position. The first grace is at -1/16 beat and the second at -1/32 beat from the primary. The primary strokes land on the grid positions 1 and &.
