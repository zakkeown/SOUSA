# PAS #37: Drag Paradiddle #2

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 37 |
| **Name** | Drag Paradiddle #2 |
| **Category** | drag |
| **Family** | compound drag |
| **Composed Of** | Two Drags (#31) + two taps + diddle; extends Drag Paradiddle #1 (#36) by adding a second drag, creating a pattern of drag-tap-drag-tap-diddle |
| **Related** | #31 Drag (the primitive drag ornament used twice per group), #36 Drag Paradiddle #1 (simpler version with one drag), #35 Single Dragadiddle (simpler drag + diddle), #33 Double Drag Tap (two drags + tap, similar double-drag structure), #16 Single Paradiddle (underlying paradiddle sticking concept) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 5 beats |
| **Strokes Per Cycle** | 20 (12 primary strokes + 8 grace notes) |
| **Primary Strokes Per Cycle** | 12 |

Drag Paradiddle #2 extends Drag Paradiddle #1 by adding a second drag to each group. Where Drag Paradiddle #1 has one drag followed by a paradiddle (drag-tap-tap-diddle), Drag Paradiddle #2 has two alternating drags followed by a tap and diddle (drag-tap-drag-tap-diddle). Each group contains 6 primary strokes (two drag accents + two taps + one diddle pair) plus 4 grace notes (two pairs of drag graces). The pattern spans 5 beats for one full left-right alternation. The relationship between #36 and #37 mirrors the single-to-double progression found throughout the PAS rudiments: the "#2" version adds an extra drag before the paradiddle's diddle closure.

The critical difference from Drag Paradiddle #1 (#36): each group has TWO drags alternating with taps before closing with a diddle, creating a longer, more complex phrase. The critical difference from Double Drag Tap (#33): the diddle at the end replaces the simple tap closure, tying this pattern to the paradiddle family.

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/16 | (1st drag grace before 1) | L | g |
| 2 | -1/32 | (2nd drag grace before 1) | L | g |
| 3 | 0 | 1 | R | A |
| 4 | 1/4 | 1e | L | t |
| 5 | 1/2 - 1/16 | (1st drag grace before 1&) | R | g |
| 6 | 1/2 - 1/32 | (2nd drag grace before 1&) | R | g |
| 7 | 1/2 | 1& | L | A |
| 8 | 3/4 | 1a | R | t |
| 9 | 1 | 2 | L | d1 |
| 10 | 1 + 1/4 | 2e | L | d2 |
| 11 | 2 + 1/2 - 1/16 | (1st drag grace before 3&) | R | g |
| 12 | 2 + 1/2 - 1/32 | (2nd drag grace before 3&) | R | g |
| 13 | 2 + 1/2 | 3& | L | A |
| 14 | 2 + 3/4 | 3a | R | t |
| 15 | 3 - 1/16 | (1st drag grace before 4) | L | g |
| 16 | 3 - 1/32 | (2nd drag grace before 4) | L | g |
| 17 | 3 | 4 | R | A |
| 18 | 3 + 1/4 | 4e | L | t |
| 19 | 3 + 1/2 | 4& | R | d1 |
| 20 | 3 + 3/4 | 4a | R | d2 |

Note: Each group occupies 6 sixteenth-note positions for its primary strokes: drag accent (1), tap (2), drag accent (3), tap (4), diddle d1 (5), diddle d2 (6). Plus 4 grace notes for the two drags. The first group spans beats 1 through 2e (positions 1, 1e, 1&, 1a, 2, 2e). The second group spans beats 3& through 4a (positions 3&, 3a, 4, 4e, 4&, 4a). The remaining positions between groups serve as transition space. The diddle at the end forces the opposite hand to lead the next group.

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>        >                 |  >        >
llR  L   rrL  R   L L     |  rrL  R   llR  L   R R
```

Each group contains two alternating drags separated by taps, followed by a closing diddle. The first group: right-hand drag (llR), tap (L), left-hand drag (rrL), tap (R), left-hand diddle (LL). The second group mirrors: left-hand drag (rrL), tap (R), right-hand drag (llR), tap (L), right-hand diddle (RR).

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | drag grace 1 | #3 | First drag grace, first drag of group 1 |
| 2 | L | grace | drag grace 2 | #3 | Second drag grace, first drag of group 1 |
| 3 | R | accent | primary | n/a | Right-hand drag primary, first drag of group 1 |
| 4 | L | tap | none | n/a | Tap between the two drags |
| 5 | R | grace | drag grace 1 | #7 | First drag grace, second drag of group 1 |
| 6 | R | grace | drag grace 2 | #7 | Second drag grace, second drag of group 1 |
| 7 | L | accent | primary | n/a | Left-hand drag primary, second drag of group 1 |
| 8 | R | tap | none | n/a | Tap before the diddle |
| 9 | L | d1 | none | n/a | First stroke of left-hand diddle, closing group 1 |
| 10 | L | d2 | none | n/a | Second stroke of left-hand diddle, closing group 1 |
| 11 | R | grace | drag grace 1 | #13 | First drag grace, first drag of group 2 |
| 12 | R | grace | drag grace 2 | #13 | Second drag grace, first drag of group 2 |
| 13 | L | accent | primary | n/a | Left-hand drag primary, first drag of group 2 |
| 14 | R | tap | none | n/a | Tap between the two drags |
| 15 | L | grace | drag grace 1 | #17 | First drag grace, second drag of group 2 |
| 16 | L | grace | drag grace 2 | #17 | Second drag grace, second drag of group 2 |
| 17 | R | accent | primary | n/a | Right-hand drag primary, second drag of group 2 |
| 18 | L | tap | none | n/a | Tap before the diddle |
| 19 | R | d1 | none | n/a | First stroke of right-hand diddle, closing group 2 |
| 20 | R | d2 | none | n/a | Second stroke of right-hand diddle, closing group 2 |

### Ornament Timing

**Drag Grace Notes:**
- First grace offset from primary: **-1/16 beat** (the earlier of the two drag grace notes)
- Second grace offset from primary: **-1/32 beat** (the later of the two, closer to the primary)
- Both grace notes played by the **same hand** (opposite from the primary stroke)
- Each group contains two drags with their own grace note pairs:
  - Group 1: strokes #1-2 (LL) precede #3 (R); strokes #5-6 (RR) precede #7 (L)
  - Group 2: strokes #11-12 (RR) precede #13 (L); strokes #15-16 (LL) precede #17 (R)
- The drag-primary-to-drag-grace transition: after the first drag's accent (#3 R), the same hand must produce drag grace notes (#5-6 R) for the second drag. This is the same accent-to-grace challenge found in the double drag tap (#33), requiring the stick to be caught low after the accent and immediately initiate a double bounce. There is a tap (#4) on the opposite hand between the accent and the grace notes, providing a brief window for the transition.
- The diddle-to-drag-grace transition between groups (strokes #9-10 L diddle, then #11-12 R grace) crosses hands, so the grace hand has rested during the diddle.

**Diddle Timing:**
- Each diddle pair occupies one eighth-note duration (1/2 beat)
- Each individual diddle stroke occupies 1/4 beat (one sixteenth note)
- d1 and d2 fall on consecutive sixteenth-note positions
- At slow tempos: both strokes are distinct and clearly separated (open)
- At fast tempos: strokes closely spaced, nearly indistinguishable (closed)

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (drag primary) | 1.0 | Accented primary strokes for all four drags |
| tap | 0.65 - 0.77 | Unaccented single strokes (2 per group) |
| d1 | 0.65 - 0.77 | First diddle stroke at tap level |
| d2 | 0.90 - 0.98 x d1 | Slight decay on second bounce of diddle |
| grace (drag) | 0.45 - 0.65 | Each of the eight drag grace notes |

The Drag Paradiddle #2 has the same four-level dynamic hierarchy as the other drag paradiddle rudiments but with two accented drags per group rather than one. This creates a "strong-weak-strong-weak-taper" pattern (drag accent, tap, drag accent, tap, diddle) that gives the rudiment a more intense, driving character than the single-drag Drag Paradiddle #1.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>     -  >     -  - -   |  >     -  >     -  - -
llR   L  rrL   R  L L   |  rrL   R  llR   L  R R
```

Two accents per group (both on drag primaries), with the two taps and the diddle unaccented. This creates a "strong-weak-strong-weak-taper" grouping that gives the pattern its driving, pulsating character.

### Dynamic Contour

The Drag Paradiddle #2 produces a distinctive double-accented contour within each group: the first drag creates a "ruff-accent" peak, the tap drops to medium level, the second drag creates another "ruff-accent" peak, the second tap drops again, and the diddle tapers off with slight d2 decay. The two accented drags create paired impulses within each group, producing a more intense and rhythmically complex shape than the single-accent Drag Paradiddle #1. The double-drag aspect gives the pattern a resemblance to the double drag tap (#33), but the diddle closure instead of a simple tap gives it a paradiddle-family feel that resolves each group more smoothly. The alternating accent-tap pairs create a driving "push-pull" sensation that makes this one of the most rhythmically compelling compound drag rudiments.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace (drag 1) | tap (< 2") | fingers |
| 2 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 3 | R | accent | full (12"+) | wrist + arm |
| 4 | L | tap | low (2-6") | wrist |
| 5 | R | grace (drag 1) | tap (< 2") | fingers |
| 6 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 7 | L | accent | full (12"+) | wrist + arm |
| 8 | R | tap | low (2-6") | wrist |
| 9 | L | d1 | low (2-6") | wrist |
| 10 | L | d2 | tap (< 2") | fingers / rebound |
| 11 | R | grace (drag 1) | tap (< 2") | fingers |
| 12 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 13 | L | accent | full (12"+) | wrist + arm |
| 14 | R | tap | low (2-6") | wrist |
| 15 | L | grace (drag 1) | tap (< 2") | fingers |
| 16 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 17 | R | accent | full (12"+) | wrist + arm |
| 18 | L | tap | low (2-6") | wrist |
| 19 | R | d1 | low (2-6") | wrist |
| 20 | R | d2 | tap (< 2") | fingers / rebound |

The defining physical challenge of the Drag Paradiddle #2 is the accent-to-grace transition between the two drags within each group. Consider the right hand's path in the first group: it plays a full-height accent (#3), then must immediately produce two grace notes (#5-6) for the second drag while the left hand plays a tap (#4) in between. The accent at full height must be caught low (downstroke) so the stick is near the head when it needs to produce the drag grace double bounce. This is the same challenge found in the double drag tap (#33), but here it occurs within a paradiddle framework that adds the additional coordination demand of the closing diddle. The tap between the two drags (#4) provides a brief window (one sixteenth note) for the accent hand to reset to tap height.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for all four accented drag primaries. Grace notes individually articulated as open drags. Clear separation between all six primary strokes per group. Diddles are open with two distinct wrist strokes. The accent-to-grace transition is manageable due to wider spacing. The group structure is clearly audible: ruff-ACCENT-tap-ruff-ACCENT-tap-bounce-bounce. |
| Moderate (100-120 BPM) | Wrist-driven accents. Drag grace notes close up to standard ruff sound. The accent-to-grace transition tightens, requiring efficient downstroke technique. The two drag-tap pairs within each group begin to flow as a continuous phrase. Diddles shift to wrist-finger technique. |
| Fast (120-140 BPM) | Finger control for grace notes, taps, and diddle d2. Accents drop to half height class. Drag grace notes compress toward -1/32 and -1/64. The accent-to-grace transition becomes the critical bottleneck. Diddles become nearly closed. The rudiment feels like a driving stream of alternating accented drags with a diddle punctuation at the end of each phrase. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Drag paradiddle #2 without drags**: Practice as accent-tap-accent-tap-diddle (>R L >L R LL | >L R >R L RR) to isolate the rhythmic structure and accent pattern
- **Open drag paradiddle #2**: Exaggerated drag spacing for slow practice and ornament clarity
- **Single-drag preparation**: Practice with only the first drag (llR L L R LL | rrL R R L RR) before adding the second drag
- **Drag paradiddle #2 in triplets**: Some interpretations place the pattern in a triplet subdivision for a different rhythmic feel

### Prerequisites

- #31 Drag -- the basic drag ornament must be consistent on both hands
- #36 Drag Paradiddle #1 -- the simpler one-drag version should be comfortable before adding the second drag
- #33 Double Drag Tap -- develops the accent-to-grace transition between consecutive drags
- #16 Single Paradiddle -- the paradiddle sticking concept (accent-tap-diddle closure)
- #6 Double Stroke Open Roll -- controlled diddles at various tempos
- Downstroke control -- the ability to play a full accent and immediately catch the stick low for the following grace notes

### Builds Toward

- Application in advanced rudimental solos and drum corps literature
- Development of multi-drag coordination and endurance
- Preparation for improvised rudimental combinations using multiple drag ornaments within paradiddle frameworks

### Teaching Notes

Drag Paradiddle #2 is the most complex of the compound drag rudiments, combining two drags with a paradiddle closure in a 6-primary-stroke group. It builds directly on the skills developed in the Drag Paradiddle #1 (#36) by adding a second drag, and on the double drag tap (#33) by replacing the simple tap closure with a paradiddle diddle. The key to mastering this rudiment is the accent-to-grace transition between the two drags.

**Common mistakes:**
- Collapsing the second drag: at faster tempos, the second drag often loses one or both grace notes, reducing it to a flam or plain accent. Both drags must maintain full ruff quality.
- Uneven drag quality: the first and second drag within each group should be dynamically and rhythmically identical. Students often execute the first drag well but rush or compress the second.
- Rushing the diddle: the diddle at the end of each group is often cut short or rushed. It should occupy its full rhythmic value (two sixteenth notes).
- Losing the accent-to-grace transition: after the first drag's accent, the same hand must produce grace notes for the second drag. This transition is the most technically demanding aspect.
- Inconsistent spacing: the six primary strokes per group should be evenly spaced on the sixteenth-note grid.
- Confusing with double drag tap (#33): the Drag Paradiddle #2 closes with a diddle, not a single tap. The diddle provides a distinct "paradiddle" ending that differentiates it from the double drag tap.

**Practice strategies:**
- Start with the accent pattern without ornaments: >R L >L R LL | >L R >R L RR
- Practice the accent-to-grace transition in isolation: play a full accent, then immediately produce two grace notes on the same hand
- Build up from Drag Paradiddle #1: add the second drag to an already-comfortable #36 pattern
- Use a metronome and count carefully: "1 e & a 2 e" for the first group
- Practice each group separately (right-lead, then left-lead) before combining
- Compare the two drags within each group for equal ruff quality
- Record and listen for consistent accent placement and ornament clarity across both groups

**What to listen for:**
- Both drags in each group should have identical ruff quality
- Clear accent-tap dynamic contrast throughout
- Even sixteenth-note spacing of all six primary strokes per group
- Balanced, controlled diddle closing each group with natural d2 decay
- Both groups (right-lead and left-lead) should sound identical
- The "strong-weak-strong-weak-taper" phrasing should create a natural musical grouping
- Smooth transition between groups across the rest/pickup space

### Historical Context

Drag Paradiddle #2 was included in the original NARD 26 rudiments when NARD was established in 1933. Its origins trace back to early American and European military field drumming traditions, where progressive extensions of rudiment patterns were standard pedagogical practice. The relationship between Drag Paradiddle #1 (#36) and Drag Paradiddle #2 (#37) follows the same progressive pattern seen throughout rudimental drumming: the "#2" version extends the "#1" by adding complexity, just as the double paradiddle (#17) extends the single paradiddle (#16), and the double drag tap (#33) extends the single drag tap (#32). The rudiment was retained as PAS #37 in the 1984 expansion to 40 rudiments. In drum corps and rudimental solo literature, the Drag Paradiddle #2 appears less frequently than the #1 version due to its greater length and complexity, but when it does appear, it provides a powerful, driving rhythmic phrase that showcases the performer's control of both drag technique and paradiddle coordination.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>        >                      >        >
llR  L   rrL  R   L L      |   rrL  R   llR  L   R R
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1     e     &     a   | 2     e                       |
Hand:    | (ll)R L     (rr)L R   | L     L                       |
Type:    |  gg A t      gg A t   | d1    d2                      |
Accent:  |     >            >    |                               |

Beat:    |                   3&    3a  | 4     e     &     a       |
Hand:    |                   (rr)L R   | (ll)R L     R     R       |
Type:    |                    gg A t   |  gg A t     d1    d2      |
Accent:  |                       >    |      >                    |
```

Grace notes shown in doubled parentheses indicate two same-hand grace notes (drag) falling just before the grid position. The first grace is at -1/16 beat and the second at -1/32 beat from the primary. The first group occupies beats 1 through 2e (6 sixteenth-note positions for primary strokes). The second group occupies beats 3& through 4a (6 sixteenth-note positions for primary strokes). The space between groups (beats 2& through 3&) serves as transition/rest space.
