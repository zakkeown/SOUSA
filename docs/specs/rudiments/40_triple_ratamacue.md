# PAS #40: Triple Ratamacue

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 40 |
| **Name** | Triple Ratamacue |
| **Category** | drag |
| **Family** | ratamacue |
| **Composed Of** | Three drags (#31) + three single strokes ending with an accent; the double ratamacue (#39) with an additional drag prepended |
| **Related** | #38 Single Ratamacue (one drag + singles + accent), #39 Double Ratamacue (two drags + singles + accent), #31 Drag (the primitive drag ornament), #18 Triple Paradiddle (analogous "triple" extension in the diddle family) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 4 beats |
| **Strokes Per Cycle** | 24 (12 primary strokes + 12 grace notes) |
| **Primary Strokes Per Cycle** | 12 |

The triple ratamacue is the final rudiment in the PAS 40 International Drum Rudiments list and the most extended member of the ratamacue family. It adds a third drag to the beginning of the double ratamacue (#39), creating a pattern with three consecutive drags followed by three single strokes ending with an accent -- six primary strokes per group. Each group spans 1.5 beats (6 sixteenth notes), and two groups fill the 4-beat cycle (occupying 3 beats of primary strokes, with the remaining beat filled by grace note timing and the spacing between groups). The three consecutive drags create the longest ruff sequence in the ratamacue family, producing a dramatic, rolling buildup that resolves emphatically on the closing accent. As PAS #40, it serves as the culmination of the entire rudiment system.

The six primary strokes per group fall on consecutive sixteenth-note positions: three drag primaries (each preceded by grace notes), then three single strokes with the accent on the last. The groups alternate lead hands.

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/16 | (1st grace before 1) | L | g |
| 2 | -1/32 | (2nd grace before 1) | L | g |
| 3 | 0 | 1 | R | t |
| 4 | 1/4 - 1/16 | (1st grace before 1e) | R | g |
| 5 | 1/4 - 1/32 | (2nd grace before 1e) | R | g |
| 6 | 1/4 | 1e | L | t |
| 7 | 1/2 - 1/16 | (1st grace before 1&) | L | g |
| 8 | 1/2 - 1/32 | (2nd grace before 1&) | L | g |
| 9 | 1/2 | 1& | R | t |
| 10 | 3/4 | 1a | L | t |
| 11 | 1 | 2 | R | t |
| 12 | 1 + 1/4 | 2e | L | A |
| 13 | 1 + 1/2 - 1/16 | (1st grace before 2&) | R | g |
| 14 | 1 + 1/2 - 1/32 | (2nd grace before 2&) | R | g |
| 15 | 1 + 1/2 | 2& | L | t |
| 16 | 1 + 3/4 - 1/16 | (1st grace before 2a) | L | g |
| 17 | 1 + 3/4 - 1/32 | (2nd grace before 2a) | L | g |
| 18 | 1 + 3/4 | 2a | R | t |
| 19 | 2 - 1/16 | (1st grace before 3) | R | g |
| 20 | 2 - 1/32 | (2nd grace before 3) | R | g |
| 21 | 2 | 3 | L | t |
| 22 | 2 + 1/4 | 3e | R | t |
| 23 | 2 + 1/2 | 3& | L | t |
| 24 | 2 + 3/4 | 3a | R | A |

Note: The first group (strokes #1-12) spans from beat 1 through 2e: three drags on 1, 1e, and 1&, then taps on 1a, 2, and accent on 2e. The second group (strokes #13-24) spans from beat 2& through 3a: three drags on 2&, 2a, and 3, then taps on 3e, 3&, and accent on 3a. The lead hand alternates between groups.

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
                              >  |                                >
llR  rrL  llR  L  R  L       |  rrL  llR  rrL  R  L  R
```

Each group consists of three drags (alternating the lead hand for each successive drag) followed by three single strokes with the accent on the last. The first group begins with a right-hand drag primary, the second with a left-hand drag primary. See [#38 Single Ratamacue](./38_single_ratamacue.md) for the shared "cue" accent pattern that concludes each group.

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | drag grace 1 | #3 | First grace, first drag of group 1 |
| 2 | L | grace | drag grace 2 | #3 | Second grace, first drag of group 1 |
| 3 | R | tap | primary | n/a | First drag primary (unaccented), group 1 |
| 4 | R | grace | drag grace 1 | #6 | First grace, second drag of group 1 |
| 5 | R | grace | drag grace 2 | #6 | Second grace, second drag of group 1 |
| 6 | L | tap | primary | n/a | Second drag primary (unaccented), group 1 |
| 7 | L | grace | drag grace 1 | #9 | First grace, third drag of group 1 |
| 8 | L | grace | drag grace 2 | #9 | Second grace, third drag of group 1 |
| 9 | R | tap | primary | n/a | Third drag primary (unaccented), group 1 |
| 10 | L | tap | none | n/a | First single stroke; the "ta" |
| 11 | R | tap | none | n/a | Second single stroke; the "ma" |
| 12 | L | accent | none | n/a | Closing accent; the "cue" |
| 13 | R | grace | drag grace 1 | #15 | First grace, first drag of group 2 |
| 14 | R | grace | drag grace 2 | #15 | Second grace, first drag of group 2 |
| 15 | L | tap | primary | n/a | First drag primary (unaccented), group 2 |
| 16 | L | grace | drag grace 1 | #18 | First grace, second drag of group 2 |
| 17 | L | grace | drag grace 2 | #18 | Second grace, second drag of group 2 |
| 18 | R | tap | primary | n/a | Second drag primary (unaccented), group 2 |
| 19 | R | grace | drag grace 1 | #21 | First grace, third drag of group 2 |
| 20 | R | grace | drag grace 2 | #21 | Second grace, third drag of group 2 |
| 21 | L | tap | primary | n/a | Third drag primary (unaccented), group 2 |
| 22 | R | tap | none | n/a | First single stroke; the "ta" |
| 23 | L | tap | none | n/a | Second single stroke; the "ma" |
| 24 | R | accent | none | n/a | Closing accent; the "cue" |

### Ornament Timing

**Drag Grace Notes:**
- First grace offset from primary: **-1/16 beat** (the earlier of the two drag grace notes)
- Second grace offset from primary: **-1/32 beat** (the later of the two, closer to the primary)
- Both grace notes played by the **same hand** (opposite from the primary stroke)
- Each drag follows the standard ornament timing from #31 Drag
- There are six drags per cycle, each with its own pair of grace notes:
  - Group 1: Strokes #1-2 (L) precede #3 (R); strokes #4-5 (R) precede #6 (L); strokes #7-8 (L) precede #9 (R)
  - Group 2: Strokes #13-14 (R) precede #15 (L); strokes #16-17 (L) precede #18 (R); strokes #19-20 (R) precede #21 (L)
- Critical transitions: Between each consecutive drag, the hand that just played a drag primary must immediately produce two grace notes for the next drag. There are two such transitions per group (first-to-second and second-to-third drag), making this the most technically demanding pattern in the ratamacue family.
- As in all ratamacues, the drag primaries are **taps** (unaccented). The accent is reserved for the final stroke of each group.

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent ("cue") | 1.0 | Closing accent, the loudest stroke in each group |
| tap (drag primaries + singles) | 0.65 - 0.77 | All drag primaries and middle taps are unaccented |
| grace (drag) | 0.45 - 0.65 | Each of the twelve drag grace notes |

The same three-level dynamic hierarchy as the single (#38) and double (#39) ratamacues. The three consecutive drags at the beginning of each group create the longest soft-to-medium preparation in the family, making the closing accent feel maximally emphatic.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
-     -     -     -  -  >  |  -     -     -     -  -  >
llR   rrL   llR   L  R  L  |  rrL   llR   rrL   R  L  R
```

One accent per group (on the final stroke), with all five preceding primary strokes unaccented. The accent placement creates the same off-beat emphasis characteristic of the ratamacue family.

### Dynamic Contour

The triple ratamacue produces the most dramatic "build-and-release" contour in the ratamacue family. Each group begins with three consecutive drag-preceded taps (six grace notes and three primaries), creating a sustained buzzing preparation that accumulates rhythmic energy. The pattern then passes through two more taps before resolving on a full accent. This extended ruff-ruff-ruff-tap-tap-ACCENT shape creates a powerful sense of forward motion and dramatic anticipation. The three drags function like a drum roll building to a climax, and the closing accent provides a satisfying release. As the final rudiment in the PAS system, the triple ratamacue represents the most elaborate development of the drag-ornament concept: the maximum number of drags before the ratamacue resolution.

---

## 5. Physical / Kinesthetic

### Stroke Map

<!-- Height class and motion type for each stroke position. See _conventions.md for definitions. -->

| # | Hand | Stroke Type | Height Class | Motion Type |
|---|------|-------------|-------------|-------------|
| 1 | L | grace (drag 1) | tap (< 2") | fingers |
| 2 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 3 | R | tap (drag primary) | low (2-6") | wrist |
| 4 | R | grace (drag 1) | tap (< 2") | fingers |
| 5 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 6 | L | tap (drag primary) | low (2-6") | wrist |
| 7 | L | grace (drag 1) | tap (< 2") | fingers |
| 8 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 9 | R | tap (drag primary) | low (2-6") | wrist |
| 10 | L | tap | low (2-6") | wrist |
| 11 | R | tap | low (2-6") | wrist |
| 12 | L | accent | full (12"+) | wrist + arm |
| 13 | R | grace (drag 1) | tap (< 2") | fingers |
| 14 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 15 | L | tap (drag primary) | low (2-6") | wrist |
| 16 | L | grace (drag 1) | tap (< 2") | fingers |
| 17 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 18 | R | tap (drag primary) | low (2-6") | wrist |
| 19 | R | grace (drag 1) | tap (< 2") | fingers |
| 20 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 21 | L | tap (drag primary) | low (2-6") | wrist |
| 22 | R | tap | low (2-6") | wrist |
| 23 | L | tap | low (2-6") | wrist |
| 24 | R | accent | full (12"+) | wrist + arm |

The triple ratamacue contains two drag-primary-to-drag-grace transitions per group (between the first and second drags, and between the second and third drags), making it the most physically demanding pattern in the ratamacue family. Each transition requires the hand that just played a tap-height drag primary to immediately produce two finger-controlled grace notes for the next drag. The three consecutive drags demand sustained finger control and bounce technique, as the non-lead hand must alternate between receiving drag primaries and producing grace notes in rapid succession. Consider the left hand in group 1: it plays grace notes (#1-2), then a drag primary tap (#6), then grace notes again (#7-8), then a tap (#10), then the closing accent (#12). This alternating grace-primary-grace-tap-accent path requires precise control of stick height transitions.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for accents. Grace notes individually articulated as open drags. All six primary strokes per group are clearly separated. The three consecutive drags are audible as three distinct ruff events. The two drag-primary-to-grace transitions within each group are manageable. The pattern's six-note grouping is clearly perceptible. |
| Moderate (100-120 BPM) | Wrist-driven strokes. Drag grace notes close up to standard ruff sound. The three drags flow together as a connected series of ruffs, creating a sustained buzzing texture. Standard grace offsets at -1/16 and -1/32. The transitions between consecutive drags tighten, requiring efficient bounce technique. The six primary strokes and accent flow as a connected group. |
| Fast (120-140 BPM) | Finger control for grace notes and taps. Accent drops to half height class. Drag grace notes compress toward -1/32 and -1/64 offsets. The three drags blur into a continuous buzzing run that propels toward the accent. The drag-primary-to-grace transitions become the critical bottleneck -- each hand must convert impact energy into controlled bounces almost instantaneously. The pattern approaches a rapid stream of sixteenths with an embedded buzz texture and off-beat accent. |

---

## 6. Variations & Pedagogy

### Common Variations

- **4/4 common time grouping**: The triple ratamacue fits naturally into 4/4 time with its 4-beat cycle, making it a natural fit for common time musical contexts.
- **No-ornament preparation**: Practice as six alternating single strokes with the accent on the last (R L R L R >L | L R L R L >R) to isolate the rhythm before adding drags.
- **Progressive ratamacue exercise**: Playing single, double, then triple ratamacues in sequence, adding one drag at a time.
- **Continuous triple ratamacues**: Linking multiple patterns for extended passages in solo and ensemble literature.

### Prerequisites

- #38 Single Ratamacue -- the foundational ratamacue accent pattern
- #39 Double Ratamacue -- the two-drag version should be comfortable before adding a third drag
- #31 Drag -- consistent drag ornament on both hands
- #33 Double Drag Tap -- the drag-primary-to-drag-grace transition technique
- #6 Double Stroke Open Roll -- controlled double bounce technique
- Strong finger control and bounce technique for sustained consecutive drags

### Builds Toward

- The triple ratamacue is PAS #40, the final rudiment in the system. It does not directly build toward other PAS rudiments, but it develops:
- Advanced rudimental solo technique
- Extended drag-based passages in drum corps and marching percussion
- The sustained consecutive-drag skill applicable to any complex drag combination
- Musical phrasing and dynamic contour control

### Teaching Notes

The triple ratamacue is the most extended pattern in the ratamacue family and the final rudiment in the PAS 40 system. It adds a third drag to the double ratamacue, creating a pattern with three consecutive drags before the single-stroke resolution. The primary technical challenge is sustaining three consecutive drags with clean grace notes and even primary strokes. See [#38 Single Ratamacue](./38_single_ratamacue.md) for foundational teaching guidance on the ratamacue accent pattern and dynamic contour.

**Common mistakes:**
- Accenting the drag primaries: As in all ratamacues, the drag primaries are taps. The accent belongs only on the final stroke.
- Progressive drag degradation: The first drag is clean, the second is acceptable, and the third loses grace notes. All three drags must have identical quality.
- Rushing through the drags: Students often compress the three-drag section, arriving at the single strokes too early. Even sixteenth-note spacing must be maintained.
- Losing rhythmic orientation: With six primary strokes per group plus twelve grace notes per cycle, students can lose their place. Counting the sixteenth-note grid is essential.
- Fatigue: The sustained consecutive drags require endurance in finger control. Students may tire and lose control of the grace notes.
- Inconsistent drag quality across the six drags per cycle.

**Practice strategies:**
- Master the single (#38) and double (#39) ratamacues before attempting the triple.
- Build up incrementally: play the single ratamacue, then the double, then the triple, feeling how each additional drag extends the pattern.
- Practice three consecutive drags in isolation (llR rrL llR, then rrL llR rrL) to develop the sustained bounce technique.
- Count the primary strokes on the sixteenth-note grid to maintain even spacing.
- Practice each group separately (right-lead, then left-lead) before alternating.
- Compare all six drags per cycle for equal quality.
- Use slow tempos to establish clean grace notes on every drag before increasing speed.

**What to listen for:**
- The extended "build-and-release" contour: three ruffs leading through taps to a climactic accent.
- All six drags per cycle should have identical ruff quality -- no progressive degradation.
- Clear dynamic separation between the taps and the closing accent.
- Even sixteenth-note spacing across all six primary strokes per group.
- Both groups (right-lead and left-lead) should sound identical.
- A dramatic, rolling quality during the three-drag section that builds momentum toward the accent.

### Historical Context

The triple ratamacue was included in the original NARD 26 rudiments and has been practiced in American rudimental drumming traditions since at least the early 19th century. As PAS #40, it occupies the final position in the Percussive Arts Society's International Drum Rudiments list, a placement that reflects its status as the most elaborate development of the drag-ornament concept. The progression from single (#38) to double (#39) to triple (#40) ratamacue follows the same graduated extension principle seen throughout the PAS system (single/double/triple paradiddles, single/double drag taps), and placing the triple ratamacue last gives the list a sense of culmination. The rudiment appears in advanced drum corps snare drum literature, rudimental contest solos, and military drumming contexts where its extended buildup and dramatic accent resolution provide a powerful musical effect. In the rudimental tradition, the triple ratamacue is often considered a capstone skill -- mastery of this pattern demonstrates command of drag technique, hand alternation, accent placement, and the sustained coordination required for advanced rudimental playing.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
                                    >  |                                      >
llR   rrL   llR   L   R   L        |  rrL   llR   rrL   R   L   R
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1        e        &        a   | 2        e        &        a        | 3        e        &     a   |  4  |
Hand:    | (ll)R    (rr)L    (ll)R    L   | R        L        (rr)L    (ll)R    | (rr)L    R        L     R   |     |
Type:    |  gg  t    gg  t    gg  t   t   | t        A         gg  t    gg  t   |  gg  t   t        t     A   |     |
Accent:  |                                |          >                          |                         >   |     |
```

Grace notes shown in doubled parentheses indicate two same-hand grace notes falling just before the grid position. The first grace is at -1/16 beat and the second at -1/32 beat from the primary. The first group occupies beats 1 through 2e (drags on 1, 1e, and 1&; taps on 1a and 2; accent on 2e). The second group occupies beats 2& through 3a (drags on 2&, 2a, and 3; taps on 3e and 3&; accent on 3a). Beat 4 is empty, completing the 4-beat cycle.
