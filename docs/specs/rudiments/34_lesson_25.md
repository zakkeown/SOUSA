# PAS #34: Lesson 25

> See [Conventions & Shared Definitions](./_conventions.md) for stroke types, velocity ratios, height classes, motion types, and notation rules.

---

## 1. Identity

| Field | Value |
|-------|-------|
| **PAS Number** | 34 |
| **Name** | Lesson 25 |
| **Category** | drag |
| **Family** | compound drag |
| **Composed Of** | Two Drags (#31) + one Flam (#20); a cross-category rudiment combining drag and flam ornaments in a six-note phrase (three primary strokes per group) |
| **Related** | #31 Drag (the primitive drag ornament used twice per group), #20 Flam (the primitive flam ornament closing each group), #32 Single Drag Tap (simpler drag + tap), #33 Double Drag Tap (two drags + tap, similar multi-drag structure), #30 Flam Drag (another cross-category rudiment combining flam and drag) |
| **NARD Original** | Yes |

---

## 2. Rhythmic Structure

| Field | Value |
|-------|-------|
| **Time Signature** | 4/4 |
| **Base Subdivision** | sixteenth |
| **Cycle Length** | 4 beats |
| **Strokes Per Cycle** | 16 (6 primary strokes + 10 grace notes) |
| **Primary Strokes Per Cycle** | 6 |

Lesson 25 is one of the most distinctive rudiments in the PAS 40, notable for being a cross-category rudiment that incorporates both drag and flam ornaments. Each group of three primary strokes spans 2 beats: two consecutive drags (each an accented primary preceded by two same-hand grace notes) followed by a flam (an accented primary preceded by a single opposite-hand grace note). The pattern spans 4 beats for one full left-right alternation. The three primary strokes within each group fall on the sixteenth-note grid (positions 1, 1e, 1&, then 3, 3e, 3&), creating a three-note phrase of alternating sixteenth notes. Lesson 25 is unique among the PAS 40 drag rudiments in that it contains a flam, making it a hybrid drag-flam rudiment.

### Stroke Grid

<!-- Each stroke's position as a beat fraction. Position 0 = downbeat of the cycle. Grace notes at negative offset from their primary. -->

| # | Position (beats) | Grid Slot | Hand | Type |
|---|------------------|-----------|------|------|
| 1 | -1/16 | (1st drag grace before 1) | L | g |
| 2 | -1/32 | (2nd drag grace before 1) | L | g |
| 3 | 0 | 1 | R | A |
| 4 | 1/4 - 1/16 | (1st drag grace before 1e) | R | g |
| 5 | 1/4 - 1/32 | (2nd drag grace before 1e) | R | g |
| 6 | 1/4 | 1e | L | A |
| 7 | 1/2 - 1/32 | (flam grace before 1&) | R | g |
| 8 | 1/2 | 1& | L | A |
| 9 | 2 - 1/16 | (1st drag grace before 3) | R | g |
| 10 | 2 - 1/32 | (2nd drag grace before 3) | R | g |
| 11 | 2 | 3 | L | A |
| 12 | 2 + 1/4 - 1/16 | (1st drag grace before 3e) | L | g |
| 13 | 2 + 1/4 - 1/32 | (2nd drag grace before 3e) | L | g |
| 14 | 2 + 1/4 | 3e | R | A |
| 15 | 2 + 1/2 - 1/32 | (flam grace before 3&) | L | g |
| 16 | 2 + 1/2 | 3& | R | A |

Note: Each group places three primary strokes on consecutive sixteenth-note positions. The first two primaries are preceded by drag grace note pairs (two same-hand grace notes each), and the third primary is preceded by a single flam grace note (opposite hand). The second group (strokes #9-16) mirrors the first with opposite hands. The remaining beats (2 and 4) serve as rests or pickup space between groups.

---

## 3. Sticking & Articulation

### Sticking Sequence

<!-- Write the full sticking using R/L with accent markers (>) and ornament prefixes. Use | to separate beats or groups. -->

```
>        >        >     |  >        >        >
llR      rrL      rL    |  rrL      llR      lR
```

Each group contains two drags followed by one flam, then the pattern mirrors with the opposite hand leading. The first group has: right-hand drag (llR), left-hand drag (rrL), left-hand flam (rL). The second group has: left-hand drag (rrL), right-hand drag (llR), right-hand flam (lR).

### Stroke Detail Table

<!-- One row per stroke in the cycle. For ornaments, specify the parent stroke. -->

| # | Hand | Stroke Type | Ornament Role | Parent Stroke | Notes |
|---|------|-------------|---------------|---------------|-------|
| 1 | L | grace | drag grace 1 | #3 | First drag grace, first drag of group 1 |
| 2 | L | grace | drag grace 2 | #3 | Second drag grace, first drag of group 1 |
| 3 | R | accent | primary | n/a | Right-hand drag primary, first drag |
| 4 | R | grace | drag grace 1 | #6 | First drag grace, second drag of group 1 |
| 5 | R | grace | drag grace 2 | #6 | Second drag grace, second drag of group 1 |
| 6 | L | accent | primary | n/a | Left-hand drag primary, second drag |
| 7 | R | grace | flam grace | #8 | Flam grace note closing group 1 |
| 8 | L | accent | primary | n/a | Left-hand flam primary, closing group 1 |
| 9 | R | grace | drag grace 1 | #11 | First drag grace, first drag of group 2 |
| 10 | R | grace | drag grace 2 | #11 | Second drag grace, first drag of group 2 |
| 11 | L | accent | primary | n/a | Left-hand drag primary, first drag |
| 12 | L | grace | drag grace 1 | #14 | First drag grace, second drag of group 2 |
| 13 | L | grace | drag grace 2 | #14 | Second drag grace, second drag of group 2 |
| 14 | R | accent | primary | n/a | Right-hand drag primary, second drag |
| 15 | L | grace | flam grace | #16 | Flam grace note closing group 2 |
| 16 | R | accent | primary | n/a | Right-hand flam primary, closing group 2 |

### Ornament Timing

**Drag Grace Notes:**
- First grace offset from primary: **-1/16 beat** (the earlier of the two drag grace notes)
- Second grace offset from primary: **-1/32 beat** (the later of the two, closer to the primary)
- Both grace notes played by the **same hand** (opposite from the primary stroke)
- Each group contains two drags: strokes #1-2 (LL) precede #3 (R); strokes #4-5 (RR) precede #6 (L); strokes #9-10 (RR) precede #11 (L); strokes #12-13 (LL) precede #14 (R)
- The drag-primary-to-drag-grace transition within each group (e.g., #3 R accent to #4-5 R grace) requires the same hand that just played a full drag primary to immediately produce two grace notes. This is the same technical challenge found in #33 Double Drag Tap.

**Flam Grace Notes:**
- Grace offset from primary: **-1/32 beat** (standard at moderate tempos)
- Allowable range: **-1/64 to -1/16** depending on tempo and style
- Grace hand: always the **opposite hand** from the primary stroke
- Stroke #7 (R grace) precedes stroke #8 (L primary); stroke #15 (L grace) precedes stroke #16 (R primary)
- The flam grace arrives from the hand that just completed two drag grace notes (e.g., #4-5 R drag graces, then #7 R flam grace). Since the flam grace note is softer than the drag grace notes (0.50-0.70 vs. 0.45-0.65), there is a subtle dynamic continuity as the hand transitions from drag-grace to flam-grace role.
- Key distinction: drag grace notes are two strokes on the same hand (a double bounce), while the flam grace note is a single stroke on the opposite hand. Both ornament types appear in this rudiment, making ornament clarity essential.

---

## 4. Dynamics Model

### Velocity Ratios

<!-- List the velocity ratio for each stroke type present in this rudiment. Reference _conventions.md for the full scale. -->

| Stroke Type | Velocity Ratio | Notes |
|-------------|---------------|-------|
| accent (drag primary) | 1.0 | Full accented primary strokes for all four drags |
| accent (flam primary) | 1.0 | Full accented primary strokes for both flams |
| grace (drag) | 0.45 - 0.65 | Each of the eight drag grace notes |
| grace (flam) | 0.50 - 0.70 | Each of the two flam grace notes; slightly louder than drag graces |

All six primary strokes are accented at equal velocity. The two ornament types have slightly different velocity ranges: drag grace notes (0.45-0.65) sit slightly softer than flam grace notes (0.50-0.70), reflecting the different physical mechanics (controlled double bounce vs. offset single stroke). In practice, the difference is subtle, but maintaining it helps preserve the distinct character of each ornament type.

### Accent Pattern

<!-- Text visualization showing accent placement across the cycle. Use > for accents, - for unaccented. -->

```
>        >        >     |  >        >        >
llR      rrL      rL    |  rrL      llR      lR
```

All six primary strokes are accented. There are no taps or unaccented primary strokes in Lesson 25. This makes it one of the most heavily accented rudiments in the PAS 40 -- every primary stroke receives full accent treatment, with the dynamic contrast coming entirely from the soft grace notes.

### Dynamic Contour

Lesson 25 has a unique dynamic contour driven by three consecutive accented strokes per group, each preceded by a different ornament. The first two accents are preceded by drags (soft-soft-LOUD), creating the characteristic "ruff-accent" onset. The third accent is preceded by a flam (soft-LOUD), producing the broader "grace-accent" effect. The three accents create a brief burst of loud, decorated strokes, followed by a rest or transition before the mirrored group begins. The absence of taps gives Lesson 25 a more aggressive, punchy character compared to the single and double drag taps. The alternation between drag ornaments (double bounce) and flam ornament (opposite-hand offset) within a single phrase creates a textural variety unique to this rudiment.

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
| 7 | R | grace (flam) | tap (< 2") | fingers |
| 8 | L | accent | full (12"+) | wrist + arm |
| 9 | R | grace (drag 1) | tap (< 2") | fingers |
| 10 | R | grace (drag 2) | tap (< 2") | fingers / rebound |
| 11 | L | accent | full (12"+) | wrist + arm |
| 12 | L | grace (drag 1) | tap (< 2") | fingers |
| 13 | L | grace (drag 2) | tap (< 2") | fingers / rebound |
| 14 | R | accent | full (12"+) | wrist + arm |
| 15 | L | grace (flam) | tap (< 2") | fingers |
| 16 | R | accent | full (12"+) | wrist + arm |

The physical demands of Lesson 25 combine the challenges of drag and flam techniques within rapid succession. The right hand's path through the first group illustrates the complexity: it plays a full-height drag primary (#3), then must immediately drop to tap height for two drag grace notes (#4-5) preceding the opposite hand's drag. After the second drag's primary (#6, left hand), the right hand produces a flam grace (#7) at tap height, and then the left hand plays the flam primary (#8). The accent-to-grace transition on the same hand (e.g., #3 accent to #4-5 grace) is the same challenge encountered in #33 Double Drag Tap. The additional complexity is the transition from drag grace technique (double bounce) to flam grace technique (single offset stroke) within the same phrase.

### Tempo-Dependent Behavior

<!-- Describe how technique changes across the tempo range. Include specific thresholds where applicable. -->

| Tempo Range | Technique Changes |
|-------------|-------------------|
| Slow (60-100 BPM) | Full wrist + arm motion for all six accented primaries. Both drag and flam grace notes are individually articulated. Drag grace notes are open -- two distinct soft strokes. Flam grace notes have wide separation from primary. The drag-to-drag transition and the drag-to-flam transition within each group are manageable due to wider spacing. Each group's three decorated accents are clearly distinguished. |
| Moderate (100-120 BPM) | Wrist-driven accents. Drag grace notes close up, with the second grace relying on rebound. Flam grace notes at standard -1/32 offset. The accent-to-grace transition tightens, requiring efficient downstroke technique. The three sixteenth-note accents per group begin to feel like a rapid burst. Distinguishing the drag ornament from the flam ornament becomes the primary aural focus. |
| Fast (120-140 BPM) | Finger control for all grace notes. Accents drop to half height class. Drag grace notes compress toward -1/32 and -1/64. Flam grace notes tighten toward -1/64. At the fastest tempos, the drag and flam ornaments begin to blur together sonically, though the physical technique remains distinct (double bounce vs. opposite-hand offset). The three-accent burst becomes a rapid decorated phrase with minimal space between ornaments. |

---

## 6. Variations & Pedagogy

### Common Variations

- **Open Lesson 25**: Exaggerated drag and flam spacing for slow tempo practice and clarity
- **Closed Lesson 25**: Tight ornament spacing at fast tempos where drags approach a crushed ruff
- **Lesson 25 with tap substitution**: Replacing the flam with a plain tap for isolating the drag technique before adding the flam
- **Lesson 25 without drags**: Playing the pattern as plain accent-accent-flam to isolate the rhythmic structure and flam technique
- **All-drag version**: Replacing the flam with a third drag (making the pattern three consecutive drags) as a practice variant

### Prerequisites

- #31 Drag -- the basic drag ornament must be consistent on both hands
- #20 Flam -- the basic flam ornament must be consistent on both hands
- #33 Double Drag Tap -- develops the accent-to-grace transition required for consecutive drags
- #6 Double Stroke Open Roll -- controlled double bounce for the drag grace notes
- Ability to distinguish and produce both drag and flam ornaments in close succession

### Builds Toward

- #30 Flam Drag (the other cross-category drag-flam rudiment)
- Application in drum corps and rudimental solos where mixed ornament textures are required
- Development of ornament versatility -- switching between drag and flam techniques within a phrase

### Teaching Notes

Lesson 25 is the most technically distinctive compound drag rudiment. Its cross-category nature (combining both drags and flams) makes it unique among the PAS 40. Mastering this rudiment requires not only solid drag and flam technique independently, but the ability to transition between the two ornament types smoothly and rapidly.

**Common mistakes:**
- Confusing drag and flam grace notes: the drag uses two same-hand grace notes (double bounce), while the flam uses one opposite-hand grace note. Students often produce a drag where the flam should be, or collapse the drag into a flam
- Missing the accent-to-grace transition: after the first drag primary, the same hand must immediately produce grace notes for the second drag. This requires catching the stick low after the accent
- Inconsistent ornament quality: all four drags should have identical ruff quality, and both flams should have identical flam quality
- Treating all three accents as equally strong: while all are accented, the phrasing naturally gives slightly more weight to the first drag, with the flam serving as a resolving accent
- Rushing the second group: the mirror-image second group should be dynamically and rhythmically identical to the first
- Losing the sixteenth-note pulse: the three primary strokes should be evenly spaced on the sixteenth-note grid

**Practice strategies:**
- Start with the three-note rhythm without ornaments: R L L | L R R (three sixteenth notes, rest, repeat mirrored)
- Add drags first (llR rrL L | rrL llR R), then add the flam (llR rrL rL | rrL llR lR)
- Practice drag-to-flam transitions in isolation: rrL rL (drag then flam on the same beat pair)
- Use a metronome and count: "1 e & (rest)" for each group
- Practice each group separately, then combine
- Record and compare ornament quality across all four drags and both flams

**What to listen for:**
- Clear distinction between drag ornaments (buzz/ruff onset) and flam ornaments (grace-note onset)
- All six accents at consistent volume
- Even sixteenth-note spacing of the three primary strokes per group
- Identical ornament quality between the two groups (right-lead and left-lead)
- The drag grace notes should sound like brief ruffs, and the flam grace notes should sound like clean, slightly separated grace notes

### Historical Context

Lesson 25 gets its name from its position as the 25th lesson in Gardiner A. Strube's *Drum and Fife Instructor* (1870). Strube either did not know the traditional name for this rudiment, could not decide on one, or simply forgot to include one, so the pattern was identified only by its lesson number. When the National Association of Rudimental Drummers (NARD) standardized its list of 26 rudiments in 1933, the NARD 26 drew almost entirely from Strube's 25 lessons (with the addition of the Single Stroke Roll), and the nameless 25th lesson retained its designation as "Lesson 25." The rudiment has been a staple of British and American field drumming since at least the 18th century, with some historians tracing patterns resembling it back to the English Foot March of the 1500s. The Percussive Arts Society retained it as PAS #34 in the 1984 expansion to 40 rudiments. Lesson 25 is notable as a cross-category rudiment -- despite being classified under the drag category, it contains a flam, making it one of the few rudiments that bridges two ornament families. The only other cross-category rudiment in the PAS 40 is #30 Flam Drag, which is classified under flam but contains a drag.

---

## 7. Notation

### Stick Notation

<!-- Text-based notation using R/L, accent markers, grace note prefixes, and bar separators. -->

```
>          >          >       |  >          >          >
llR        rrL        rL      |  rrL        llR        lR
```

### Grid Visualization

<!-- Aligned grid showing beat position, hand, stroke type, and accents. -->

```
Beat:    | 1           e           &         | (rest)      |
Hand:    | (ll)R       (rr)L       (r)L      |             |
Type:    |  gg  A       gg  A       g  A     |             |
Accent:  |      >           >          >     |             |
Orn:     | drag         drag        flam     |             |

Beat:    | 3           e           &         | (rest)      |
Hand:    | (rr)L       (ll)R       (l)R      |             |
Type:    |  gg  A       gg  A       g  A     |             |
Accent:  |      >           >          >     |             |
Orn:     | drag         drag        flam     |             |
```

Grace notes shown in doubled parentheses `(ll)` or `(rr)` indicate drag grace notes (two same-hand grace notes forming a double bounce). Grace notes shown in single parentheses `(r)` or `(l)` indicate flam grace notes (single opposite-hand grace note). Drag graces fall at -1/16 and -1/32 beat from the primary. Flam graces fall at -1/32 beat from the primary. The three primary strokes in each group occupy consecutive sixteenth-note positions. The Orn row labels each ornament type to distinguish between drags and flams.
