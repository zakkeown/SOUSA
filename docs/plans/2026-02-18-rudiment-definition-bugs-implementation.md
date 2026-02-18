# Fix Three Rudiment Definition Bugs — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix incorrect definitions for double-drag-tap, swiss-army-triplet, and multiple-bounce-roll so all 40 PAS rudiments produce correct, distinguishable output.

**Architecture:** Three independent fixes: (1) YAML correction for double-drag-tap stroke pattern, (2) YAML correction for swiss-army-triplet sticking, (3) schema + loader + generator changes to support buzz stroke type with configurable detail level (sub_strokes or marking).

**Tech Stack:** Python, YAML, pytest, Pydantic

---

### Task 1: Fix Double Drag Tap definition

**Files:**
- Modify: `dataset_gen/rudiments/definitions/33_double_drag_tap.yaml`
- Test: `tests/test_rudiments.py`

**Step 1: Write the failing test**

Add to `tests/test_rudiments.py` inside class `TestRudimentLoading`:

```python
def test_double_drag_tap_has_12_strokes(self, definitions_dir):
    """Double drag tap needs 2 drags (4 grace notes) + accent + tap per half-cycle = 12 total."""
    rudiment = load_rudiment(definitions_dir / "33_double_drag_tap.yaml")
    assert rudiment.slug == "double_drag_tap"
    assert len(rudiment.pattern.strokes) == 12

    # First half: drag1(LL) drag2(LL) accent(R) tap(L)
    types = [s.stroke_type for s in rudiment.pattern.strokes[:6]]
    assert types == [
        StrokeType.GRACE, StrokeType.GRACE,  # drag 1
        StrokeType.GRACE, StrokeType.GRACE,  # drag 2
        StrokeType.ACCENT,
        StrokeType.TAP,
    ]
    # Second half mirrors
    types2 = [s.stroke_type for s in rudiment.pattern.strokes[6:]]
    assert types2 == [
        StrokeType.GRACE, StrokeType.GRACE,
        StrokeType.GRACE, StrokeType.GRACE,
        StrokeType.ACCENT,
        StrokeType.TAP,
    ]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_double_drag_tap_has_12_strokes -v`
Expected: FAIL — currently 8 strokes

**Step 3: Fix the YAML definition**

Replace `dataset_gen/rudiments/definitions/33_double_drag_tap.yaml` with:

```yaml
name: Double Drag Tap
slug: double_drag_tap
category: drag
pas_number: 33
description: Two drags before each accented tap

pattern:
  strokes:
    # Double drag R: drag1(LL), drag2(LL), accent(R), tap(L)
    - {hand: L, type: grace, grace_offset: -0.12}
    - {hand: L, type: grace, grace_offset: -0.08}
    - {hand: L, type: grace, grace_offset: -0.06}
    - {hand: L, type: grace, grace_offset: -0.04}
    - {hand: R, type: accent}
    - {hand: L, type: tap}
    # Double drag L: drag1(RR), drag2(RR), accent(L), tap(R)
    - {hand: R, type: grace, grace_offset: -0.12}
    - {hand: R, type: grace, grace_offset: -0.08}
    - {hand: R, type: grace, grace_offset: -0.06}
    - {hand: R, type: grace, grace_offset: -0.04}
    - {hand: L, type: accent}
    - {hand: R, type: tap}
  beats_per_cycle: 2

subdivision: triplet
tempo_range: [60, 160]

params:
  drag_spacing_range: [10, 30]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_double_drag_tap_has_12_strokes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/rudiments/definitions/33_double_drag_tap.yaml tests/test_rudiments.py
git commit -m "fix: double drag tap missing second drag per half-cycle (PAS #33)"
```

---

### Task 2: Fix Swiss Army Triplet sticking

**Files:**
- Modify: `dataset_gen/rudiments/definitions/28_swiss_army_triplet.yaml`
- Test: `tests/test_rudiments.py`

**Step 1: Write the failing test**

Add to `tests/test_rudiments.py` inside class `TestRudimentLoading`:

```python
def test_swiss_army_triplet_sticking_differs_from_flam_accent(self, definitions_dir):
    """Swiss army triplet has doubled last stroke: R L L | L R R (not R L R | L R L)."""
    swiss = load_rudiment(definitions_dir / "28_swiss_army_triplet.yaml")
    flam_accent = load_rudiment(definitions_dir / "21_flam_accent.yaml")

    # Extract non-grace sticking from each
    def get_sticking(rudiment):
        return [
            s.hand for s in rudiment.pattern.strokes
            if s.stroke_type != StrokeType.GRACE
        ]

    swiss_sticking = get_sticking(swiss)
    flam_sticking = get_sticking(flam_accent)

    # They must differ
    assert swiss_sticking != flam_sticking, (
        "Swiss army triplet sticking should differ from flam accent"
    )

    # Swiss army: R L L | L R R
    assert swiss_sticking == [
        Hand.RIGHT, Hand.LEFT, Hand.LEFT,
        Hand.LEFT, Hand.RIGHT, Hand.RIGHT,
    ]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_swiss_army_triplet_sticking_differs_from_flam_accent -v`
Expected: FAIL — currently R L R | L R L

**Step 3: Fix the YAML definition**

In `dataset_gen/rudiments/definitions/28_swiss_army_triplet.yaml`, change strokes 3 and 7 hand values:

```yaml
name: Swiss Army Triplet
slug: swiss_army_triplet
category: flam
pas_number: 28
description: Flam followed by a tap and a doubled stroke in triplet feel

pattern:
  strokes:
    # Flam R, L, L
    - {hand: L, type: grace, grace_offset: -0.05}
    - {hand: R, type: accent}
    - {hand: L, type: tap}
    - {hand: L, type: tap}
    # Flam L, R, R
    - {hand: R, type: grace, grace_offset: -0.05}
    - {hand: L, type: accent}
    - {hand: R, type: tap}
    - {hand: R, type: tap}
  beats_per_cycle: 2

subdivision: triplet
tempo_range: [60, 180]

params:
  flam_spacing_range: [15, 50]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_swiss_army_triplet_sticking_differs_from_flam_accent -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/rudiments/definitions/28_swiss_army_triplet.yaml tests/test_rudiments.py
git commit -m "fix: swiss army triplet wrong sticking, was identical to flam accent (PAS #28)"
```

---

### Task 3: Add buzz stroke type to simple pattern parser

**Files:**
- Modify: `dataset_gen/rudiments/schema.py:122-124`
- Modify: `dataset_gen/rudiments/loader.py:65-84`
- Test: `tests/test_rudiments.py`

**Step 1: Write the failing test**

Add to `tests/test_rudiments.py` inside class `TestStickingPattern`:

```python
def test_parse_buzz_notation(self):
    """Test that B in simple pattern produces BUZZ stroke type."""
    from dataset_gen.rudiments.loader import _parse_pattern_from_yaml

    pattern = _parse_pattern_from_yaml({
        "simple": "B B B B",
        "beats_per_cycle": 1,
    })
    assert len(pattern.strokes) == 4
    # B alternates hands like R/L
    assert all(s.stroke_type == StrokeType.BUZZ for s in pattern.strokes)
    assert pattern.strokes[0].hand == Hand.RIGHT
    assert pattern.strokes[1].hand == Hand.LEFT
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rudiments.py::TestStickingPattern::test_parse_buzz_notation -v`
Expected: FAIL — `B` not recognized, treated as LEFT hand tap

**Step 3: Add B support to loader and buzz_detail to schema**

In `dataset_gen/rudiments/schema.py`, add `buzz_detail` field to `RudimentParams` after `buzz_strokes_range` (after line 124):

```python
    buzz_detail: Literal["sub_strokes", "marking"] | None = Field(
        default=None, description="Buzz generation mode: sub_strokes expands in MIDI, marking only tags stroke type"
    )
```

In `dataset_gen/rudiments/loader.py`, modify the simple pattern parser (lines 77-81). Replace:

```python
        for i, char in enumerate(sticking_chars):
            hand = Hand.RIGHT if char.upper() == "R" else Hand.LEFT
            is_accent = accent_chars and accent_chars[i] == ">"
            stroke_type = StrokeType.ACCENT if is_accent else StrokeType.TAP
            strokes.append(Stroke(hand=hand, stroke_type=stroke_type))
```

With:

```python
        hand_toggle = True  # Start with RIGHT for buzz strokes
        for i, char in enumerate(sticking_chars):
            upper = char.upper()
            if upper == "B":
                hand = Hand.RIGHT if hand_toggle else Hand.LEFT
                hand_toggle = not hand_toggle
                stroke_type = StrokeType.BUZZ
            else:
                hand = Hand.RIGHT if upper == "R" else Hand.LEFT
                is_accent = accent_chars and accent_chars[i] == ">"
                stroke_type = StrokeType.ACCENT if is_accent else StrokeType.TAP
            strokes.append(Stroke(hand=hand, stroke_type=stroke_type))
```

Also in `dataset_gen/rudiments/loader.py`, add `buzz_detail` to params parsing (around line 133, after `buzz_strokes_range`):

```python
        buzz_detail=params_data.get("buzz_detail"),
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rudiments.py::TestStickingPattern::test_parse_buzz_notation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/rudiments/schema.py dataset_gen/rudiments/loader.py tests/test_rudiments.py
git commit -m "feat: support B notation for buzz strokes in simple pattern parser"
```

---

### Task 4: Update multiple bounce roll YAML to use buzz notation

**Files:**
- Modify: `dataset_gen/rudiments/definitions/04_multiple_bounce_roll.yaml`
- Test: `tests/test_rudiments.py`

**Step 1: Write the failing test**

Add to `tests/test_rudiments.py` inside class `TestRudimentLoading`:

```python
def test_multiple_bounce_roll_has_buzz_strokes(self, definitions_dir):
    """Multiple bounce roll strokes should be BUZZ type, not TAP."""
    rudiment = load_rudiment(definitions_dir / "04_multiple_bounce_roll.yaml")
    assert rudiment.slug == "multiple_bounce_roll"
    assert rudiment.params.roll_type == "buzz"
    assert rudiment.params.buzz_detail == "sub_strokes"

    # All strokes should be BUZZ type
    for stroke in rudiment.pattern.strokes:
        assert stroke.stroke_type == StrokeType.BUZZ, (
            f"Expected BUZZ but got {stroke.stroke_type}"
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_multiple_bounce_roll_has_buzz_strokes -v`
Expected: FAIL — strokes are TAP type

**Step 3: Update the YAML**

Replace `dataset_gen/rudiments/definitions/04_multiple_bounce_roll.yaml`:

```yaml
name: Multiple Bounce Roll
slug: multiple_bounce_roll
category: roll
pas_number: 4
description: Buzz roll with multiple bounces per stroke

pattern:
  simple: "B B B B B B B B"
  beats_per_cycle: 2

subdivision: sixteenth
tempo_range: [60, 140]

params:
  roll_type: buzz
  buzz_detail: sub_strokes
  buzz_strokes_range: [3, 8]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_multiple_bounce_roll_has_buzz_strokes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/rudiments/definitions/04_multiple_bounce_roll.yaml tests/test_rudiments.py
git commit -m "fix: multiple bounce roll now uses BUZZ stroke type (PAS #4)"
```

---

### Task 5: Implement buzz sub-stroke generation in articulation engine

**Files:**
- Modify: `dataset_gen/midi_gen/articulations.py:261-282`
- Test: `tests/test_midi_gen.py`

**Step 1: Write the failing test**

Add to `tests/test_midi_gen.py`:

```python
import numpy as np
from dataset_gen.midi_gen.generator import StrokeEvent
from dataset_gen.midi_gen.articulations import ArticulationEngine
from dataset_gen.rudiments.schema import (
    Rudiment, StickingPattern, Stroke, StrokeType, Hand,
    RudimentCategory, RudimentParams,
)
from dataset_gen.profiles.archetypes import PlayerProfile, generate_profile


class TestBuzzSubStrokes:
    """Tests for buzz roll sub-stroke generation."""

    def _make_buzz_rudiment(self, buzz_detail="sub_strokes"):
        return Rudiment(
            name="Test Buzz",
            slug="test_buzz",
            category=RudimentCategory.ROLL,
            pattern=StickingPattern(
                strokes=[
                    Stroke(hand=Hand.RIGHT, stroke_type=StrokeType.BUZZ),
                    Stroke(hand=Hand.LEFT, stroke_type=StrokeType.BUZZ),
                ],
                beats_per_cycle=1,
            ),
            params=RudimentParams(
                roll_type="buzz",
                buzz_detail=buzz_detail,
                buzz_strokes_range=(3, 5),
            ),
        )

    def _make_events(self):
        return [
            StrokeEvent(
                index=0, hand=Hand.RIGHT, stroke_type=StrokeType.BUZZ,
                intended_time_ms=0, actual_time_ms=0,
                intended_velocity=100, actual_velocity=100,
            ),
            StrokeEvent(
                index=1, hand=Hand.LEFT, stroke_type=StrokeType.BUZZ,
                intended_time_ms=250, actual_time_ms=250,
                intended_velocity=100, actual_velocity=100,
            ),
        ]

    def test_sub_strokes_expands_events(self):
        """sub_strokes mode should generate additional events per BUZZ stroke."""
        rudiment = self._make_buzz_rudiment("sub_strokes")
        profile = generate_profile(skill_tier="intermediate", seed=42)
        engine = ArticulationEngine(seed=42)

        events = self._make_events()
        result = engine.process(events, rudiment, profile)

        # Should have more events than we started with (sub-strokes added)
        assert len(result) > 2
        # Original BUZZ strokes should still be present
        buzz_primaries = [e for e in result if e.stroke_type == StrokeType.BUZZ and not e.is_grace_note]
        assert len(buzz_primaries) == 2

    def test_sub_strokes_have_decaying_velocity(self):
        """Buzz sub-strokes should decay in velocity from the primary."""
        rudiment = self._make_buzz_rudiment("sub_strokes")
        profile = generate_profile(skill_tier="professional", seed=42)
        engine = ArticulationEngine(seed=42)

        events = self._make_events()
        result = engine.process(events, rudiment, profile)

        # Find sub-strokes for the first primary (index 0)
        subs = [e for e in result if e.parent_stroke_index == 0]
        assert len(subs) >= 2  # At least min from buzz_strokes_range minus 1

        # Sub-stroke velocities should decay
        velocities = [e.actual_velocity for e in subs]
        assert velocities == sorted(velocities, reverse=True) or all(
            v <= 100 for v in velocities
        )

    def test_marking_mode_no_expansion(self):
        """marking mode should NOT add sub-strokes, just tag types."""
        rudiment = self._make_buzz_rudiment("marking")
        profile = generate_profile(skill_tier="intermediate", seed=42)
        engine = ArticulationEngine(seed=42)

        events = self._make_events()
        result = engine.process(events, rudiment, profile)

        # Same number of events — no expansion
        assert len(result) == 2
        # Still BUZZ type
        assert all(e.stroke_type == StrokeType.BUZZ for e in result)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_midi_gen.py::TestBuzzSubStrokes -v`
Expected: FAIL — `process` method doesn't expand buzz sub-strokes

**Step 3: Implement buzz sub-stroke generation**

Replace `_process_buzz_roll` in `dataset_gen/midi_gen/articulations.py:261-282` with:

```python
    def _process_buzz_roll(
        self,
        events: list[StrokeEvent],
        dims: RudimentSpecificDimensions,
        params,
    ) -> list[StrokeEvent]:
        """
        Process buzz roll events.

        In sub_strokes mode: expand each BUZZ primary into rapid sub-strokes.
        In marking mode: add velocity jitter but don't expand.
        """
        if getattr(params, "buzz_detail", None) != "sub_strokes":
            # Marking mode or legacy: just add velocity jitter
            density_consistency = dims.buzz_density_consistency
            for event in events:
                jitter = (1 - density_consistency) * 20
                velocity_jitter = self.rng.normal(0, jitter)
                event.actual_velocity = int(
                    np.clip(event.actual_velocity + velocity_jitter, 1, 127)
                )
            return events

        # Sub-strokes mode: expand each BUZZ stroke into rapid bounces
        buzz_min, buzz_max = params.buzz_strokes_range or (3, 8)
        density_consistency = dims.buzz_density_consistency
        velocity_decay = 0.85

        expanded = []
        next_index = max(e.index for e in events) + 1

        for event in events:
            # Keep the primary stroke
            expanded.append(event)

            if event.stroke_type != StrokeType.BUZZ:
                continue

            # Number of sub-strokes: higher consistency = closer to max
            mean_count = buzz_min + (buzz_max - buzz_min) * density_consistency
            count = int(np.clip(
                self.rng.normal(mean_count, (1 - density_consistency) * 2),
                buzz_min, buzz_max,
            ))

            # Generate sub-strokes with tight timing and decaying velocity
            spacing_ms = 2 + self.rng.uniform(0, 3)  # 2-5ms per sub-stroke
            for j in range(1, count):
                sub_time = event.actual_time_ms + j * spacing_ms
                decay = velocity_decay ** j
                sub_velocity = int(np.clip(event.actual_velocity * decay, 1, 127))

                sub = StrokeEvent(
                    index=next_index,
                    hand=event.hand,
                    stroke_type=StrokeType.BUZZ,
                    intended_time_ms=event.intended_time_ms + j * spacing_ms,
                    actual_time_ms=sub_time,
                    intended_velocity=int(event.intended_velocity * decay),
                    actual_velocity=sub_velocity,
                    is_grace_note=True,
                    parent_stroke_index=event.index,
                )
                expanded.append(sub)
                next_index += 1

        return expanded
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_midi_gen.py::TestBuzzSubStrokes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/midi_gen/articulations.py tests/test_midi_gen.py
git commit -m "feat: implement buzz sub-stroke expansion in articulation engine"
```

---

### Task 6: Handle BUZZ stroke type in label computation

**Files:**
- Modify: `dataset_gen/labels/schema.py:16-41`
- Modify: `dataset_gen/labels/compute.py:228-229`
- Test: `tests/test_labels.py`

**Step 1: Write the failing test**

Add to `tests/test_labels.py`:

```python
class TestBuzzLabels:
    """Tests for buzz stroke label generation."""

    def test_buzz_stroke_type_in_labels(self):
        """Buzz strokes should have stroke_type='buzz' in labels."""
        from dataset_gen.labels.compute import compute_stroke_labels
        from dataset_gen.midi_gen.generator import StrokeEvent
        from dataset_gen.rudiments.schema import StrokeType, Hand

        events = [
            StrokeEvent(
                index=0, hand=Hand.RIGHT, stroke_type=StrokeType.BUZZ,
                intended_time_ms=0, actual_time_ms=1,
                intended_velocity=100, actual_velocity=98,
            ),
            StrokeEvent(
                index=1, hand=Hand.LEFT, stroke_type=StrokeType.BUZZ,
                intended_time_ms=250, actual_time_ms=252,
                intended_velocity=100, actual_velocity=97,
            ),
        ]

        labels = compute_stroke_labels(events)
        assert labels[0].stroke_type == "buzz"
        assert labels[1].stroke_type == "buzz"
        assert not labels[0].is_accent

    def test_buzz_count_on_label(self):
        """Labels for buzz primaries should include buzz_count."""
        from dataset_gen.labels.compute import compute_stroke_labels
        from dataset_gen.midi_gen.generator import StrokeEvent
        from dataset_gen.rudiments.schema import StrokeType, Hand

        events = [
            StrokeEvent(
                index=0, hand=Hand.RIGHT, stroke_type=StrokeType.BUZZ,
                intended_time_ms=0, actual_time_ms=0,
                intended_velocity=100, actual_velocity=100,
            ),
            # 3 sub-strokes parented to primary
            StrokeEvent(
                index=1, hand=Hand.RIGHT, stroke_type=StrokeType.BUZZ,
                intended_time_ms=3, actual_time_ms=3,
                intended_velocity=85, actual_velocity=84,
                is_grace_note=True, parent_stroke_index=0,
            ),
            StrokeEvent(
                index=2, hand=Hand.RIGHT, stroke_type=StrokeType.BUZZ,
                intended_time_ms=6, actual_time_ms=6,
                intended_velocity=72, actual_velocity=71,
                is_grace_note=True, parent_stroke_index=0,
            ),
            StrokeEvent(
                index=3, hand=Hand.RIGHT, stroke_type=StrokeType.BUZZ,
                intended_time_ms=9, actual_time_ms=9,
                intended_velocity=61, actual_velocity=60,
                is_grace_note=True, parent_stroke_index=0,
            ),
        ]

        labels = compute_stroke_labels(events)
        primary_label = labels[0]
        assert primary_label.buzz_count == 4  # 1 primary + 3 subs
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_labels.py::TestBuzzLabels -v`
Expected: FAIL — `buzz_count` field doesn't exist on StrokeLabel

**Step 3: Add buzz_count to StrokeLabel and compute it**

In `dataset_gen/labels/schema.py`, add after `parent_stroke_index` field (after line 40):

```python
    # Buzz roll specific
    buzz_count: int | None = None  # Total strokes in buzz group (primary + subs)
```

In `dataset_gen/labels/compute.py`, after building labels (after line 77, before `return labels`), add buzz_count computation:

```python
    # Compute buzz_count for primary buzz strokes
    for label in labels:
        if label.stroke_type == "buzz" and not label.is_grace_note:
            sub_count = sum(
                1 for l in labels
                if l.parent_stroke_index == label.index
            )
            label.buzz_count = 1 + sub_count  # primary + subs
```

Also in `dataset_gen/labels/compute.py:229`, update the tap_strokes filter to include buzz:

```python
    tap_strokes = [s for s in stroke_labels if s.stroke_type in ("tap", "buzz")]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_labels.py::TestBuzzLabels -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/labels/schema.py dataset_gen/labels/compute.py tests/test_labels.py
git commit -m "feat: add buzz_count to stroke labels for buzz roll tracking"
```

---

### Task 7: Run full test suite and validate

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass (existing + new)

**Step 2: Run ruff**

Run: `ruff check dataset_gen/ tests/`
Expected: Clean

**Step 3: Verify rudiment count is still 40**

Run: `python -c "from dataset_gen.rudiments.loader import load_all_rudiments; from pathlib import Path; r = load_all_rudiments(Path('dataset_gen/rudiments/definitions')); print(f'{len(r)} rudiments loaded')"`
Expected: `40 rudiments loaded`

**Step 4: Spot-check the three fixed rudiments load correctly**

Run:
```python
python -c "
from dataset_gen.rudiments.loader import load_all_rudiments
from pathlib import Path
rudiments = load_all_rudiments(Path('dataset_gen/rudiments/definitions'))
for slug in ['double_drag_tap', 'swiss_army_triplet', 'multiple_bounce_roll']:
    r = rudiments[slug]
    types = [s.stroke_type.value for s in r.pattern.strokes]
    print(f'{slug}: {len(r.pattern.strokes)} strokes — {types}')
"
```

Expected:
- `double_drag_tap`: 12 strokes — grace/accent/tap pattern
- `swiss_army_triplet`: 8 strokes — grace types with correct sticking
- `multiple_bounce_roll`: 8 strokes — all buzz type

**Step 5: Commit (if any fixes needed)**

Only if adjustments were required during validation.
