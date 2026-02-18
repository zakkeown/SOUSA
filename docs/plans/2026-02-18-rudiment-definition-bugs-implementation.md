# Fix Three Rudiment Definition Bugs - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three incorrect rudiment definitions: double drag tap (missing drag), swiss army triplet (wrong sticking), and multiple bounce roll (no buzz generation).

**Architecture:** Bugs 1-2 are YAML-only fixes. Bug 3 requires plumbing `StrokeType.BUZZ` through the simple pattern parser, adding a `buzz_detail` param to the schema, implementing sub-stroke expansion in the articulation processor, and handling BUZZ strokes in ideal event generation.

**Tech Stack:** Python 3.12, pydantic, numpy, midiutil, pytest

---

### Task 1: Fix Double Drag Tap YAML (PAS #33)

**Files:**
- Modify: `dataset_gen/rudiments/definitions/33_double_drag_tap.yaml`
- Test: `tests/test_rudiments.py`

**Step 1: Write the failing test**

Add to `tests/test_rudiments.py` in the `TestRudimentLoading` class:

```python
def test_double_drag_tap_has_12_strokes(self, definitions_dir):
    """Double drag tap has two drags per accent, not one (12 strokes per cycle)."""
    rudiment = load_rudiment(definitions_dir / "33_double_drag_tap.yaml")
    assert rudiment.name == "Double Drag Tap"
    # Two drags (4 grace notes) + accent + tap, mirrored = 12 strokes
    assert len(rudiment.pattern.strokes) == 12
    # First half: GG GG A T
    types = [s.stroke_type for s in rudiment.pattern.strokes]
    assert types[:6] == [
        StrokeType.GRACE, StrokeType.GRACE,  # Drag 1
        StrokeType.GRACE, StrokeType.GRACE,  # Drag 2
        StrokeType.ACCENT,
        StrokeType.TAP,
    ]
    # Second half mirrors
    assert types[6:12] == [
        StrokeType.GRACE, StrokeType.GRACE,
        StrokeType.GRACE, StrokeType.GRACE,
        StrokeType.ACCENT,
        StrokeType.TAP,
    ]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_double_drag_tap_has_12_strokes -v`
Expected: FAIL — currently 8 strokes, expected 12.

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
    # Double drag R: drag-drag-accent, tap L
    - {hand: L, type: grace, grace_offset: -0.12}
    - {hand: L, type: grace, grace_offset: -0.09}
    - {hand: L, type: grace, grace_offset: -0.06}
    - {hand: L, type: grace, grace_offset: -0.03}
    - {hand: R, type: accent}
    - {hand: L, type: tap}
    # Double drag L: drag-drag-accent, tap R
    - {hand: R, type: grace, grace_offset: -0.12}
    - {hand: R, type: grace, grace_offset: -0.09}
    - {hand: R, type: grace, grace_offset: -0.06}
    - {hand: R, type: grace, grace_offset: -0.03}
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

```
git add dataset_gen/rudiments/definitions/33_double_drag_tap.yaml tests/test_rudiments.py
git commit -m "fix: double drag tap — add missing second drag per accent (PAS #33)"
```

---

### Task 2: Fix Swiss Army Triplet YAML (PAS #28)

**Files:**
- Modify: `dataset_gen/rudiments/definitions/28_swiss_army_triplet.yaml`
- Test: `tests/test_rudiments.py`

**Step 1: Write the failing test**

Add to `tests/test_rudiments.py` in the `TestRudimentLoading` class:

```python
def test_swiss_army_triplet_sticking_differs_from_flam_accent(self, definitions_dir):
    """Swiss army triplet has doubled last stroke, not alternating like flam accent."""
    swiss = load_rudiment(definitions_dir / "28_swiss_army_triplet.yaml")
    flam_accent = load_rudiment(definitions_dir / "21_flam_accent.yaml")

    swiss_hands = [(s.hand, s.stroke_type) for s in swiss.pattern.strokes]
    flam_hands = [(s.hand, s.stroke_type) for s in flam_accent.pattern.strokes]

    # They must differ (they're currently identical — that's the bug)
    assert swiss_hands != flam_hands

    # Swiss army: G-R L L | G-L R R (doubled last stroke per group)
    swiss_main = [s.hand for s in swiss.pattern.strokes if s.stroke_type != StrokeType.GRACE]
    assert swiss_main == [Hand.RIGHT, Hand.LEFT, Hand.LEFT, Hand.LEFT, Hand.RIGHT, Hand.RIGHT]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_swiss_army_triplet_sticking_differs_from_flam_accent -v`
Expected: FAIL — currently identical to flam accent.

**Step 3: Fix the YAML definition**

Replace `dataset_gen/rudiments/definitions/28_swiss_army_triplet.yaml` with:

```yaml
name: Swiss Army Triplet
slug: swiss_army_triplet
category: flam
pas_number: 28
description: Flam followed by a tap and a same-hand double in triplet feel

pattern:
  strokes:
    # Flam R, L, L (doubled last stroke)
    - {hand: L, type: grace, grace_offset: -0.05}
    - {hand: R, type: accent}
    - {hand: L, type: tap}
    - {hand: L, type: tap}
    # Flam L, R, R (doubled last stroke)
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

```
git add dataset_gen/rudiments/definitions/28_swiss_army_triplet.yaml tests/test_rudiments.py
git commit -m "fix: swiss army triplet — correct sticking to R L L | L R R (PAS #28)"
```

---

### Task 3: Add buzz_detail param to schema and loader

**Files:**
- Modify: `dataset_gen/rudiments/schema.py:122` (after `buzz_strokes_range`)
- Modify: `dataset_gen/rudiments/loader.py:127-138`
- Test: `tests/test_rudiments.py`

**Step 1: Write the failing test**

Add to `tests/test_rudiments.py`:

```python
def test_buzz_detail_param_loaded(self, definitions_dir):
    """Multiple bounce roll should have buzz_detail parameter."""
    rudiment = load_rudiment(definitions_dir / "04_multiple_bounce_roll.yaml")
    assert rudiment.params.buzz_detail == "sub_strokes"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_buzz_detail_param_loaded -v`
Expected: FAIL — `buzz_detail` attribute doesn't exist yet.

**Step 3: Add buzz_detail to schema**

In `dataset_gen/rudiments/schema.py`, after line 124 (`buzz_strokes_range`), add:

```python
    buzz_detail: Literal["sub_strokes", "marking"] | None = Field(
        default=None, description="Buzz mode: sub_strokes generates bounce events, marking just tags stroke type"
    )
```

**Step 4: Add buzz_detail to loader**

In `dataset_gen/rudiments/loader.py`, inside `_parse_rudiment_params` at line 138 (before the closing paren), add:

```python
        buzz_detail=params_data.get("buzz_detail"),
```

**Step 5: Add buzz_detail to YAML**

In `dataset_gen/rudiments/definitions/04_multiple_bounce_roll.yaml`, add under params:

```yaml
  buzz_detail: sub_strokes
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_buzz_detail_param_loaded -v`
Expected: PASS

**Step 7: Commit**

```
git add dataset_gen/rudiments/schema.py dataset_gen/rudiments/loader.py dataset_gen/rudiments/definitions/04_multiple_bounce_roll.yaml tests/test_rudiments.py
git commit -m "feat: add buzz_detail parameter to rudiment schema and loader"
```

---

### Task 4: Support BUZZ strokes in simple pattern parser

**Files:**
- Modify: `dataset_gen/rudiments/loader.py:77-81`
- Modify: `dataset_gen/rudiments/definitions/04_multiple_bounce_roll.yaml`
- Test: `tests/test_rudiments.py`

**Step 1: Write the failing test**

Add to `tests/test_rudiments.py` in `TestRudimentLoading`:

```python
def test_multiple_bounce_roll_has_buzz_strokes(self, definitions_dir):
    """Multiple bounce roll strokes should be BUZZ type, not TAP."""
    rudiment = load_rudiment(definitions_dir / "04_multiple_bounce_roll.yaml")
    for stroke in rudiment.pattern.strokes:
        assert stroke.stroke_type == StrokeType.BUZZ, (
            f"Expected BUZZ, got {stroke.stroke_type}"
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_multiple_bounce_roll_has_buzz_strokes -v`
Expected: FAIL — strokes are TAP, not BUZZ.

**Step 3: Update simple parser to support B notation**

In `dataset_gen/rudiments/loader.py`, replace lines 77-81:

```python
        for i, char in enumerate(sticking_chars):
            hand = Hand.RIGHT if char.upper() == "R" else Hand.LEFT
            is_accent = accent_chars and accent_chars[i] == ">"
            stroke_type = StrokeType.ACCENT if is_accent else StrokeType.TAP
            strokes.append(Stroke(hand=hand, stroke_type=stroke_type))
```

with:

```python
        for i, char in enumerate(sticking_chars):
            upper = char.upper()
            if upper == "B":
                # Buzz notation: alternates R/L based on position
                hand = Hand.RIGHT if i % 2 == 0 else Hand.LEFT
                stroke_type = StrokeType.BUZZ
            else:
                hand = Hand.RIGHT if upper == "R" else Hand.LEFT
                is_accent = accent_chars and accent_chars[i] == ">"
                stroke_type = StrokeType.ACCENT if is_accent else StrokeType.TAP
            strokes.append(Stroke(hand=hand, stroke_type=stroke_type))
```

**Step 4: Update YAML to use B notation**

Replace the pattern in `04_multiple_bounce_roll.yaml`:

```yaml
pattern:
  simple: "B B B B B B B B"
  beats_per_cycle: 2
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_rudiments.py::TestRudimentLoading::test_multiple_bounce_roll_has_buzz_strokes -v`
Expected: PASS

**Step 6: Commit**

```
git add dataset_gen/rudiments/loader.py dataset_gen/rudiments/definitions/04_multiple_bounce_roll.yaml tests/test_rudiments.py
git commit -m "feat: support B (buzz) notation in simple pattern parser"
```

---

### Task 5: Handle BUZZ in ideal event generation

**Files:**
- Modify: `dataset_gen/midi_gen/generator.py:199-206`
- Test: `tests/test_rudiments.py`

**Step 1: Write the failing test**

Add a new test class to `tests/test_rudiments.py`:

```python
from dataset_gen.midi_gen.generator import MIDIGenerator
from dataset_gen.profiles.archetypes import generate_profile


class TestBuzzGeneration:
    """Tests for buzz roll MIDI generation."""

    @pytest.fixture
    def definitions_dir(self):
        return Path(__file__).parent.parent / "dataset_gen" / "rudiments" / "definitions"

    def test_buzz_events_have_correct_velocity(self, definitions_dir):
        """Buzz strokes should have their own ideal velocity (not default tap)."""
        rudiment = load_rudiment(definitions_dir / "04_multiple_bounce_roll.yaml")
        profile = generate_profile(tier="intermediate", seed=42)
        gen = MIDIGenerator(seed=42)
        events = gen._generate_ideal_events(rudiment, tempo_bpm=100, num_cycles=1)
        for event in events:
            assert event.stroke_type == StrokeType.BUZZ
            # Buzz should have a distinct velocity (not tap's 65)
            assert event.intended_velocity == 70
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rudiments.py::TestBuzzGeneration::test_buzz_events_have_correct_velocity -v`
Expected: FAIL — velocity is 65 (tap default), expected 70.

**Step 3: Add BUZZ velocity in generator**

In `dataset_gen/midi_gen/generator.py`, in `_generate_ideal_events`, replace lines 199-206:

```python
                # Calculate ideal velocity
                if stroke.stroke_type == StrokeType.ACCENT:
                    ideal_velocity = 120
                elif stroke.stroke_type == StrokeType.GRACE:
                    ideal_velocity = 40
                elif stroke.stroke_type == StrokeType.DIDDLE:
                    ideal_velocity = 75
                else:  # TAP
                    ideal_velocity = 65
```

with:

```python
                # Calculate ideal velocity
                if stroke.stroke_type == StrokeType.ACCENT:
                    ideal_velocity = 120
                elif stroke.stroke_type == StrokeType.GRACE:
                    ideal_velocity = 40
                elif stroke.stroke_type == StrokeType.DIDDLE:
                    ideal_velocity = 75
                elif stroke.stroke_type == StrokeType.BUZZ:
                    ideal_velocity = 70
                else:  # TAP
                    ideal_velocity = 65
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rudiments.py::TestBuzzGeneration::test_buzz_events_have_correct_velocity -v`
Expected: PASS

**Step 5: Commit**

```
git add dataset_gen/midi_gen/generator.py tests/test_rudiments.py
git commit -m "feat: add BUZZ stroke velocity in ideal event generation"
```

---

### Task 6: Implement buzz sub-stroke expansion

**Files:**
- Modify: `dataset_gen/midi_gen/articulations.py:261-282`
- Test: `tests/test_rudiments.py`

**Step 1: Write the failing test**

Add to `TestBuzzGeneration` in `tests/test_rudiments.py`:

```python
def test_buzz_sub_strokes_generated(self, definitions_dir):
    """sub_strokes mode should expand each BUZZ into multiple rapid sub-strokes."""
    from dataset_gen.midi_gen.articulations import ArticulationProcessor

    rudiment = load_rudiment(definitions_dir / "04_multiple_bounce_roll.yaml")
    profile = generate_profile(tier="intermediate", seed=42)
    gen = MIDIGenerator(seed=42)
    events = gen._generate_ideal_events(rudiment, tempo_bpm=100, num_cycles=1)

    # Apply articulations (which should expand buzz strokes)
    proc = ArticulationProcessor(seed=42)
    processed = proc.process(events, rudiment, profile)

    # Should have more events than the 8 primaries (3-8 sub-strokes each)
    assert len(processed) > len(events)

    # Sub-strokes should be tightly spaced (2-5ms apart)
    buzz_events = [e for e in processed if e.stroke_type == StrokeType.BUZZ]
    for i in range(1, len(buzz_events)):
        if buzz_events[i].parent_stroke_index == buzz_events[i - 1].parent_stroke_index:
            spacing = buzz_events[i].actual_time_ms - buzz_events[i - 1].actual_time_ms
            assert 1 <= spacing <= 8, f"Sub-stroke spacing {spacing}ms out of range"

def test_buzz_marking_mode_no_expansion(self, definitions_dir):
    """marking mode should NOT expand strokes, just tag them as BUZZ."""
    from dataset_gen.midi_gen.articulations import ArticulationProcessor

    rudiment = load_rudiment(definitions_dir / "04_multiple_bounce_roll.yaml")
    # Temporarily override buzz_detail
    rudiment.params.buzz_detail = "marking"
    profile = generate_profile(tier="intermediate", seed=42)
    gen = MIDIGenerator(seed=42)
    events = gen._generate_ideal_events(rudiment, tempo_bpm=100, num_cycles=1)

    proc = ArticulationProcessor(seed=42)
    processed = proc.process(events, rudiment, profile)

    # Same number of events (no expansion)
    assert len(processed) == len(events)
    # Still tagged as BUZZ
    assert all(e.stroke_type == StrokeType.BUZZ for e in processed)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rudiments.py::TestBuzzGeneration -v`
Expected: `test_buzz_sub_strokes_generated` FAILS — no expansion happens.

**Step 3: Implement buzz sub-stroke expansion**

Replace `_process_buzz_roll` in `dataset_gen/midi_gen/articulations.py` (lines 261-282):

```python
    def _process_buzz_roll(
        self,
        events: list[StrokeEvent],
        dims: RudimentSpecificDimensions,
        params,
    ) -> list[StrokeEvent]:
        """
        Process buzz roll events.

        In sub_strokes mode, expands each BUZZ primary into rapid sub-strokes.
        In marking mode, just applies velocity jitter.
        """
        if params.buzz_detail == "sub_strokes":
            return self._expand_buzz_sub_strokes(events, dims, params)

        # Default / marking mode: tag and jitter only
        density_consistency = dims.buzz_density_consistency
        for event in events:
            jitter = (1 - density_consistency) * 20
            velocity_jitter = self.rng.normal(0, jitter)
            event.actual_velocity = int(np.clip(event.actual_velocity + velocity_jitter, 1, 127))
        return events

    def _expand_buzz_sub_strokes(
        self,
        events: list[StrokeEvent],
        dims: RudimentSpecificDimensions,
        params,
    ) -> list[StrokeEvent]:
        """Expand each BUZZ stroke into rapid sub-strokes."""
        min_buzz, max_buzz = params.buzz_strokes_range or (3, 8)
        density = dims.buzz_density_consistency
        expanded = []

        for event in events:
            if event.stroke_type != StrokeType.BUZZ:
                expanded.append(event)
                continue

            # Number of sub-strokes: higher consistency → more consistent count
            mean_count = (min_buzz + max_buzz) / 2
            count_jitter = (1 - density) * (max_buzz - min_buzz) / 2
            num_sub = int(np.clip(
                self.rng.normal(mean_count, count_jitter),
                min_buzz, max_buzz,
            ))

            # Primary stroke
            event.parent_stroke_index = event.index
            expanded.append(event)

            # Sub-strokes: tightly spaced with velocity decay
            sub_spacing_ms = self.rng.uniform(2, 5)
            decay = params.buzz_velocity_decay if hasattr(params, "buzz_velocity_decay") else 0.85
            for s in range(1, num_sub):
                sub_event = StrokeEvent(
                    index=event.index,  # Share parent index
                    hand=event.hand,
                    stroke_type=StrokeType.BUZZ,
                    intended_time_ms=event.intended_time_ms + s * sub_spacing_ms,
                    actual_time_ms=event.actual_time_ms + s * sub_spacing_ms + self.rng.normal(0, 0.5),
                    intended_velocity=int(event.intended_velocity * (decay ** s)),
                    actual_velocity=int(event.actual_velocity * (decay ** s)),
                    is_grace_note=False,
                    parent_stroke_index=event.index,
                )
                expanded.append(event)  # BUG: should be sub_event — see note below
                expanded.append(sub_event)

        return expanded
```

**IMPORTANT:** The above has an intentional placeholder note. The correct final line in the inner loop is just `expanded.append(sub_event)` — NOT `expanded.append(event)` followed by `expanded.append(sub_event)`. Only append `sub_event`.

Corrected inner loop body:

```python
                expanded.append(sub_event)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rudiments.py::TestBuzzGeneration -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass. If label computation breaks on BUZZ strokes, fix in Task 7.

**Step 6: Commit**

```
git add dataset_gen/midi_gen/articulations.py tests/test_rudiments.py
git commit -m "feat: implement buzz sub-stroke expansion in articulation processor"
```

---

### Task 7: Handle BUZZ in label computation

**Files:**
- Modify: `dataset_gen/labels/compute.py:229` (tap_strokes filter)
- Test: `tests/test_rudiments.py`

**Step 1: Write the failing test**

Add to `TestBuzzGeneration`:

```python
def test_buzz_strokes_included_in_velocity_control(self, definitions_dir):
    """BUZZ strokes should be included in velocity metrics, not just taps."""
    from dataset_gen.labels.compute import compute_stroke_labels, compute_measure_scores

    rudiment = load_rudiment(definitions_dir / "04_multiple_bounce_roll.yaml")
    profile = generate_profile(tier="intermediate", seed=42)
    gen = MIDIGenerator(seed=42)
    sample = gen.generate(rudiment, profile, tempo_bpm=100, num_cycles=4)

    stroke_labels = compute_stroke_labels(sample.events, rudiment)
    # BUZZ strokes should appear in labels
    buzz_labels = [s for s in stroke_labels if s.stroke_type == "buzz"]
    assert len(buzz_labels) > 0

    # Measure scores should compute without error
    measure_scores = compute_measure_scores(stroke_labels, rudiment, sample.events)
    assert len(measure_scores) > 0
    for m in measure_scores:
        assert 0 <= m.velocity_control <= 100
```

**Step 2: Run test to verify behavior**

Run: `pytest tests/test_rudiments.py::TestBuzzGeneration::test_buzz_strokes_included_in_velocity_control -v`

This may pass already since `stroke_type` is stored as a string value and velocity_control uses all strokes. If it fails, fix the specific issue (e.g., if `tap_strokes` filter at line 229 should also include buzz strokes for accent differentiation fallback).

**Step 3: Fix if needed**

In `dataset_gen/labels/compute.py` line 229, if accent differentiation breaks because there are no tap strokes in a buzz roll:

```python
    tap_strokes = [s for s in stroke_labels if s.stroke_type in ("tap", "buzz")]
```

**Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass.

**Step 5: Commit**

```
git add dataset_gen/labels/compute.py tests/test_rudiments.py
git commit -m "fix: include BUZZ strokes in velocity metrics for label computation"
```

---

### Task 8: Full validation

**Step 1: Run complete test suite**

Run: `pytest tests/ -v`
Expected: All 199+ tests pass.

**Step 2: Run linting**

Run: `ruff check dataset_gen/ && black --check dataset_gen/`
Expected: Clean.

**Step 3: Generate small dataset and validate**

Run:
```bash
python scripts/generate_dataset.py --preset small
python scripts/check_generation.py output/dataset
```
Expected: All checks pass.

**Step 4: Verify the three rudiments specifically**

Manually inspect generated samples for `double_drag_tap`, `swiss_army_triplet`, and `multiple_bounce_roll` to confirm:
- Double drag tap has 12 strokes per cycle
- Swiss army triplet sticking is R L L | L R R
- Multiple bounce roll has buzz sub-strokes in MIDI

**Step 5: Commit any validation fixes**

If validation reveals issues, fix and commit.
