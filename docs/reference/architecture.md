# Pipeline Architecture

This document describes SOUSA's data generation pipeline architecture, including data flow, class relationships, and module organization.

## Pipeline Overview

The generation pipeline transforms rudiment definitions and player profiles into labeled audio samples through seven stages:

```mermaid
flowchart LR
    subgraph Input
        A[Rudiment YAML] --> B[Loader]
        C[Profile Config] --> D[Generator]
    end

    subgraph Generation
        B --> E[StickingPattern]
        D --> F[PlayerProfile]
        E --> G[MIDI Generator]
        F --> G
        G --> H[StrokeEvents]
    end

    subgraph Audio
        H --> I[FluidSynth]
        I --> J[Raw Audio]
        J --> K[Room Sim]
        K --> L[Mic Sim]
        L --> M[Chain/EQ]
        M --> N[Degradation]
    end

    subgraph Labels
        H --> O[Stroke Labels]
        O --> P[Measure Labels]
        P --> Q[Exercise Scores]
    end

    subgraph Output
        N --> R[FLAC Audio]
        G --> S[MIDI File]
        Q --> T[Parquet Labels]
    end
```

## Detailed Pipeline Stages

### Stage 1: Rudiment Loading

```mermaid
flowchart TD
    A[YAML Files] --> B[yaml.safe_load]
    B --> C[Pydantic Validation]
    C --> D[Rudiment Object]

    subgraph Validation
        C --> C1{Valid Schema?}
        C1 -->|Yes| D
        C1 -->|No| E[ValidationError]
    end
```

**Module**: `dataset_gen/rudiments/loader.py`

The loader reads YAML definitions from `dataset_gen/rudiments/definitions/` and validates them against Pydantic schemas. Each rudiment file specifies:

- Pattern structure (strokes with hand, type, timing)
- Subdivision and tempo range
- Category-specific parameters

### Stage 2: Profile Generation

```mermaid
flowchart TD
    A[SkillTier] --> B[ARCHETYPE_PARAMS]
    B --> C[Gaussian Sampling]
    C --> D[Dimension Clamping]
    D --> E[PlayerProfile]

    subgraph Dimensions
        C --> C1[TimingDimensions]
        C --> C2[DynamicsDimensions]
        C --> C3[HandBalanceDimensions]
        C --> C4[RudimentSpecificDimensions]
    end
```

**Module**: `dataset_gen/profiles/archetypes.py`

Profiles are generated from archetype parameters with Gaussian sampling. Each skill tier (beginner, intermediate, advanced, professional) has distinct parameter ranges derived from percussion research literature.

### Stage 3: MIDI Generation

```mermaid
flowchart TD
    A[Rudiment + Profile] --> B[Ideal Events]
    B --> C[Apply Deviations]

    subgraph Deviations
        C --> D[Timing Error]
        C --> E[Tempo Drift]
        C --> F[Hand Bias]
        C --> G[Fatigue Effect]
        C --> H[Velocity Variance]
    end

    D --> I[StrokeEvents]
    E --> I
    F --> I
    G --> I
    H --> I

    I --> J[MIDI Bytes]
```

**Module**: `dataset_gen/midi_gen/generator.py`

The MIDI generator:

1. Creates ideal stroke events from the rudiment pattern
2. Applies player-specific deviations based on profile dimensions
3. Encodes events as Standard MIDI Format bytes

### Stage 4: Audio Synthesis

```mermaid
flowchart TD
    A[MIDI Bytes] --> B[FluidSynth]
    B --> C{Soundfont}
    C --> D[GeneralUser GS]
    C --> E[Marching Snare]
    C --> F[MT Power Drums]
    C --> G[Other SF2]

    D --> H[Raw Audio]
    E --> H
    F --> H
    G --> H
```

**Module**: `dataset_gen/audio_synth/synthesizer.py`

FluidSynth renders MIDI to audio using SF2 soundfonts. Multiple soundfonts provide timbral variety across practice pad, marching snare, and drum kit sounds.

### Stage 5: Audio Augmentation

```mermaid
flowchart TD
    A[Raw Audio] --> B[Room Simulation]
    B --> C[Mic Simulation]
    C --> D[Recording Chain]
    D --> E[Degradation]
    E --> F[Augmented Audio]

    subgraph Room
        B --> B1[IR Convolution]
        B --> B2[Wet/Dry Mix]
    end

    subgraph Mic
        C --> C1[Frequency Response]
        C --> C2[Proximity Effect]
        C --> C3[Distance Rolloff]
    end

    subgraph Chain
        D --> D1[Preamp]
        D --> D2[Compression]
        D --> D3[EQ]
    end

    subgraph Degrade
        E --> E1[Bit Depth]
        E --> E2[Sample Rate]
        E --> E3[Noise]
    end
```

**Module**: `dataset_gen/audio_aug/pipeline.py`

The augmentation pipeline applies realistic recording conditions through four stages, each with configurable parameters.

### Stage 6: Label Computation

```mermaid
flowchart TD
    A[StrokeEvents] --> B[compute_stroke_labels]
    B --> C[StrokeLabel list]
    C --> D[compute_measure_labels]
    D --> E[MeasureLabel list]
    C --> F[compute_exercise_scores]
    E --> F
    F --> G[ExerciseScores]

    C --> H[Sample]
    E --> H
    G --> H
```

**Module**: `dataset_gen/labels/compute.py`

Labels are computed hierarchically:

1. **Stroke level**: Individual timing/velocity errors
2. **Measure level**: Aggregate statistics per measure
3. **Exercise level**: Composite performance scores

### Stage 7: Storage

```mermaid
flowchart TD
    A[Sample Objects] --> B[DataFrame Conversion]
    B --> C[Parquet Writer]

    C --> D[samples.parquet]
    C --> E[exercises.parquet]
    C --> F[measures.parquet]
    C --> G[strokes.parquet]

    H[MIDI Bytes] --> I[midi/*.mid]
    J[Audio Array] --> K[audio/*.flac]
```

**Module**: `dataset_gen/pipeline/storage.py`

Final outputs are written to:

- Parquet files for structured labels
- MIDI files for symbolic data
- FLAC files for lossless audio

---

## Data Structure Relationships

### Class Diagram

```mermaid
classDiagram
    class Sample {
        +str sample_id
        +str profile_id
        +str rudiment_slug
        +int tempo_bpm
        +float duration_sec
        +int num_cycles
        +str skill_tier
        +str dominant_hand
        +List~StrokeLabel~ strokes
        +List~MeasureLabel~ measures
        +ExerciseScores exercise_scores
        +AudioAugmentation audio_augmentation
    }

    class StrokeLabel {
        +int index
        +str hand
        +str stroke_type
        +float intended_time_ms
        +float actual_time_ms
        +float timing_error_ms
        +int intended_velocity
        +int actual_velocity
        +int velocity_error
        +bool is_grace_note
        +bool is_accent
        +int diddle_position
        +float flam_spacing_ms
        +int parent_stroke_index
    }

    class MeasureLabel {
        +int index
        +int stroke_start
        +int stroke_end
        +float timing_mean_error_ms
        +float timing_std_ms
        +float timing_max_error_ms
        +float velocity_mean
        +float velocity_std
        +float velocity_consistency
        +float lr_velocity_ratio
        +float lr_timing_diff_ms
    }

    class ExerciseScores {
        +float timing_accuracy
        +float timing_consistency
        +float tempo_stability
        +float subdivision_evenness
        +float velocity_control
        +float accent_differentiation
        +float accent_accuracy
        +float hand_balance
        +float weak_hand_index
        +float flam_quality
        +float diddle_quality
        +float roll_sustain
        +float groove_feel_proxy
        +float overall_score
        +float tier_confidence
    }

    class AudioAugmentation {
        +str soundfont
        +str room_type
        +float room_wet_dry
        +float mic_distance
        +str mic_type
        +float compression_ratio
        +float noise_level_db
        +int bit_depth
        +int sample_rate
    }

    Sample "1" *-- "many" StrokeLabel : contains
    Sample "1" *-- "many" MeasureLabel : contains
    Sample "1" *-- "1" ExerciseScores : contains
    Sample "1" *-- "0..1" AudioAugmentation : contains
    MeasureLabel --> StrokeLabel : references via stroke_start/end
```

### Rudiment Structure

```mermaid
classDiagram
    class Rudiment {
        +str name
        +str slug
        +RudimentCategory category
        +StickingPattern pattern
        +Subdivision subdivision
        +Tuple tempo_range
        +RudimentParams params
        +int pas_number
        +str description
        +bool starts_on_left
    }

    class StickingPattern {
        +List~Stroke~ strokes
        +float beats_per_cycle
        +stroke_count()
        +accent_positions()
        +grace_note_positions()
    }

    class Stroke {
        +Hand hand
        +StrokeType stroke_type
        +float grace_offset
        +int diddle_position
        +is_accented()
        +is_grace_note()
    }

    class RudimentParams {
        +Tuple flam_spacing_range
        +Tuple diddle_ratio_range
        +str roll_type
        +int roll_strokes_per_beat
        +Tuple buzz_strokes_range
        +Tuple drag_spacing_range
    }

    Rudiment "1" *-- "1" StickingPattern : contains
    Rudiment "1" *-- "1" RudimentParams : contains
    StickingPattern "1" *-- "many" Stroke : contains
```

### Player Profile Structure

```mermaid
classDiagram
    class PlayerProfile {
        +str id
        +SkillTier skill_tier
        +ExecutionDimensions dimensions
        +str dominant_hand
        +float fatigue_coefficient
        +Tuple tempo_comfort_range
        +get_tempo_penalty()
    }

    class ExecutionDimensions {
        +TimingDimensions timing
        +DynamicsDimensions dynamics
        +HandBalanceDimensions hand_balance
        +RudimentSpecificDimensions rudiment_specific
    }

    class TimingDimensions {
        +float timing_accuracy
        +float timing_consistency
        +float tempo_drift
        +float subdivision_evenness
    }

    class DynamicsDimensions {
        +float velocity_mean
        +float velocity_variance
        +float accent_differentiation
        +float accent_accuracy
    }

    class HandBalanceDimensions {
        +float lr_velocity_ratio
        +float lr_timing_bias
        +float lr_consistency_delta
    }

    class RudimentSpecificDimensions {
        +float flam_spacing
        +float flam_spacing_variance
        +float diddle_evenness
        +float diddle_variance
        +float roll_sustain
        +float buzz_density_consistency
    }

    PlayerProfile "1" *-- "1" ExecutionDimensions : contains
    ExecutionDimensions "1" *-- "1" TimingDimensions
    ExecutionDimensions "1" *-- "1" DynamicsDimensions
    ExecutionDimensions "1" *-- "1" HandBalanceDimensions
    ExecutionDimensions "1" *-- "1" RudimentSpecificDimensions
```

---

## Module Organization

```
dataset_gen/
├── rudiments/
│   ├── definitions/     # 40 YAML files
│   ├── schema.py        # Pydantic models
│   └── loader.py        # YAML loading
│
├── profiles/
│   ├── archetypes.py    # Profile generation
│   └── sampler.py       # Batch sampling
│
├── midi_gen/
│   ├── generator.py     # MIDI generation
│   └── articulations.py # Articulation handling
│
├── audio_synth/
│   └── synthesizer.py   # FluidSynth wrapper
│
├── audio_aug/
│   ├── room.py          # Room simulation
│   ├── mic.py           # Mic simulation
│   ├── chain.py         # Recording chain
│   ├── degradation.py   # Quality degradation
│   └── pipeline.py      # Augmentation orchestration
│
├── labels/
│   ├── schema.py        # Label Pydantic models
│   ├── compute.py       # Score computation
│   └── groove.py        # Groove metrics
│
├── pipeline/
│   ├── generate.py      # Main orchestration
│   ├── parallel.py      # Multiprocessing
│   ├── checkpoint.py    # Resumable generation
│   ├── storage.py       # Parquet writing
│   └── splits.py        # Train/val/test splits
│
├── validation/
│   ├── verify.py        # Data integrity checks
│   ├── realism.py       # Literature validation
│   └── report.py        # Report generation
│
└── hub/
    └── uploader.py      # HuggingFace upload
```

---

## Parallel Processing

```mermaid
flowchart TD
    A[Sample Tasks] --> B[Task Queue]
    B --> C1[Worker 1]
    B --> C2[Worker 2]
    B --> C3[Worker N]

    C1 --> D[Result Queue]
    C2 --> D
    C3 --> D

    D --> E[Checkpoint Manager]
    E --> F[Storage Writer]

    subgraph Checkpoint
        E --> G[Progress File]
        E --> H[Partial Results]
    end
```

**Module**: `dataset_gen/pipeline/parallel.py`

The pipeline supports:

- **Multiprocessing**: CPU-parallel sample generation
- **Checkpointing**: Resume interrupted generation
- **Batch writing**: Efficient Parquet append operations

---

## Data Flow Summary

| Stage | Input | Output | Key Module |
|-------|-------|--------|------------|
| Load | YAML files | `Rudiment` objects | `rudiments/loader.py` |
| Profile | Skill tier | `PlayerProfile` objects | `profiles/archetypes.py` |
| MIDI | Rudiment + Profile | `StrokeEvent` list + MIDI bytes | `midi_gen/generator.py` |
| Synth | MIDI bytes | Raw audio array | `audio_synth/synthesizer.py` |
| Augment | Raw audio | Augmented audio | `audio_aug/pipeline.py` |
| Label | StrokeEvents | `Sample` with all labels | `labels/compute.py` |
| Store | Samples | Parquet + MIDI + FLAC files | `pipeline/storage.py` |

## See Also

- [Rudiment Schema](rudiment-schema.md) - YAML format specification
- [Score Computation](score-computation.md) - Scoring algorithms
- [Audio Processing](audio-processing.md) - Augmentation details
