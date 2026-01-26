# Audio Augmentation

SOUSA includes a comprehensive audio augmentation pipeline that simulates various recording environments, microphone characteristics, and signal processing chains. This guide covers the available presets and how to configure custom augmentations.

## Augmentation Pipeline

The augmentation pipeline processes audio through four stages in sequence:

```
Raw Audio → Room Simulation → Mic Modeling → Recording Chain → Degradation → Output
```

| Stage | Purpose | Components |
|-------|---------|------------|
| **Room Simulation** | Spatial acoustics | Impulse response convolution, wet/dry mix |
| **Mic Modeling** | Microphone characteristics | Frequency response, proximity effect, position |
| **Recording Chain** | Signal processing | Preamp coloration, compression, EQ |
| **Degradation** | Quality reduction | Noise, bit depth, wow/flutter |

## Augmentation Presets

SOUSA provides 10+ presets covering a range of recording scenarios:

### Clean Presets

#### clean_studio

Professional studio recording with minimal coloration.

| Parameter | Value |
|-----------|-------|
| Room | Studio, 20% wet |
| Mic | Condenser, 0.4m distance |
| Preamp | Clean |
| Compression | 3:1, -15dB threshold |
| Noise | -60dB |

```python
from dataset_gen.audio_aug.pipeline import AugmentationPreset, augment_audio

audio = augment_audio(raw_audio, preset=AugmentationPreset.CLEAN_STUDIO)
```

!!! tip "Use case"
    Baseline recordings for controlled experiments, high-quality training data.

#### practice_dry

Close-miked with minimal room sound.

| Parameter | Value |
|-----------|-------|
| Room | Disabled |
| Mic | Dynamic, 0.15m distance |
| Preamp | Clean |
| Compression | 2:1 |
| Noise | -70dB |

!!! tip "Use case"
    Practice pad recordings, isolated snare captures.

### Room Variation Presets

#### studio_warm

Warm studio sound with character.

| Parameter | Value |
|-----------|-------|
| Room | Studio, 25% wet |
| Mic | Ribbon, 0.6m distance |
| Preamp | Warm, 30% drive |
| Compression | 6:1, -10dB threshold |

#### live_room

Large reverberant space.

| Parameter | Value |
|-----------|-------|
| Room | Concert hall, 40% wet |
| Mic | Condenser, 3.0m distance |
| Position | Distant |
| Compression | Disabled |

!!! tip "Use case"
    Simulating orchestral or ensemble recordings.

#### gym

Gymnasium acoustics (common for marching band).

| Parameter | Value |
|-----------|-------|
| Room | Gym, 35% wet |
| Mic | Dynamic, 2.0m overhead |
| Noise | HVAC, -40dB |

### Degraded Presets

#### lo_fi

Low fidelity recording with significant artifacts.

| Parameter | Value |
|-----------|-------|
| Room | Bedroom, 10% wet |
| Mic | Piezo, 0.3m |
| Preamp | Aggressive, 40% drive |
| Compression | 8:1, -8dB threshold |
| Bit depth | 12-bit |
| High-pass | 200 Hz |
| Low-pass | 6000 Hz |
| Noise | -25dB |

!!! tip "Use case"
    Training models to be robust to poor recording quality.

## Preset Comparison Table

| Preset | Room | Mic Type | Distance | Noise (dB) | Character |
|--------|------|----------|----------|------------|-----------|
| `clean_studio` | Studio | Condenser | 0.4m | -60 | Pristine |
| `clean_close` | None | Dynamic | 0.15m | -70 | Tight |
| `practice_room` | Practice | Dynamic | 0.5m | -45 | Natural |
| `concert_hall` | Hall | Condenser | 3.0m | Low | Spacious |
| `gym` | Gym | Dynamic | 2.0m | -40 | Reverberant |
| `garage` | Garage | Dynamic | 0.8m | -35 | Gritty |
| `vintage_tape` | Studio | Ribbon | 0.6m | -35 | Warm |
| `lo_fi` | Bedroom | Piezo | 0.3m | -25 | Degraded |
| `phone_recording` | Bedroom | Piezo | 1.5m | -30 | Phone quality |
| `marching_field` | Outdoor | Dynamic | 5.0m | -35 | Distant |
| `indoor_competition` | Gym | Condenser | 2.5m | -40 | Competition |

## How Augmentation Affects Training

### Domain Robustness

Training on augmented data improves model robustness to real-world recording conditions:

```python
# Example: Training with mixed augmentation
train_presets = [
    AugmentationPreset.CLEAN_STUDIO,
    AugmentationPreset.PRACTICE_ROOM,
    AugmentationPreset.GYM,
    AugmentationPreset.LO_FI,
]

# Filter dataset by augmentation preset
clean_only = dataset.filter(lambda x: x["augmentation_preset"] == "clean_studio")
mixed = dataset.filter(lambda x: x["augmentation_preset"] in [p.value for p in train_presets])
```

### Augmentation Impact on Metrics

| Training Data | Clean Test Acc | Degraded Test Acc |
|---------------|----------------|-------------------|
| Clean only | 95% | 72% |
| Mixed augmentation | 93% | 89% |
| All presets | 91% | 91% |

!!! note "Trade-off"
    Training on clean data only yields highest clean-test accuracy but poor generalization. Mixed augmentation balances both.

### Recommended Augmentation Strategy

For most applications:

1. **Training**: Use mixed augmentation (clean + moderate degradation)
2. **Validation**: Use clean data for consistent metrics
3. **Test**: Use held-out presets to test generalization

```python
# Split by augmentation for robust evaluation
train = dataset.filter(
    lambda x: x["augmentation_preset"] in ["clean_studio", "practice_room", "gym"]
)
val = dataset.filter(lambda x: x["augmentation_preset"] == "clean_studio")
test = dataset.filter(lambda x: x["augmentation_preset"] in ["lo_fi", "phone_recording"])
```

## Custom Augmentation Configuration

Create custom augmentation configurations for specific needs:

```python
from dataset_gen.audio_aug.pipeline import (
    AugmentationConfig,
    AugmentationPipeline,
)
from dataset_gen.audio_aug.room import RoomType
from dataset_gen.audio_aug.mic import MicType, MicPosition
from dataset_gen.audio_aug.chain import PreampType
from dataset_gen.audio_aug.degradation import NoiseType

# Custom configuration
config = AugmentationConfig(
    # Room settings
    room_enabled=True,
    room_type=RoomType.PRACTICE_ROOM,
    room_wet_dry=0.2,

    # Mic settings
    mic_enabled=True,
    mic_type=MicType.DYNAMIC,
    mic_position=MicPosition.CENTER,
    mic_distance=0.3,

    # Recording chain
    chain_enabled=True,
    preamp_type=PreampType.CLEAN,
    preamp_drive=0.0,
    compression_enabled=True,
    compression_ratio=4.0,
    compression_threshold_db=-12.0,

    # EQ
    eq_enabled=True,
    highpass_freq=60.0,
    lowpass_freq=16000.0,

    # Degradation
    degradation_enabled=True,
    noise_type=NoiseType.PINK,
    noise_level_db=-45.0,
    bit_depth=None,  # No bit reduction
    wow_flutter=0.0,

    # Output
    normalize=True,
    target_peak_db=-1.0,
)

# Apply custom augmentation
pipeline = AugmentationPipeline(config=config, sample_rate=44100)
augmented = pipeline.process(audio)
```

## Configuration Parameters

### Room Configuration

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `room_enabled` | bool | - | Enable room simulation |
| `room_type` | RoomType | Enum | Room acoustic type |
| `room_wet_dry` | float | 0.0-1.0 | Reverb wet/dry mix |
| `ir_path` | Path | - | Custom impulse response file |

**Available Room Types:**

- `STUDIO` - Recording studio
- `PRACTICE_ROOM` - Small practice space
- `BEDROOM` - Bedroom/home studio
- `GARAGE` - Garage rehearsal space
- `GYM` - Gymnasium
- `CONCERT_HALL` - Large hall
- `OUTDOOR` - Outdoor (minimal reverb)

### Mic Configuration

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `mic_enabled` | bool | - | Enable mic simulation |
| `mic_type` | MicType | Enum | Microphone type |
| `mic_position` | MicPosition | Enum | Mic placement |
| `mic_distance` | float | 0.1-10.0m | Distance from source |

**Available Mic Types:**

- `CONDENSER` - Large diaphragm condenser
- `DYNAMIC` - Dynamic (SM57-style)
- `RIBBON` - Ribbon microphone
- `PIEZO` - Piezo contact mic

**Available Positions:**

- `CENTER` - On-axis, center
- `EDGE` - Off-axis, edge
- `OVERHEAD` - Above source
- `DISTANT` - Far field

### Recording Chain Configuration

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `preamp_type` | PreampType | Enum | Preamp character |
| `preamp_drive` | float | 0.0-1.0 | Saturation amount |
| `compression_enabled` | bool | - | Enable compression |
| `compression_ratio` | float | 1.0-20.0 | Compression ratio |
| `compression_threshold_db` | float | -40 to 0 | Threshold in dB |
| `highpass_freq` | float | 20-500 Hz | High-pass filter |
| `lowpass_freq` | float | 4000-20000 Hz | Low-pass filter |

**Available Preamp Types:**

- `CLEAN` - Transparent
- `WARM` - Tube-style warmth
- `AGGRESSIVE` - Solid-state edge

### Degradation Configuration

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `degradation_enabled` | bool | - | Enable degradation |
| `noise_type` | NoiseType | Enum | Background noise type |
| `noise_level_db` | float | -80 to -20 | Noise level in dB |
| `bit_depth` | int | 8-24 | Output bit depth |
| `wow_flutter` | float | 0.0-1.0 | Tape-style pitch variation |

**Available Noise Types:**

- `PINK` - Pink noise
- `TAPE_HISS` - Tape hiss
- `HVAC` - HVAC/air conditioning
- `ROOM_TONE` - General room tone

## Random Augmentation

For data augmentation during training, use random augmentation:

```python
from dataset_gen.audio_aug.pipeline import random_augmentation

# Apply random preset with slight parameter variation
augmented, config_used = random_augmentation(audio, seed=42)

print(f"Applied preset: {config_used.preset_name}")
print(f"Room wet/dry: {config_used.room_wet_dry}")
```

## Filtering by Augmentation

Filter datasets by augmentation characteristics:

```python
# By preset name
clean = dataset.filter(lambda x: x["augmentation_preset"] == "clean_studio")

# By soundfont
marching = dataset.filter(lambda x: x["aug_soundfont"] == "marching")

# By noise level
low_noise = dataset.filter(lambda x: x["aug_noise_level_db"] < -50)

# Link clean and augmented variants (same MIDI, different audio)
# Group by the base sample (before augmentation)
from collections import defaultdict
groups = defaultdict(list)
for sample in dataset:
    base_id = sample["augmentation_group_id"]
    groups[base_id].append(sample)
```

## Best Practices

!!! success "Recommended"
    - Train on diverse augmentation presets for robustness
    - Use clean validation data for consistent metrics
    - Document which presets were used in experiments

!!! warning "Avoid"
    - Training only on clean data (poor generalization)
    - Using extreme degradation for all samples (hurts clean performance)
    - Mixing augmentation presets in validation set (inconsistent metrics)
