# Audio Augmentation Module

The audio augmentation module provides a comprehensive pipeline for adding realistic recording characteristics to synthesized audio. This increases dataset diversity and helps models generalize to real-world recording conditions.

The augmentation chain includes:

- **Room simulation**: Convolution reverb with impulse responses
- **Microphone modeling**: Frequency response curves and positioning effects
- **Recording chain**: Preamp saturation, compression, and EQ
- **Degradation**: Noise injection, bit depth reduction, sample rate artifacts

## Recording Chain

Simulates analog recording chain including preamp coloration, dynamics compression, and master EQ.

::: dataset_gen.audio_aug.chain
    options:
      show_root_heading: false
      members:
        - PreampType
        - PreampConfig
        - CompressorConfig
        - EQConfig
        - ChainConfig
        - RecordingChain
        - apply_chain

## Room Simulation

Applies realistic room acoustics using convolution reverb with impulse responses or synthetic reverb generation.

::: dataset_gen.audio_aug.room
    options:
      show_root_heading: false
      members:
        - RoomType
        - RoomConfig
        - RoomSimulator
        - apply_room

## Microphone Modeling

Simulates different microphone types, positions, and distances with characteristic frequency responses.

::: dataset_gen.audio_aug.mic
    options:
      show_root_heading: false
      members:
        - MicType
        - MicPosition
        - MicConfig
        - MicSimulator
        - apply_mic

## Audio Degradation

Adds various degradations including noise, bit depth reduction, sample rate artifacts, and tape-style effects.

::: dataset_gen.audio_aug.degradation
    options:
      show_root_heading: false
      members:
        - NoiseType
        - NoiseConfig
        - BitDepthConfig
        - SampleRateConfig
        - DegradationConfig
        - AudioDegrader
        - add_noise
