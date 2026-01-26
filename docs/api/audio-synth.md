# Audio Synthesis Module

The audio synthesis module provides a FluidSynth wrapper for rendering MIDI performances to audio using SF2 soundfonts. It supports multiple soundfonts for different drum sounds (practice pad, marching snare, drum kits) and configurable synthesis parameters.

Key features:

- Real-time MIDI event processing
- Support for multiple loaded soundfonts
- Configurable reverb and chorus effects
- Output to numpy arrays or audio files (WAV, FLAC)

**Note**: This module requires FluidSynth to be installed on the system:

```bash
# macOS
brew install fluid-synth && pip install pyfluidsynth

# Linux
apt install fluidsynth && pip install pyfluidsynth
```

::: dataset_gen.audio_synth.synthesizer
    options:
      show_root_heading: false
      members:
        - SynthConfig
        - SoundfontInfo
        - AudioSynthesizer
        - render_midi_to_audio
