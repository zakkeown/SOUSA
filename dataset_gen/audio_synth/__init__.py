"""Audio synthesis module using FluidSynth."""

from dataset_gen.audio_synth.synthesizer import (
    AudioSynthesizer,
    SynthConfig,
    render_midi_to_audio,
)

__all__ = [
    "AudioSynthesizer",
    "SynthConfig",
    "render_midi_to_audio",
]
