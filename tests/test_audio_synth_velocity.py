"""Tests for post-synthesis velocity gain curve."""

import numpy as np

from dataset_gen.midi_gen.generator import StrokeEvent, StrokeType, Hand


class TestApplyVelocityCurve:
    def _make_stroke(self, time_ms, velocity, stroke_type=StrokeType.TAP):
        return StrokeEvent(
            index=0,
            hand=Hand.RIGHT,
            stroke_type=stroke_type,
            intended_time_ms=time_ms,
            actual_time_ms=time_ms,
            intended_velocity=velocity,
            actual_velocity=velocity,
        )

    def test_louder_velocity_produces_higher_amplitude(self):
        """vel=120 region should be louder than vel=40 region."""
        from dataset_gen.audio_synth.synthesizer import apply_velocity_curve

        sr = 44100
        # 1 second of uniform noise (simulating flat FluidSynth output)
        audio = np.random.default_rng(42).uniform(-0.5, 0.5, (sr, 2))

        strokes = [
            self._make_stroke(100, 40),  # quiet at 100ms
            self._make_stroke(500, 120),  # loud at 500ms
        ]

        result = apply_velocity_curve(audio, strokes, sr)

        # Measure RMS in each region
        quiet_rms = np.sqrt(np.mean(result[int(0.15 * sr) : int(0.45 * sr)] ** 2))
        loud_rms = np.sqrt(np.mean(result[int(0.55 * sr) : int(0.85 * sr)] ** 2))

        assert (
            loud_rms > quiet_rms * 1.5
        ), f"Loud region ({loud_rms:.3f}) should be >1.5x quiet region ({quiet_rms:.3f})"

    def test_gain_follows_velocity_curve(self):
        """Gain should follow linear velocity mapping."""
        from dataset_gen.audio_synth.synthesizer import apply_velocity_curve

        sr = 44100
        audio = np.ones((sr, 2)) * 0.5  # Constant amplitude

        strokes = [self._make_stroke(0, 127)]
        result_loud = apply_velocity_curve(audio, strokes, sr)

        strokes = [self._make_stroke(0, 64)]
        result_mid = apply_velocity_curve(audio, strokes, sr)

        # vel=127 should produce ~2x the amplitude of vel=64
        ratio = np.mean(np.abs(result_loud)) / np.mean(np.abs(result_mid))
        assert 1.5 < ratio < 2.5, f"Amplitude ratio {ratio:.2f} not in expected range"

    def test_empty_strokes_returns_unchanged(self):
        """No strokes means no gain change."""
        from dataset_gen.audio_synth.synthesizer import apply_velocity_curve

        audio = np.random.default_rng(0).uniform(-0.5, 0.5, (44100, 2))
        result = apply_velocity_curve(audio, [], 44100)
        np.testing.assert_array_equal(result, audio)

    def test_mono_audio(self):
        """Should handle mono input."""
        from dataset_gen.audio_synth.synthesizer import apply_velocity_curve

        audio = np.ones(44100) * 0.5
        strokes = [self._make_stroke(0, 80)]
        result = apply_velocity_curve(audio, strokes, 44100)
        assert result.shape == audio.shape

    def test_no_clipping(self):
        """Output should not exceed [-1, 1]."""
        from dataset_gen.audio_synth.synthesizer import apply_velocity_curve

        audio = np.ones((44100, 2)) * 0.9
        strokes = [self._make_stroke(0, 127)]
        result = apply_velocity_curve(audio, strokes, 44100)
        assert np.max(np.abs(result)) <= 1.0
