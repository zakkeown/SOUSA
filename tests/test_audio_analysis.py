"""Tests for audio analysis module."""

from __future__ import annotations

import numpy as np
import pytest
import mido
import soundfile as sf

from dataset_gen.audio_analysis.onsets import detect_onsets, DetectedOnset
from dataset_gen.audio_analysis.midi_alignment import MidiOnset, extract_midi_onsets
from dataset_gen.audio_analysis.views import (
    render_waveform,
    render_onset_timeline,
    render_dashboard,
)


class TestDataTypes:
    def test_detected_onset_creation(self):
        onset = DetectedOnset(time_sec=0.5, strength=0.8)
        assert onset.time_sec == 0.5
        assert onset.strength == 0.8

    def test_midi_onset_creation(self):
        onset = MidiOnset(time_sec=0.5, velocity=100, note=38)
        assert onset.time_sec == 0.5
        assert onset.velocity == 100
        assert onset.note == 38


class TestMidiAlignment:
    @pytest.fixture
    def simple_midi_file(self, tmp_path):
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=500000))
        # Four quarter notes at 120 BPM = 500ms apart
        track.append(mido.Message("note_on", note=38, velocity=100, time=0))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240))
        track.append(mido.Message("note_on", note=38, velocity=60, time=240))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240))
        track.append(mido.Message("note_on", note=38, velocity=100, time=240))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240))
        track.append(mido.Message("note_on", note=38, velocity=60, time=240))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240))
        midi_path = tmp_path / "test.mid"
        mid.save(str(midi_path))
        return midi_path

    @pytest.fixture
    def flam_midi_file(self, tmp_path):
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=500000))
        track.append(mido.Message("note_on", note=38, velocity=40, time=0))
        track.append(mido.Message("note_off", note=38, velocity=0, time=14))
        track.append(mido.Message("note_on", note=38, velocity=110, time=15))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240))
        midi_path = tmp_path / "flam.mid"
        mid.save(str(midi_path))
        return midi_path

    def test_extract_basic_onsets(self, simple_midi_file):
        onsets = extract_midi_onsets(simple_midi_file)
        assert len(onsets) == 4
        assert all(o.note == 38 for o in onsets)
        assert [o.velocity for o in onsets] == [100, 60, 100, 60]

    def test_extract_onset_timing(self, simple_midi_file):
        onsets = extract_midi_onsets(simple_midi_file)
        for i in range(1, len(onsets)):
            gap_ms = (onsets[i].time_sec - onsets[i - 1].time_sec) * 1000
            assert abs(gap_ms - 500.0) < 5.0, f"Gap {i}: {gap_ms}ms, expected ~500ms"

    def test_extract_flam_timing(self, flam_midi_file):
        onsets = extract_midi_onsets(flam_midi_file)
        assert len(onsets) == 2
        gap_ms = (onsets[1].time_sec - onsets[0].time_sec) * 1000
        assert 25.0 < gap_ms < 35.0, f"Flam gap: {gap_ms}ms, expected ~30ms"

    def test_extract_from_path_string(self, simple_midi_file):
        onsets = extract_midi_onsets(str(simple_midi_file))
        assert len(onsets) == 4


class TestOnsetDetection:
    @pytest.fixture
    def click_audio(self, tmp_path):
        sr = 44100
        duration_sec = 2.5
        n_samples = int(sr * duration_sec)
        audio = np.zeros(n_samples)
        click_times = [0.25, 0.75, 1.25, 1.75]
        click_len = int(0.005 * sr)
        for t in click_times:
            start = int(t * sr)
            click = np.exp(-np.linspace(0, 10, click_len)) * 0.9
            audio[start : start + click_len] += click
        audio_path = tmp_path / "clicks.wav"
        sf.write(str(audio_path), audio, sr)
        return audio_path, click_times

    def test_detect_onsets_returns_list(self, click_audio):
        audio_path, _ = click_audio
        onsets = detect_onsets(audio_path)
        assert isinstance(onsets, list)
        assert all(isinstance(o, DetectedOnset) for o in onsets)

    def test_detect_onsets_finds_clicks(self, click_audio):
        audio_path, expected_times = click_audio
        onsets = detect_onsets(audio_path)
        assert 3 <= len(onsets) <= 6, f"Expected ~4 onsets, got {len(onsets)}"

    def test_detect_onsets_timing_accuracy(self, click_audio):
        audio_path, expected_times = click_audio
        onsets = detect_onsets(audio_path)
        detected_times = [o.time_sec for o in onsets]
        for expected in expected_times:
            closest = min(detected_times, key=lambda t: abs(t - expected))
            error_ms = abs(closest - expected) * 1000
            assert (
                error_ms < 50
            ), f"Onset at {expected}s: closest detection {closest}s ({error_ms}ms off)"

    def test_detect_onsets_from_array(self, click_audio):
        audio_path, _ = click_audio
        audio, sr = sf.read(str(audio_path))
        onsets = detect_onsets(audio, sample_rate=sr)
        assert len(onsets) >= 3


class TestWaveformView:
    @pytest.fixture
    def sample_audio_data(self):
        sr = 44100
        audio = np.zeros(sr)
        for t in [0.25, 0.75]:
            start = int(t * sr)
            click_len = int(0.005 * sr)
            audio[start : start + click_len] = 0.8 * np.exp(-np.linspace(0, 10, click_len))
        return audio, sr

    def test_render_waveform_creates_file(self, sample_audio_data, tmp_path):
        audio, sr = sample_audio_data
        output = tmp_path / "waveform.png"
        render_waveform(audio, sr, output_path=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_render_waveform_with_midi_onsets(self, sample_audio_data, tmp_path):
        audio, sr = sample_audio_data
        midi_onsets = [
            MidiOnset(time_sec=0.25, velocity=100, note=38),
            MidiOnset(time_sec=0.75, velocity=60, note=38),
        ]
        output = tmp_path / "waveform_midi.png"
        render_waveform(audio, sr, midi_onsets=midi_onsets, output_path=output)
        assert output.exists()

    def test_render_waveform_with_detected_onsets(self, sample_audio_data, tmp_path):
        audio, sr = sample_audio_data
        detected = [
            DetectedOnset(time_sec=0.251, strength=0.9),
            DetectedOnset(time_sec=0.749, strength=0.7),
        ]
        output = tmp_path / "waveform_detected.png"
        render_waveform(audio, sr, detected_onsets=detected, output_path=output)
        assert output.exists()

    def test_render_waveform_with_zoom(self, sample_audio_data, tmp_path):
        audio, sr = sample_audio_data
        output = tmp_path / "waveform_zoom.png"
        render_waveform(audio, sr, start_ms=200, end_ms=350, output_path=output)
        assert output.exists()


class TestOnsetTimelineView:
    def test_render_onset_timeline_creates_file(self, tmp_path):
        midi_onsets = [
            MidiOnset(time_sec=0.0, velocity=100, note=38),
            MidiOnset(time_sec=0.5, velocity=60, note=38),
            MidiOnset(time_sec=1.0, velocity=100, note=38),
            MidiOnset(time_sec=1.5, velocity=60, note=38),
        ]
        output = tmp_path / "onsets.png"
        render_onset_timeline(midi_onsets=midi_onsets, output_path=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_render_with_detected_onsets(self, tmp_path):
        midi_onsets = [
            MidiOnset(time_sec=0.0, velocity=100, note=38),
            MidiOnset(time_sec=0.5, velocity=60, note=38),
        ]
        detected = [
            DetectedOnset(time_sec=0.002, strength=0.9),
            DetectedOnset(time_sec=0.498, strength=0.6),
        ]
        output = tmp_path / "onsets_both.png"
        render_onset_timeline(
            midi_onsets=midi_onsets,
            detected_onsets=detected,
            output_path=output,
        )
        assert output.exists()

    def test_render_detected_only(self, tmp_path):
        detected = [
            DetectedOnset(time_sec=0.1, strength=0.9),
            DetectedOnset(time_sec=0.6, strength=0.5),
        ]
        output = tmp_path / "onsets_detected.png"
        render_onset_timeline(detected_onsets=detected, output_path=output)
        assert output.exists()


class TestDashboardView:
    @pytest.fixture
    def dashboard_data(self):
        sr = 44100
        audio = np.zeros(sr * 2)
        click_times = [0.25, 0.75, 1.25, 1.75]
        click_len = int(0.005 * sr)
        for t in click_times:
            start = int(t * sr)
            audio[start : start + click_len] = 0.8 * np.exp(-np.linspace(0, 10, click_len))
        midi_onsets = [
            MidiOnset(
                time_sec=t,
                velocity=100 if i % 2 == 0 else 60,
                note=38,
            )
            for i, t in enumerate(click_times)
        ]
        detected = [
            DetectedOnset(
                time_sec=t + 0.002,
                strength=0.9 if i % 2 == 0 else 0.5,
            )
            for i, t in enumerate(click_times)
        ]
        return audio, sr, midi_onsets, detected

    def test_render_dashboard_creates_file(self, dashboard_data, tmp_path):
        audio, sr, midi_onsets, detected = dashboard_data
        output = tmp_path / "dashboard.png"
        render_dashboard(
            audio,
            sr,
            midi_onsets=midi_onsets,
            detected_onsets=detected,
            output_path=output,
        )
        assert output.exists()
        assert output.stat().st_size > 1000

    def test_render_dashboard_audio_only(self, dashboard_data, tmp_path):
        audio, sr, _, _ = dashboard_data
        output = tmp_path / "dashboard_audio_only.png"
        render_dashboard(audio, sr, output_path=output)
        assert output.exists()

    def test_render_dashboard_with_activation(self, dashboard_data, tmp_path):
        audio, sr, midi_onsets, detected = dashboard_data
        activation = np.random.rand(200).astype(np.float32) * 0.3
        output = tmp_path / "dashboard_activation.png"
        render_dashboard(
            audio,
            sr,
            midi_onsets=midi_onsets,
            detected_onsets=detected,
            activation=activation,
            activation_fps=100,
            output_path=output,
        )
        assert output.exists()
