"""Tests for audio analysis module."""

from __future__ import annotations

import pytest
import mido

from dataset_gen.audio_analysis.onsets import DetectedOnset
from dataset_gen.audio_analysis.midi_alignment import MidiOnset, extract_midi_onsets


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
