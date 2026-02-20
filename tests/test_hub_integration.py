"""Integration test: generation output is compatible with Parquet hub uploader."""

import json
import numpy as np
import pytest

from dataset_gen.pipeline.storage import StorageConfig, DatasetWriter
from dataset_gen.hub.uploader import HubConfig, DatasetUploader
from dataset_gen.labels.schema import Sample, ExerciseScores


@pytest.fixture
def generated_dataset(tmp_path):
    """Generate a small dataset using the storage pipeline."""
    config = StorageConfig(output_dir=tmp_path)
    writer = DatasetWriter(config)

    rudiments = ["flam", "paradiddle", "single_stroke_roll"]
    profiles = {"p1": "train", "p2": "val", "p3": "test"}

    for profile_id in profiles:
        for slug in rudiments:
            sample_id = f"test_{slug}_{profile_id}"
            sample = Sample(
                sample_id=sample_id,
                profile_id=profile_id,
                rudiment_slug=slug,
                tempo_bpm=120,
                duration_sec=2.0,
                num_cycles=2,
                skill_tier="beginner",
                skill_tier_binary="novice",
                dominant_hand="right",
                strokes=[],
                measures=[],
                exercise_scores=ExerciseScores(
                    timing_accuracy=75.0,
                    timing_consistency=70.0,
                    tempo_stability=80.0,
                    subdivision_evenness=65.0,
                    velocity_control=72.0,
                    accent_differentiation=60.0,
                    accent_accuracy=68.0,
                    hand_balance=74.0,
                    weak_hand_index=50.0,
                    groove_feel_proxy=0.7,
                    overall_score=71.0,
                ),
            )
            audio = np.zeros(44100, dtype=np.float32)
            midi = b"\x00" * 100
            writer.write_sample(sample, midi_data=midi, audio_data=audio)

    writer.flush()

    splits = {
        "train_profile_ids": ["p1"],
        "val_profile_ids": ["p2"],
        "test_profile_ids": ["p3"],
    }
    with open(tmp_path / "splits.json", "w") as f:
        json.dump(splits, f)

    return tmp_path


def test_generated_dataset_builds_labels_only(generated_dataset):
    """Labels-only config builds successfully from generated data."""
    config = HubConfig(dataset_dir=generated_dataset, repo_id="test/repo")
    uploader = DatasetUploader(config)
    dd = uploader.build_dataset_dict("labels_only")

    assert dd["train"].num_rows == 3
    assert dd["validation"].num_rows == 3
    assert dd["test"].num_rows == 3
    assert "overall_score" in dd["train"].column_names


def test_generated_dataset_builds_midi_only(generated_dataset):
    """MIDI-only config includes MIDI bytes from generated data."""
    config = HubConfig(dataset_dir=generated_dataset, repo_id="test/repo")
    uploader = DatasetUploader(config)
    dd = uploader.build_dataset_dict("midi_only")

    assert "midi" in dd["train"].column_names
    midi_val = dd["train"][0]["midi"]
    assert isinstance(midi_val, bytes)
    assert len(midi_val) > 0


def test_generated_dataset_builds_audio(generated_dataset):
    """Audio config includes audio from generated data."""
    config = HubConfig(dataset_dir=generated_dataset, repo_id="test/repo")
    uploader = DatasetUploader(config)
    dd = uploader.build_dataset_dict("audio")

    assert "audio" in dd["train"].column_names
    assert "midi" in dd["train"].column_names


def test_all_splits_have_correct_sample_count(generated_dataset):
    """Each split has the correct number of samples based on profile assignment."""
    config = HubConfig(dataset_dir=generated_dataset, repo_id="test/repo")
    uploader = DatasetUploader(config)
    dd = uploader.build_dataset_dict("labels_only")

    total = sum(ds.num_rows for ds in dd.values())
    assert total == 9
