"""Integration test: generation output layout matches uploader staging."""

import json
import numpy as np
import pandas as pd
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

    # Create splits.json
    splits = {
        "train_profile_ids": ["p1"],
        "val_profile_ids": ["p2"],
        "test_profile_ids": ["p3"],
    }
    with open(tmp_path / "splits.json", "w") as f:
        json.dump(splits, f)

    return tmp_path


def test_generation_layout_matches_uploader(generated_dataset):
    """Verify uploader can stage a dataset generated with rudiment subdirs."""
    config = HubConfig(dataset_dir=generated_dataset, repo_id="test/repo")
    uploader = DatasetUploader(config)

    staging_dir = uploader.prepare()

    # Check parquet files exist for all splits
    assert (staging_dir / "data" / "train-00000-of-00001.parquet").exists()
    assert (staging_dir / "data" / "validation-00000-of-00001.parquet").exists()
    assert (staging_dir / "data" / "test-00000-of-00001.parquet").exists()

    # Every audio and midi path in every split parquet should point to a real file
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(staging_dir / "data" / f"{split}-00000-of-00001.parquet")

        assert len(df) > 0, f"No samples in {split} split"

        for _, row in df.iterrows():
            audio_path = row.get("audio")
            if audio_path and pd.notna(audio_path):
                full_path = staging_dir / audio_path
                assert full_path.exists(), f"Missing staged audio: {audio_path}"

            midi_path = row.get("midi")
            if midi_path and pd.notna(midi_path):
                full_path = staging_dir / midi_path
                assert full_path.exists(), f"Missing staged midi: {midi_path}"


def test_staged_paths_include_rudiment_slug(generated_dataset):
    """Verify staged parquet paths use rudiment subdirectory format."""
    config = HubConfig(dataset_dir=generated_dataset, repo_id="test/repo")
    uploader = DatasetUploader(config)

    staging_dir = uploader.prepare()
    train_df = pd.read_parquet(staging_dir / "data" / "train-00000-of-00001.parquet")

    # Audio paths should be like "audio/{rudiment_slug}/{filename}"
    for _, row in train_df.iterrows():
        audio = row.get("audio")
        if audio and pd.notna(audio):
            parts = audio.split("/")
            assert len(parts) == 3, f"Expected audio/slug/file, got: {audio}"
            assert parts[0] == "audio"
            assert parts[1] in ["flam", "paradiddle", "single_stroke_roll"]

        midi = row.get("midi")
        if midi and pd.notna(midi):
            parts = midi.split("/")
            assert len(parts) == 3, f"Expected midi/slug/file, got: {midi}"
            assert parts[0] == "midi"
            assert parts[1] in ["flam", "paradiddle", "single_stroke_roll"]
