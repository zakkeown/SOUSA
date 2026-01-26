"""
SOUSA: Synthetic Open Unified Snare Assessment

A synthetic dataset generator for all 40 PAS drum rudiments, designed to train
ML models for drumming performance assessment. Generates 100K+ samples with:

- MIDI performances with realistic timing/velocity variations
- Multi-soundfont audio synthesis via FluidSynth
- Extensive audio augmentation (rooms, mics, compression, noise)
- Hierarchical labels (stroke-level, measure-level, exercise-level scores)
- Profile-based train/val/test splits for generalization testing
"""

from dataset_gen.rudiments.schema import Rudiment, StickingPattern, RudimentCategory
from dataset_gen.profiles.archetypes import PlayerProfile, SkillTier
from dataset_gen.labels.schema import Sample, StrokeLabel, MeasureLabel, ExerciseScores

__version__ = "0.1.0"

__all__ = [
    "Rudiment",
    "StickingPattern",
    "RudimentCategory",
    "PlayerProfile",
    "SkillTier",
    "Sample",
    "StrokeLabel",
    "MeasureLabel",
    "ExerciseScores",
]
