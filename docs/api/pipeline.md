# Pipeline Module

The pipeline module orchestrates the complete dataset generation process, from player profiles to stored samples. It coordinates all components of the generation pipeline:

1. Generate player profiles with correlated skill dimensions
2. Generate MIDI performances with realistic timing/velocity
3. Render audio via FluidSynth (multiple soundfonts)
4. Apply audio augmentation (rooms, mics, compression, noise)
5. Compute hierarchical labels (stroke, measure, exercise)
6. Save to disk with profile-based train/val/test splits

## Dataset Generator

Main orchestration for dataset generation with configurable scale and augmentation options.

::: dataset_gen.pipeline.generate
    options:
      show_root_heading: false
      members:
        - GenerationConfig
        - GenerationProgress
        - DatasetGenerator
        - generate_dataset

## Storage

Utilities for writing samples to disk in efficient formats (Parquet for labels, FLAC for audio, MIDI files).

::: dataset_gen.pipeline.storage
    options:
      show_root_heading: false
      members:
        - StorageConfig
        - DatasetWriter
        - ParquetReader
        - write_sample
        - write_batch
