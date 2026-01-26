"""
SOUSA Experiments
=================

Experimental validation scripts for SOUSA dataset methodology.

Experiments:
- score_analysis: Analyze score correlations and recommend minimal score sets
- split_validation: Compare profile-based vs random splits
- augmentation_ablation: Test augmentation impact on model performance
- soundfont_ablation: Test cross-soundfont generalization

Run all experiments:
    python scripts/run_experiments.py --data-dir output/dataset

Run individual experiment:
    python -m experiments.score_analysis --data-dir output/dataset
"""
