# SOUSA Dataset Limitations

This document explicitly describes the known limitations and potential biases of the SOUSA dataset. Understanding these limitations is critical for appropriate use in ML research.

## Synthetic Data Characteristics

### Self-Referential Labels

SOUSA labels are **derived from the same parameters used to generate the data**. This creates a circular relationship:

```
PlayerProfile parameters → Generate timing/velocity errors → Compute scores from errors
```

**Implications:**
- Models may learn the generation function rather than "drumming quality"
- High accuracy on SOUSA may not transfer to real-world assessment
- Labels are mathematically consistent but not validated against human perception

**Recommendation:** Treat SOUSA as a pretraining or development dataset. Validate models on real human performances before deployment.

### Well-Behaved Error Distribution

Timing and velocity errors are sampled from **Gaussian distributions** with skill-tier-dependent parameters:

| Dimension | Beginner | Professional |
|-----------|----------|--------------|
| Timing std (ms) | ~50 | ~7 |
| Velocity std | ~0.12 | ~0.07 |

**What this captures:**
- Gradual skill progression
- Correlated improvement across dimensions
- Realistic variance within skill tiers

**What this misses:**
- Catastrophic failures (wrong sticking patterns, skipped notes)
- Tempo collapse under difficulty
- Fatigue effects over long exercises
- Cognitive errors (playing wrong rudiment)
- Recovery patterns after mistakes

**Implication:** Models trained on SOUSA may not handle out-of-distribution failures that occur in real practice sessions.

## Instrument Limitations

### Single Instrument

SOUSA generates **snare drum only** (MIDI note 38, General MIDI acoustic snare).

**Not included:**
- Kick drum patterns
- Hi-hat/cymbal interplay
- Tom fills
- Full drum kit coordination
- Marching tenor (quads) or bass drum parts

**Implication:** Models are specialized for snare-only assessment and will not generalize to full kit drumming.

### Soundfont Dependency

Audio is synthesized using **SF2 soundfonts**, not recorded from physical instruments.

| Soundfont | Character |
|-----------|-----------|
| `generalu` | General MIDI standard |
| `marching` | Marching snare character |
| `mtpowerd` | Modern processed sound |
| `douglasn` | Natural acoustic character |
| `fluidr3` | FluidR3 GM standard |

**Implications:**
- Timbral variety is limited to these 5 soundfonts
- Synthesized transients may differ from real drum recordings
- No stick type variation, head tuning, or snare wire adjustments
- Models may overfit to synthesis artifacts

## Label Quality

### Perceptual Score Scaling (v0.2+)

Scores now use **sigmoid scaling** that better matches human perception:

```python
# Old linear scaling (v0.1):
timing_accuracy = 100 - (mean_abs_error_ms * 2)  # 0ms = 100, 50ms = 0

# New sigmoid scaling (v0.2+):
timing_accuracy = 100 * sigmoid((25 - mean_abs_error) / 10)
# <10ms ≈ 92-100 (imperceptible)
# 25ms ≈ 50 (noticeable)
# >50ms ≈ 0-8 (clearly audible)
```

**Note:** If using pre-v0.2 datasets, scores will differ slightly. Regenerate for consistency.

### Skill Tier Overlap (Classification Ceiling)

Due to realistic skill distributions, **adjacent tiers have significant score overlap**:

| Adjacent Tiers | Score Overlap |
|----------------|---------------|
| beginner ↔ intermediate | ~67% |
| intermediate ↔ advanced | ~83% |
| advanced ↔ professional | ~83% |

**Implications:**
- 4-class skill tier classification has an inherent accuracy ceiling due to label noise
- A sample with score=50 could legitimately be beginner, intermediate, or advanced
- Models will plateau at ~70-80% accuracy regardless of capacity

**Mitigations provided:**
1. **`tier_confidence`** (0-1): Indicates how central a sample is to its tier's distribution. Filter low-confidence samples for cleaner training.
2. **`skill_tier_binary`** (novice/skilled): 2-class alternative with less overlap:
   - novice = beginner + intermediate
   - skilled = advanced + professional

**Recommendation:** For classification, use `skill_tier_binary` or filter by `tier_confidence > 0.5`. For assessment, use regression on `overall_score`.

### Score Correlations (Redundancy)

Several scores are **highly correlated** (r > 0.85):

```
timing_accuracy ↔ timing_consistency: r = 0.89
overall_score ↔ timing_consistency: r = 0.89
overall_score ↔ timing_accuracy: r = 0.88
```

**Implications:**
- Using all scores as separate targets provides no additional learning signal
- Multi-task models should use orthogonal targets

**Recommended minimal score set:**
1. `overall_score` - General quality (captures correlated cluster)
2. `tempo_stability` - Independent signal about rushing/dragging

### Hand Balance Ceiling Effect

The `hand_balance` score has a **ceiling effect** (mean=88, most samples near 100):

| Metric | Value |
|--------|-------|
| Mean | 87.7 |
| Std | 11.2 |
| Range | [35.3, 100.0] |

**Cause:** v0.1 measured only velocity ratio, which is nearly always high.

**v0.2 fix:** `hand_balance` now combines velocity (50%) + timing (50%) for better discrimination.

**Recommendation:** For pre-v0.2 datasets, exclude `hand_balance` from ML targets.

### No Human Validation

Scores are computed from **mathematical formulas**, not human ratings.

**Not validated:**
- Perceptual relevance of scoring thresholds
- Whether a score of 75 "sounds" better than 65 to human listeners
- Inter-rater reliability with music instructors
- Cultural or stylistic scoring preferences

**Recommendation:** Do not interpret score outputs as equivalent to instructor feedback without external validation.

### Rudiment-Specific Score Availability

Some scores are only computed for applicable rudiments:

| Score | Available When |
|-------|----------------|
| `flam_quality` | Rudiment contains flams |
| `diddle_quality` | Rudiment contains diddles |
| `roll_sustain` | Rudiment is a roll type |

**Implication:** Null values exist in the dataset. Handle appropriately in training (masking, separate heads, etc.).

## Domain Gap Considerations

### Synthetic vs. Real Recordings

Real drum recordings include:

| Factor | SOUSA | Real World |
|--------|-------|------------|
| Bleed from other instruments | No | Yes |
| Room acoustics | Simulated IR | Physical space |
| Microphone characteristics | Modeled | Actual hardware |
| Performance context | Isolated | Musical setting |
| Player psychology | None | Nerves, fatigue, musicality |

**Implication:** Models trained on SOUSA may struggle with:
- Multi-instrument mixes
- Unusual room characteristics
- Low-quality or phone recordings not matching augmentation presets
- Real performance variability

### Augmentation Coverage

SOUSA includes 10 augmentation presets ranging from clean studio to lo-fi:

```
clean_studio → practice_room → concert_hall → gym → lo_fi
```

**Not covered:**
- Extreme clipping or distortion
- Heavy compression artifacts (broadcast limiting)
- Pitch shifting or time stretching
- Real-world microphone failures
- Environmental noise beyond included profiles

## Statistical Considerations

### Skill Tier Distribution & Class Imbalance

Default generation uses:

| Tier | Proportion | Imbalance |
|------|------------|-----------|
| Professional | 8% | 4.26x underrepresented |
| Beginner | 26% | - |
| Advanced | 32% | - |
| Intermediate | 34% | majority class |

**Class imbalance ratio:** 4.26:1 (intermediate vs professional)

**Implications:**
- Models trained without class weights will be biased toward intermediate predictions
- Accuracy alone is misleading; use balanced accuracy
- Professional tier predictions will have lower recall

**Mitigations provided:**
1. **Class weights** are enabled by default in `skill_classification.py`
2. **Balanced accuracy** is used for early stopping and model selection
3. **Per-class metrics** are reported in evaluation

**Recommendation:** Always use class weights for skill tier classification. Report balanced accuracy, not just accuracy.

### Profile-Based Splits

Train/val/test splits are by **player profile**, not by sample:

- All samples from a profile stay in the same split
- Prevents data leakage from player-specific patterns
- May make generalization harder (testing on "new players")

See [Experiments](#) for empirical analysis of split methodology impact.

## Recommended Use Cases

### Appropriate Uses

- **Pretraining:** Initialize models before fine-tuning on real data
- **Architecture development:** Compare model architectures on controlled data
- **Ablation studies:** Isolate effects of audio augmentation, skill tiers, etc.
- **Educational tools:** Build practice aids where exact calibration is less critical
- **Baseline establishment:** Provide reproducible benchmarks for the research community

### Use With Caution

- **Production assessment:** Validate on real performances before deployment
- **Certification/grading:** Scores are not calibrated to educational standards
- **Cross-instrument transfer:** Models will not generalize to non-snare instruments

### Not Recommended

- **Direct deployment** without real-world validation
- **Claims of human-equivalent assessment** without instructor calibration studies
- **Full drum kit assessment** (snare only)

## Future Work

Potential improvements to address these limitations:

1. **Human validation study:** Collect instructor ratings on a subset for calibration
2. **Failure mode injection:** Add catastrophic errors (wrong patterns, tempo collapse)
3. **Real recording subset:** Include a small set of actual recordings for domain gap measurement
4. **Extended instrument support:** Marching percussion, full kit patterns
5. **Longitudinal profiles:** Model player improvement over time

## Citation

If you use SOUSA in research, please cite:

```bibtex
@dataset{sousa2024,
  title={SOUSA: Synthetic Open Unified Snare Assessment Dataset},
  author={...},
  year={2024},
  url={https://huggingface.co/datasets/zkeown/sousa}
}
```

---

*Last updated: January 2026 (v0.2 - added tier_confidence, perceptual scoring, class weights)*
