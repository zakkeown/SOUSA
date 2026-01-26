# Profiles Module

The profiles module handles player profile generation with skill-tier-based archetypes. Each profile represents a unique "virtual drummer" with consistent execution characteristics that determine how they perform rudiments.

Profiles capture execution dimensions across several categories:

- **Timing**: Accuracy, consistency, tempo drift, subdivision evenness
- **Dynamics**: Velocity mean/variance, accent differentiation and accuracy
- **Hand Balance**: Left/right velocity ratio, timing bias, consistency delta
- **Rudiment-Specific**: Flam spacing, diddle evenness, roll sustain, buzz consistency

The archetype system generates profiles with realistic correlations between these dimensions based on skill tier (beginner through professional), drawing from music cognition literature for parameter ranges.

::: dataset_gen.profiles.archetypes
    options:
      show_root_heading: false
      members:
        - SkillTier
        - PlayerProfile
        - TimingDimensions
        - DynamicsDimensions
        - HandBalanceDimensions
        - RudimentSpecificDimensions
        - ExecutionDimensions
        - generate_profile
        - generate_profiles_batch
