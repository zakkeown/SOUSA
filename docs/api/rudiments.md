# Rudiments Module

The rudiments module handles loading and parsing of PAS (Percussive Arts Society) drum rudiment definitions. It provides Pydantic models for representing the 40 standard drum rudiments with their sticking patterns, accent patterns, and articulation-specific parameters.

Rudiment definitions are stored as YAML files in `dataset_gen/rudiments/definitions/` and specify:

- Stroke patterns with hand assignments (R/L)
- Articulation types (tap, accent, grace note, diddle, buzz)
- Tempo ranges and subdivisions
- Category classification (roll, diddle, flam, drag)

## Schema Classes

::: dataset_gen.rudiments.schema
    options:
      show_root_heading: false
      members:
        - Hand
        - StrokeType
        - Stroke
        - StickingPattern
        - RudimentCategory
        - RudimentParams
        - Subdivision
        - Rudiment

## Loader

The loader module provides functions for parsing YAML rudiment definitions into Pydantic models.

::: dataset_gen.rudiments.loader
    options:
      show_root_heading: false
      members:
        - load_rudiment
        - load_all_rudiments
        - get_rudiments_by_category
