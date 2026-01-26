# Contributing to SOUSA

Thank you for your interest in contributing to SOUSA (Synthetic Open Unified Snare Assessment)!

## Reporting Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/zakkeown/SOUSA/issues) with:

- A clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Your environment (Python version, OS)
- Any relevant error messages or logs

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/zakkeown/SOUSA.git
   cd SOUSA
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks (**required** - CI will fail without this):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

   Or use the Makefile:
   ```bash
   make setup-hooks
   ```

5. Install FluidSynth (required for audio synthesis):
   ```bash
   # macOS
   brew install fluid-synth

   # Ubuntu/Debian
   sudo apt-get install fluidsynth libfluidsynth-dev

   # Windows
   # Download from https://github.com/FluidSynth/fluidsynth/releases
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_profiles.py

# Run with coverage
pytest --cov=dataset_gen

# Run with verbose output
pytest -v
```

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to run automated checks before each commit. **Pre-commit hooks are required** - CI will fail if they weren't run.

The hooks check:
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with a newline
- **check-yaml/check-toml**: Validates YAML and TOML syntax
- **black**: Formats Python code (line length: 100)
- **ruff**: Fast Python linter with auto-fix

```bash
# Run on all files (do this before committing)
pre-commit run --all-files

# Or use the Makefile
make pre-commit
```

## Code Style

We use [black](https://black.readthedocs.io/) for formatting and [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
# Format code
black dataset_gen/ tests/ scripts/

# Check linting
ruff check dataset_gen/ tests/ scripts/

# Auto-fix linting issues
ruff check --fix dataset_gen/ tests/ scripts/

# Or use the Makefile
make format
make lint
```

Configuration is in `pyproject.toml`:
- Line length: 100 characters
- Python target: 3.10+

## Pull Request Process

1. Fork the repository and create a branch from `main`
2. Make your changes with clear, descriptive commits
3. Add tests for any new functionality
4. Ensure all tests pass: `pytest`
5. Ensure code is formatted: `black --check .`
6. Update documentation if needed
7. Submit a pull request with a clear description

## Project Structure

```
SOUSA/
├── dataset_gen/           # Core generation modules
│   ├── rudiments/         # Rudiment definitions (YAML)
│   ├── profiles/          # Player skill modeling
│   ├── midi_gen/          # MIDI generation engine
│   ├── audio_synth/       # FluidSynth wrapper
│   ├── audio_aug/         # Augmentation pipeline
│   ├── labels/            # Label computation
│   ├── pipeline/          # Orchestration
│   └── validation/        # Dataset validation
├── tests/                 # Test suite
├── scripts/               # CLI scripts
├── data/                  # Soundfonts, IRs, noise profiles
└── output/                # Generated datasets
```

## Adding New Rudiments

Rudiment definitions are YAML files in `dataset_gen/rudiments/definitions/`. To add a new rudiment:

1. Create a YAML file following the existing pattern
2. Add tests in `tests/test_rudiments.py`
3. Run validation to ensure correctness

## Questions?

Feel free to open an issue for questions about contributing.
