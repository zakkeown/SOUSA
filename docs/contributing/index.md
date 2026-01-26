# Contributing to SOUSA

Thank you for your interest in contributing to SOUSA (Synthetic Open Unified Snare Assessment)! This guide covers everything you need to get started.

## Ways to Contribute

- **Report bugs** - Found an issue? [Open an issue](https://github.com/zakkeown/SOUSA/issues)
- **Suggest features** - Have an idea? We'd love to hear it
- **Submit pull requests** - Fix bugs, add features, improve docs
- **Add rudiments** - Expand the rudiment library
- **Share examples** - Show how you're using SOUSA

## Reporting Issues

When reporting bugs, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Your environment (Python version, OS, package versions)
- Any relevant error messages or logs

```markdown
**Environment:**
- OS: macOS 14.0
- Python: 3.11.5
- SOUSA version: 0.2.0
- FluidSynth version: 2.3.4

**Steps to reproduce:**
1. Run `python scripts/generate_dataset.py --preset small`
2. ...

**Error message:**
```
<paste error here>
```
```

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/zakkeown/SOUSA.git
cd SOUSA
```

### 2. Create a Virtual Environment

=== "macOS/Linux"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

=== "Windows"

    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

### 3. Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs:

- Core SOUSA package in editable mode
- `pytest` for testing
- `black` for code formatting
- `ruff` for linting
- `pre-commit` for git hooks

### 4. Install Pre-commit Hooks

!!! warning "Required"
    Pre-commit hooks are **required**. CI will fail if code doesn't pass the hooks.

```bash
pip install pre-commit
pre-commit install
```

Or use the Makefile:

```bash
make setup-hooks
```

### 5. Install FluidSynth

FluidSynth is required for audio synthesis.

=== "macOS"

    ```bash
    brew install fluid-synth
    ```

=== "Ubuntu/Debian"

    ```bash
    sudo apt-get update
    sudo apt-get install fluidsynth libfluidsynth-dev
    ```

=== "Windows"

    Download from [FluidSynth releases](https://github.com/FluidSynth/fluidsynth/releases) and add to PATH.

Verify installation:

```bash
fluidsynth --version
```

### 6. Download Soundfonts

```bash
python scripts/setup_soundfonts.py
```

This downloads the required SF2 soundfont files to `data/soundfonts/`.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_profiles.py

# Run specific test
pytest tests/test_profiles.py::TestProfileGeneration::test_generate_profile_beginner

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=dataset_gen

# Run tests matching a pattern
pytest -k "profile"
```

## Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com/) to run automated checks before each commit.

### Configured Hooks

| Hook | Purpose |
|------|---------|
| `trailing-whitespace` | Removes trailing whitespace |
| `end-of-file-fixer` | Ensures files end with newline |
| `check-yaml` | Validates YAML syntax |
| `check-toml` | Validates TOML syntax |
| `black` | Formats Python code |
| `ruff` | Lints Python code with auto-fix |

### Running Hooks Manually

```bash
# Run on all files
pre-commit run --all-files

# Or use the Makefile
make pre-commit

# Run specific hook
pre-commit run black --all-files
```

### Skipping Hooks (Emergency Only)

```bash
git commit --no-verify -m "Emergency fix"
```

!!! danger "Use Sparingly"
    CI will still run the checks. Only skip hooks in genuine emergencies.

## Code Style

### Formatting with Black

We use [Black](https://black.readthedocs.io/) for consistent formatting:

```bash
# Format all code
black dataset_gen/ tests/ scripts/

# Check without modifying
black --check dataset_gen/ tests/ scripts/

# Or use Makefile
make format
```

Configuration in `pyproject.toml`:

- Line length: **100 characters**
- Python target: **3.10+**

### Linting with Ruff

We use [Ruff](https://docs.astral.sh/ruff/) for fast linting:

```bash
# Check for issues
ruff check dataset_gen/ tests/ scripts/

# Auto-fix issues
ruff check --fix dataset_gen/ tests/ scripts/

# Or use Makefile
make lint
```

### Type Hints

We encourage type hints for public APIs:

```python
def generate_profile(
    skill_tier: str,
    seed: int | None = None,
) -> PlayerProfile:
    """Generate a player profile.

    Args:
        skill_tier: One of "beginner", "intermediate", "advanced", "professional"
        seed: Random seed for reproducibility

    Returns:
        Generated PlayerProfile instance
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def compute_timing_score(
    errors_ms: list[float],
    threshold_ms: float = 25.0,
) -> float:
    """Compute timing accuracy score from error values.

    Uses sigmoid scaling to map errors to a 0-100 score range.

    Args:
        errors_ms: List of timing errors in milliseconds
        threshold_ms: Error value mapping to score of 50

    Returns:
        Timing accuracy score (0-100)

    Raises:
        ValueError: If errors_ms is empty

    Example:
        >>> compute_timing_score([5, 10, 15])
        85.7
    """
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, descriptive commits
- Add tests for new functionality
- Update documentation if needed

### 3. Ensure Quality

```bash
# Format code
black dataset_gen/ tests/ scripts/

# Run linter
ruff check dataset_gen/ tests/ scripts/

# Run tests
pytest

# Run pre-commit
pre-commit run --all-files
```

### 4. Submit Pull Request

- Provide a clear description of changes
- Reference any related issues
- Include test results if relevant

### PR Title Format

```
feat: Add new flam quality metric
fix: Handle empty stroke lists in scoring
docs: Update PyTorch DataLoader example
refactor: Simplify MIDI generation logic
test: Add tests for edge cases in profiles
```

## Adding New Rudiments

Rudiment definitions are YAML files in `dataset_gen/rudiments/definitions/`.

### 1. Create YAML Definition

```yaml
# dataset_gen/rudiments/definitions/your_rudiment.yaml
name: Your Rudiment Name
slug: your_rudiment_slug
category: single-stroke  # or double-stroke, diddle, flam, drag, roll
tempo_range:
  min: 60
  max: 180
time_signature: "4/4"
subdivision: 16  # 16th notes

patterns:
  - name: basic
    sticking: "RLRL RLRL"
    articulations: "taaa taaa"
    measures: 2

# Optional articulation map
articulation_map:
  t: tap
  a: accent
  g: grace
  d: diddle
```

### 2. Add Tests

```python
# tests/test_rudiments.py

def test_your_rudiment_loads():
    rudiment = load_rudiment("your_rudiment_slug")
    assert rudiment is not None
    assert rudiment.slug == "your_rudiment_slug"


def test_your_rudiment_generation():
    rudiment = load_rudiment("your_rudiment_slug")
    midi = generate_midi(rudiment, tempo=120)
    assert len(midi.notes) > 0
```

### 3. Validate

```bash
# Run rudiment validation
python -m dataset_gen.validation.rudiments your_rudiment_slug

# Run tests
pytest tests/test_rudiments.py -v
```

## Project Structure

```
SOUSA/
├── dataset_gen/           # Core generation modules
│   ├── rudiments/         # Rudiment definitions (YAML)
│   │   └── definitions/   # 40 rudiment YAML files
│   ├── profiles/          # Player skill modeling
│   ├── midi_gen/          # MIDI generation engine
│   ├── audio_synth/       # FluidSynth wrapper
│   ├── audio_aug/         # Augmentation pipeline
│   ├── labels/            # Label computation
│   ├── pipeline/          # Orchestration
│   └── validation/        # Dataset validation
├── tests/                 # Test suite
├── scripts/               # CLI scripts
├── examples/              # Usage examples
├── docs/                  # Documentation source
├── data/                  # Soundfonts, IRs, noise profiles
└── output/                # Generated datasets
```

## Getting Help

- **Questions**: [Open an issue](https://github.com/zakkeown/SOUSA/issues) with the "question" label
- **Discussions**: Use GitHub Discussions for general topics
- **Documentation**: Check the [docs](https://zakkeown.github.io/SOUSA)

## Code of Conduct

Please be respectful and constructive in all interactions. We're building this together!

## License

By contributing to SOUSA, you agree that your contributions will be licensed under the same license as the project (MIT).
