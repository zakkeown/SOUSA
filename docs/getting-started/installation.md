# Installation

This guide covers installing SOUSA and its dependencies for dataset generation.

!!! tip "Just want to use the dataset?"
    If you only need to load the pre-generated dataset, you can skip local installation entirely:
    ```python
    pip install datasets
    from datasets import load_dataset
    dataset = load_dataset("zkeown/sousa")
    ```

## Requirements

- **Python 3.10+** (3.11 or 3.12 recommended)
- **FluidSynth** (required for audio synthesis)
- **pip** (Python package manager)

## Python Installation

SOUSA requires Python 3.10 or later. Check your version:

```bash
python --version
# or
python3 --version
```

If you need to install Python, visit [python.org/downloads](https://www.python.org/downloads/) or use a version manager like `pyenv`.

## Install SOUSA

### Basic Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/zakkeown/SOUSA.git
cd SOUSA

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install base package
pip install -e .
```

### Installation Extras

SOUSA provides optional dependency groups for different use cases:

=== "Development"

    ```bash
    # Install with development tools (pytest, black, ruff)
    pip install -e '.[dev]'
    ```

=== "HuggingFace Hub"

    ```bash
    # Install with HuggingFace dependencies for uploading/downloading
    pip install -e '.[hub]'
    ```

=== "Machine Learning"

    ```bash
    # Install with ML dependencies (PyTorch, transformers, etc.)
    pip install -e '.[ml]'
    ```

=== "Documentation"

    ```bash
    # Install with MkDocs for building documentation
    pip install -e '.[docs]'
    ```

=== "All Extras"

    ```bash
    # Install everything
    pip install -e '.[dev,hub,ml,docs]'
    ```

## FluidSynth Installation

FluidSynth is required for synthesizing audio from MIDI. Install it for your operating system:

=== "macOS"

    Using Homebrew:

    ```bash
    brew install fluid-synth
    ```

    Verify installation:

    ```bash
    fluidsynth --version
    ```

=== "Ubuntu/Debian"

    Using apt:

    ```bash
    sudo apt update
    sudo apt install fluidsynth libfluidsynth-dev
    ```

    Verify installation:

    ```bash
    fluidsynth --version
    ```

=== "Windows"

    **Option 1: Using Chocolatey**

    ```powershell
    choco install fluidsynth
    ```

    **Option 2: Manual Installation**

    1. Download FluidSynth from [GitHub Releases](https://github.com/FluidSynth/fluidsynth/releases)
    2. Extract to a folder (e.g., `C:\FluidSynth`)
    3. Add the `bin` folder to your system PATH

    Verify installation:

    ```powershell
    fluidsynth --version
    ```

=== "Conda"

    If using Conda/Mamba:

    ```bash
    conda install -c conda-forge fluidsynth
    ```

!!! warning "FluidSynth Required for Audio"
    Without FluidSynth, you can still generate MIDI files and labels, but audio synthesis will fail. Use `--no-audio` flag if FluidSynth is not available:
    ```bash
    python scripts/generate_dataset.py --preset small
    ```

## Soundfont Setup

Soundfonts (`.sf2` files) define the sounds used for audio synthesis. SOUSA includes a setup script to download recommended soundfonts:

```bash
# Download recommended soundfonts (~170 MB)
python scripts/setup_soundfonts.py
```

### Soundfont Options

=== "Default (Recommended)"

    Downloads a good set for rudiment generation:

    ```bash
    python scripts/setup_soundfonts.py
    ```

    Includes:

    - **FluidR3_GM_GS** (141 MB) - High-quality General MIDI
    - **Marching_Snare** (0.2 MB) - Marching snare drum
    - **MT_Power_DrumKit** (8.7 MB) - Punchy acoustic kit

=== "Minimal"

    Just one good soundfont (~141 MB):

    ```bash
    python scripts/setup_soundfonts.py --minimal
    ```

=== "All Soundfonts"

    Download all available soundfonts (~460 MB):

    ```bash
    python scripts/setup_soundfonts.py --all
    ```

=== "Specific Soundfont"

    Download a specific soundfont by name:

    ```bash
    python scripts/setup_soundfonts.py --name Marching_Snare
    ```

### List Available Soundfonts

```bash
python scripts/setup_soundfonts.py --list
```

Output:

```
Available Soundfonts (all verified working)
=================================================================

*[INSTALLED] FluidR3_GM_GS
   FluidR3 GM+GS merged - excellent quality, MIT license
   Size: ~141 MB | License: MIT

 [available] GeneralUser_GS
   Compact GM soundfont with great drums (~30MB)
   Size: ~30 MB | License: Free for any use

 [available] Marching_Snare
   Marching snare drum - great for rudiments
   Size: ~0.2 MB | License: CC BY
...
```

### Custom Soundfonts

You can use your own soundfonts by placing `.sf2` files in `data/soundfonts/` or specifying a path:

```bash
python scripts/generate_dataset.py --soundfont /path/to/my_soundfont.sf2
```

## Verify Installation

Run the test suite to verify everything is working:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test module
pytest tests/test_profiles.py
```

Quick verification:

```bash
# Check that SOUSA can be imported
python -c "from dataset_gen.pipeline.generate import DatasetGenerator; print('SOUSA installed successfully!')"

# Check FluidSynth binding
python -c "import fluidsynth; print('FluidSynth binding OK')"
```

## Troubleshooting

### FluidSynth Not Found

If you get `ModuleNotFoundError: No module named 'fluidsynth'`:

```bash
# Reinstall the Python binding
pip install --force-reinstall pyfluidsynth
```

If the system library is not found:

=== "macOS"

    ```bash
    # Ensure Homebrew's lib path is in DYLD_LIBRARY_PATH
    export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
    ```

=== "Linux"

    ```bash
    # Ensure libfluidsynth is installed
    sudo apt install libfluidsynth-dev

    # Check library path
    ldconfig -p | grep fluidsynth
    ```

### Soundfont Download Fails

If soundfont downloads fail:

1. Check your internet connection
2. Try downloading manually from [musical-artifacts.com](https://musical-artifacts.com/)
3. Place `.sf2` files in `data/soundfonts/`

### Import Errors

If you get import errors after installation:

```bash
# Ensure you're in the virtual environment
source .venv/bin/activate

# Reinstall in editable mode
pip install -e .
```

## Next Steps

Once installation is complete:

1. [Quick Start](quickstart.md) - Generate your first dataset
2. [Loading Data](loading.md) - Access the pre-generated dataset
3. [User Guide](../user-guide/index.md) - Detailed examples for ML tasks
