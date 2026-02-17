#!/usr/bin/env python3
"""
Extract and install SFZ drum libraries for SOUSA.

Looks for known SFZ library ZIP files in ~/Downloads and extracts them
to data/sfz/ for use with the SFZ synthesizer pipeline.

Supported libraries:
  - SM Drums (SMDrums_Sforzando_1.2.zip) — 127 velocity layers, 8 round robins
  - Frankensnare (Frankensnare_2100.zip) — 5 velocity layers, 4 round robins

Usage:
    python scripts/setup_sfz.py                    # Auto-detect and extract
    python scripts/setup_sfz.py --list             # List installed SFZ libraries
    python scripts/setup_sfz.py --validate         # Validate sample file access
"""

import argparse
from pathlib import Path
import sys
import zipfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Known SFZ library ZIP files and their expected structure
SFZ_LIBRARIES = {
    "SMDrums": {
        "zip_patterns": ["SMDrums_Sforzando*.zip", "SM_Drums*.zip", "SMDrums*.zip"],
        "extract_to": "SMDrums",
        "description": "SM Drums — 127 velocity layers, 8 random round robins",
        "sfz_files": ["Programs/Snare_bus.sfz"],
    },
    "Frankensnare": {
        "zip_patterns": ["Frankensnare*.zip"],
        "extract_to": "Frankensnare",
        "description": "Frankensnare — 5 velocity layers, 4 sequential round robins, 30 snare drums",
        "sfz_files": ["Programs/*.sfz"],
    },
}


def find_zip(downloads_dir: Path, patterns: list[str]) -> Path | None:
    """Find the first matching ZIP file in the downloads directory."""
    for pattern in patterns:
        matches = sorted(downloads_dir.glob(pattern))
        if matches:
            return matches[-1]  # Use latest version if multiple
    return None


def extract_library(zip_path: Path, output_dir: Path, lib_name: str, lib_info: dict) -> bool:
    """Extract a ZIP library to the output directory."""
    target_dir = output_dir / lib_info["extract_to"]

    if target_dir.exists():
        # Check if already extracted
        existing_files = list(target_dir.rglob("*.wav")) + list(target_dir.rglob("*.flac"))
        if existing_files:
            print(f"  Already extracted: {target_dir} ({len(existing_files)} samples)")
            return True

    print(f"  Extracting {zip_path.name} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Check for a single top-level directory in the ZIP
            names = zf.namelist()
            top_dirs = {n.split("/")[0] for n in names if "/" in n}

            if len(top_dirs) == 1:
                # ZIP has a single root folder — extract so its contents
                # end up directly under target_dir
                root = top_dirs.pop()
                for info in zf.infolist():
                    if info.filename.startswith(root + "/"):
                        # Strip the root directory prefix
                        rel = info.filename[len(root) + 1 :]
                        if not rel:
                            continue
                        out_path = target_dir / rel
                        if info.is_dir():
                            out_path.mkdir(parents=True, exist_ok=True)
                        else:
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(info) as src, open(out_path, "wb") as dst:
                                dst.write(src.read())
            else:
                # Multiple top-level entries — extract directly
                zf.extractall(target_dir)

        sample_count = len(list(target_dir.rglob("*.wav"))) + len(list(target_dir.rglob("*.flac")))
        print(f"  Extracted {sample_count} samples to {target_dir}")
        return True

    except Exception as e:
        print(f"  ERROR extracting {zip_path}: {e}")
        return False


def validate_library(sfz_dir: Path, lib_name: str, lib_info: dict) -> bool:
    """Validate that SFZ files can find their referenced samples."""
    target_dir = sfz_dir / lib_info["extract_to"]
    if not target_dir.exists():
        print(f"  {lib_name}: NOT INSTALLED")
        return False

    # Find SFZ files
    sfz_files = []
    for pattern in lib_info["sfz_files"]:
        sfz_files.extend(target_dir.glob(pattern))

    if not sfz_files:
        print(f"  {lib_name}: No SFZ files found matching {lib_info['sfz_files']}")
        return False

    print(f"  {lib_name}: {len(sfz_files)} SFZ program(s)")

    # Try to parse one SFZ file and check sample references
    try:
        from dataset_gen.audio_synth.sfz_parser import SfzParser

        parser = SfzParser()
        sfz_file = sfz_files[0]
        instrument = parser.parse(sfz_file)

        total_regions = len(instrument.regions)
        missing = 0
        checked = 0
        for region in instrument.regions[:50]:  # Check first 50
            path = instrument.resolve_sample_path(region)
            checked += 1
            if not path.exists():
                missing += 1
                if missing <= 3:
                    print(f"    Missing: {path}")

        print(
            f"    {sfz_file.name}: {total_regions} regions, "
            f"{checked} checked, {missing} missing samples"
        )
        if missing > 0 and missing <= 3:
            print(f"    (showing first {missing} missing)")
        elif missing > 3:
            print(f"    ({missing} total missing)")

        return missing == 0

    except Exception as e:
        print(f"    Parse error: {e}")
        return False


def list_installed(sfz_dir: Path) -> None:
    """List all installed SFZ libraries."""
    if not sfz_dir.exists():
        print("No SFZ libraries installed.")
        print(f"Expected directory: {sfz_dir}")
        return

    print(f"\nInstalled SFZ libraries in {sfz_dir}:\n")

    found_any = False
    for subdir in sorted(sfz_dir.iterdir()):
        if not subdir.is_dir():
            continue
        sfz_files = list(subdir.rglob("*.sfz"))
        wav_files = list(subdir.rglob("*.wav"))
        flac_files = list(subdir.rglob("*.flac"))
        sample_count = len(wav_files) + len(flac_files)

        if sfz_files or sample_count > 0:
            found_any = True
            size_mb = sum(f.stat().st_size for f in wav_files + flac_files) / (1024 * 1024)
            print(f"  {subdir.name}/")
            print(f"    SFZ programs: {len(sfz_files)}")
            print(f"    Samples: {sample_count} ({size_mb:.0f} MB)")
            for sfz in sfz_files[:5]:
                print(f"    - {sfz.relative_to(subdir)}")
            if len(sfz_files) > 5:
                print(f"    ... and {len(sfz_files) - 5} more")
            print()

    if not found_any:
        print("  (none found)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and install SFZ drum libraries for SOUSA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--downloads",
        type=Path,
        default=Path.home() / "Downloads",
        help="Directory to search for ZIP files (default: ~/Downloads)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "sfz",
        help="Output directory for extracted libraries (default: data/sfz/)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List installed SFZ libraries",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate sample file accessibility",
    )

    args = parser.parse_args()
    sfz_dir = args.output

    if args.list:
        list_installed(sfz_dir)
        return

    if args.validate:
        print("Validating SFZ libraries...\n")
        all_ok = True
        for lib_name, lib_info in SFZ_LIBRARIES.items():
            if not validate_library(sfz_dir, lib_name, lib_info):
                all_ok = False
        if all_ok:
            print("\nAll libraries validated successfully.")
        else:
            print("\nSome libraries have issues.")
            sys.exit(1)
        return

    # Auto-detect and extract
    print("SFZ Library Setup")
    print("=" * 50)
    print(f"Searching: {args.downloads}")
    print(f"Output:    {sfz_dir}\n")

    sfz_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    for lib_name, lib_info in SFZ_LIBRARIES.items():
        print(f"{lib_name}: {lib_info['description']}")
        zip_path = find_zip(args.downloads, lib_info["zip_patterns"])

        if zip_path is None:
            print(f"  ZIP not found in {args.downloads}")
            print(f"  Expected: {lib_info['zip_patterns']}")
            print()
            continue

        print(f"  Found: {zip_path.name}")
        if extract_library(zip_path, sfz_dir, lib_name, lib_info):
            extracted += 1
        print()

    print("=" * 50)
    if extracted > 0:
        print(f"Extracted {extracted} library(ies).")
        print("\nValidating...")
        for lib_name, lib_info in SFZ_LIBRARIES.items():
            validate_library(sfz_dir, lib_name, lib_info)
    else:
        print("No libraries extracted.")
        print("\nTo use SFZ soundfonts, download the ZIP files to ~/Downloads:")
        print("  - SM Drums: SMDrums_Sforzando_1.2.zip")
        print("  - Frankensnare: Frankensnare_2100.zip")

    print(f"\nTo list installed: python {__file__} --list")


if __name__ == "__main__":
    main()
