#!/usr/bin/env python3
"""
Download recommended soundfonts for audio generation.

Usage:
    python scripts/setup_soundfonts.py              # Download all recommended
    python scripts/setup_soundfonts.py --list       # List available soundfonts
    python scripts/setup_soundfonts.py --minimal    # Just one good GM soundfont
"""

import argparse
import sys
import urllib.request
from pathlib import Path

# Verified working soundfont sources (tested Jan 2026)
SOUNDFONTS = {
    # General MIDI soundfonts (good fallback, includes drums)
    "FluidR3_GM_GS": {
        "url": "https://musical-artifacts.com/artifacts/1229/FluidR3_GM_GS.sf2",
        "filename": "FluidR3_GM_GS.sf2",
        "size_mb": 141,
        "description": "FluidR3 GM+GS merged - excellent quality, MIT license",
        "license": "MIT",
        "minimal": True,
    },
    "GeneralUser_GS": {
        "url": "https://musical-artifacts.com/artifacts/4625/GeneralUser_GS_v1.471.sf2",
        "filename": "GeneralUser_GS.sf2",
        "size_mb": 30,
        "description": "Compact GM soundfont with great drums (~30MB)",
        "license": "Free for any use",
        "minimal": False,
    },
    # Drum-specific soundfonts
    "Marching_Snare": {
        "url": "https://musical-artifacts.com/artifacts/1485/Marching_Snare_Drum_SF2.sf2",
        "filename": "Marching_Snare.sf2",
        "size_mb": 0.2,
        "description": "Marching snare drum - great for rudiments",
        "license": "CC BY",
        "minimal": False,
    },
    "MT_Power_DrumKit": {
        "url": "https://musical-artifacts.com/artifacts/6294/MT_PowerDrumKit.sf2",
        "filename": "MT_PowerDrumKit.sf2",
        "size_mb": 8.7,
        "description": "MT Power Drum Kit - punchy acoustic kit",
        "license": "Free",
        "minimal": False,
    },
    "Douglas_Natural_Studio": {
        "url": "https://archive.org/download/sf2soundfonts/Drums%20Douglas%20Natural%20Studio%20Kit%20V2.0%20%2822%2C719KB%29.sf2",
        "filename": "Douglas_Natural_Studio_Kit.sf2",
        "size_mb": 22,
        "description": "Natural Studio Kit - realistic acoustic drums",
        "license": "Free",
        "minimal": False,
    },
    "Sonic_Implants_Session": {
        "url": "https://archive.org/download/sf2soundfonts/Audio%20-%20Sound%20Font%20-%20Sonic%20Implants%20Session%20Drums.sf2",
        "filename": "Sonic_Implants_Session_Drums.sf2",
        "size_mb": 11,
        "description": "Session drums - studio quality",
        "license": "Free",
        "minimal": False,
    },
    "Definitive_Perfect_Drums": {
        "url": "https://musical-artifacts.com/artifacts/1540/The_Definitive_Perfect_Drums_Soundfount_V1___1-12_.sf2",
        "filename": "Perfect_Drums_V1.sf2",
        "size_mb": 244,
        "description": "Large high-quality drum kit (244MB)",
        "license": "Free",
        "minimal": False,
    },
}


def download_with_progress(url: str, dest: Path, expected_mb: float = 0) -> bool:
    """Download a file with progress indicator."""
    print(f"Downloading {dest.name}...")

    try:
        # Handle redirects (especially for archive.org)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(request, timeout=60)
        total_size = int(response.headers.get("content-length", 0))

        if total_size == 0 and expected_mb:
            total_size = int(expected_mb * 1024 * 1024)

        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"  {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="\r")
                else:
                    mb_done = downloaded / (1024 * 1024)
                    print(f"  {mb_done:.1f} MB downloaded...", end="\r")

        final_size = downloaded / (1024 * 1024)
        print(f"  Done: {dest.name} ({final_size:.1f} MB)          ")
        return True

    except Exception as e:
        print(f"  Error downloading: {e}")
        if dest.exists():
            dest.unlink()
        return False


def list_soundfonts():
    """List available soundfonts and their status."""
    soundfont_dir = Path(__file__).parent.parent / "data" / "soundfonts"

    print("\nAvailable Soundfonts (all verified working)")
    print("=" * 65)

    total_size = 0
    for name, info in SOUNDFONTS.items():
        filename = info["filename"]
        filepath = soundfont_dir / filename
        installed = filepath.exists()
        status = "INSTALLED" if installed else "available"
        marker = "*" if info.get("minimal") else " "

        print(f"\n{marker}[{status:>9}] {name}")
        print(f"   {info['description']}")
        print(f"   Size: ~{info['size_mb']} MB | License: {info['license']}")
        total_size += info["size_mb"]

    print("\n" + "=" * 65)
    print(f"Total if all downloaded: ~{total_size:.0f} MB")
    print("* = included in --minimal")

    # Show any installed soundfonts
    if soundfont_dir.exists():
        sf_files = list(soundfont_dir.glob("*.sf2"))
        if sf_files:
            print(f"\nCurrently installed in {soundfont_dir}:")
            for sf in sf_files:
                size_mb = sf.stat().st_size / (1024 * 1024)
                print(f"  {sf.name} ({size_mb:.1f} MB)")


def download_soundfonts(minimal: bool = False, names: list[str] | None = None):
    """Download soundfonts."""
    soundfont_dir = Path(__file__).parent.parent / "data" / "soundfonts"
    soundfont_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading soundfonts to: {soundfont_dir}")
    print("=" * 60)

    downloaded = 0
    skipped = 0
    failed = 0

    for name, info in SOUNDFONTS.items():
        # Filter by names if specified
        if names and name not in names:
            continue

        # Filter by minimal flag
        if minimal and not info.get("minimal", False):
            continue

        dest = soundfont_dir / info["filename"]

        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"[skip] {name} already exists ({size_mb:.1f} MB)")
            skipped += 1
            continue

        success = download_with_progress(info["url"], dest, info["size_mb"])
        if success:
            downloaded += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")

    # Show what we have
    sf_files = list(soundfont_dir.glob("*.sf2"))
    if sf_files:
        print("\nInstalled soundfonts:")
        total = 0
        for sf in sorted(sf_files):
            size_mb = sf.stat().st_size / (1024 * 1024)
            total += size_mb
            print(f"  {sf.name} ({size_mb:.1f} MB)")
        print(f"  Total: {total:.1f} MB")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download soundfonts for audio generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_soundfonts.py              # Download recommended set
  python scripts/setup_soundfonts.py --minimal    # Just FluidR3 (~141MB)
  python scripts/setup_soundfonts.py --list       # Show available soundfonts
  python scripts/setup_soundfonts.py --name Marching_Snare  # Download specific one
        """,
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available soundfonts and installation status",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Only download FluidR3_GM_GS (~141MB, good general purpose)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available soundfonts (~460MB total)",
    )
    parser.add_argument(
        "--name",
        "-n",
        action="append",
        dest="names",
        help="Download specific soundfont by name (can repeat)",
    )

    args = parser.parse_args()

    if args.list:
        list_soundfonts()
        return 0

    if args.names:
        # Download specific soundfonts
        valid_names = set(SOUNDFONTS.keys())
        for name in args.names:
            if name not in valid_names:
                print(f"Unknown soundfont: {name}")
                print(f"Available: {', '.join(sorted(valid_names))}")
                return 1
        success = download_soundfonts(names=args.names)
    elif args.all:
        success = download_soundfonts(minimal=False)
    elif args.minimal:
        success = download_soundfonts(minimal=True)
    else:
        # Default: download a good set for rudiments
        recommended = [
            "FluidR3_GM_GS",  # Good fallback with all instruments
            "Marching_Snare",  # Essential for rudiments
            "MT_Power_DrumKit",  # Good acoustic kit
        ]
        success = download_soundfonts(names=recommended)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
