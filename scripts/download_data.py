#!/usr/bin/env python3
"""Dataset Downloader for UJIIndoorLoc (UCI Machine Learning Repository).

Downloads the official UJIIndoorLoc dataset from UCI ML Repository:
  DOI: 10.24432/C5MS59
  URL: https://archive.ics.uci.edu/dataset/310/ujiindoorloc
  Direct Zip: https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip

Member 1 (Data & Infrastructure Lead) Deliverable.
"""
import argparse
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path
import urllib.request

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_DIR, ROOT

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
EXPECTED_FILES = ["trainingData.csv", "validationData.csv"]
LOCAL_FALLBACK_DIR = Path.home() / "Downloads" / "UJIndoorLoc"


def verify_files(target_dir: Path) -> bool:
    """Check if all required CSV files exist and are non-empty."""
    for fname in EXPECTED_FILES:
        fpath = target_dir / fname
        if not fpath.exists() or fpath.stat().st_size < 1000:
            return False
    return True


def copy_from_local(target_dir: Path) -> bool:
    """Copy from local Downloads folder if available (fast path)."""
    if not LOCAL_FALLBACK_DIR.exists():
        return False
    found_all = True
    for fname in EXPECTED_FILES:
        src_file = LOCAL_FALLBACK_DIR / fname
        if not src_file.exists():
            found_all = False
            break
    if found_all:
        print(f"[download_data] Found local cached data in {LOCAL_FALLBACK_DIR}")
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in EXPECTED_FILES:
            src_file = LOCAL_FALLBACK_DIR / fname
            dest_file = target_dir / fname
            shutil.copy2(src_file, dest_file)
            print(f"  Copied {fname} -> {dest_file} ({dest_file.stat().st_size / 1e6:.1f} MB)")
        return True
    return False


def download_and_extract(target_dir: Path):
    """Download ZIP from UCI and extract trainingData.csv and validationData.csv."""
    print(f"[download_data] Downloading dataset from {UCI_ZIP_URL} ...")
    target_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        UCI_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0 (UJIIndoorLoc-Downloader/1.0)"}
    )
    with urllib.request.urlopen(req) as resp:
        zip_bytes = resp.read()
    print(f"[download_data] Downloaded {len(zip_bytes) / 1e6:.2f} MB. Extracting ...")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Some UCI archives nest files inside subfolders or zip within zip
        for member in zf.namelist():
            base = Path(member).name
            if base in EXPECTED_FILES:
                with zf.open(member) as sf, open(target_dir / base, "wb") as df:
                    shutil.copyfileobj(sf, df)
                print(f"  Extracted: {base}")
            elif base.endswith(".zip"):
                # Nested zip check (e.g. UJIndoorLoc.zip inside archive)
                sub_bytes = zf.read(member)
                with zipfile.ZipFile(io.BytesIO(sub_bytes)) as sub_zf:
                    for sub_member in sub_zf.namelist():
                        sub_base = Path(sub_member).name
                        if sub_base in EXPECTED_FILES:
                            with sub_zf.open(sub_member) as sf, open(target_dir / sub_base, "wb") as df:
                                shutil.copyfileobj(sf, df)
                            print(f"  Extracted from nested zip: {sub_base}")


def main():
    parser = argparse.ArgumentParser(description="Download UJIIndoorLoc dataset from UCI repository")
    parser.add_argument("--dest", type=str, default=str(DATA_DIR), help="Target data directory")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()

    dest_path = Path(args.dest)
    print(f"Checking dataset in: {dest_path}")

    if not args.force and verify_files(dest_path):
        print("Dataset files already exist and verified:")
        for fname in EXPECTED_FILES:
            f = dest_path / fname
            print(f"  ✓ {fname} ({f.stat().st_size / 1e6:.1f} MB)")
        return

    # Try local copy first if not forcing download
    if not args.force and copy_from_local(dest_path):
        print("Dataset restored from local cache successfully.")
        return

    # Download from UCI repository
    download_and_extract(dest_path)

    if verify_files(dest_path):
        print("\n✓ Dataset download and verification completed successfully!")
    else:
        print("\n✗ Error: Failed to extract all required dataset files.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
