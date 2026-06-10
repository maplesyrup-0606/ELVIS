"""
Download the ELVIS dataset from HuggingFace to ELVIS_DATA (or a specified path).

Usage:
    python -m scripts.download_data
    python -m scripts.download_data --output /scratch/merc0606/ELVIS/data
    python -m scripts.download_data --principles proximity similarity
"""
import argparse
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "akweury/ELVIS"
PRINCIPLES = ["proximity", "similarity", "closure", "symmetry", "continuity"]
MAX_RETRIES = 5
RETRY_DELAY = 120  # HF rate limit resets every 5 minutes


def download(output_dir: Path, principles: list[str]):
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = [f"{p}/*" for p in principles] + ["README.md"]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading {principles} → {output_dir}  (attempt {attempt}/{MAX_RETRIES})")
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=str(output_dir),
                allow_patterns=patterns,
                ignore_patterns=["*.DS_Store", "llm_pretrained/*"],
            )
            print("Download complete.")
            return
        except HfHubHTTPError as e:
            if "429" in str(e) and attempt < MAX_RETRIES:
                print(f"Rate limited. Waiting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)
            else:
                raise

    raise RuntimeError(f"Download failed after {MAX_RETRIES} attempts.")


def verify(output_dir: Path, principles: list[str]):
    missing = []
    for principle in principles:
        test_dir = output_dir / principle / "test"
        if not test_dir.exists() or not any(test_dir.iterdir()):
            missing.append(principle)
    if missing:
        print(f"WARNING: missing or empty test dirs for: {missing}")
    else:
        print(f"Verified: all {len(principles)} principles present.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=os.getenv("ELVIS_DATA", str(Path(__file__).parents[1] / "data")),
        help="Download destination (defaults to $ELVIS_DATA or repo/data/)",
    )
    parser.add_argument(
        "--principles",
        nargs="+",
        default=PRINCIPLES,
        choices=PRINCIPLES,
        help="Which principles to download (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    download(output_dir, args.principles)
    verify(output_dir, args.principles)


if __name__ == "__main__":
    main()
