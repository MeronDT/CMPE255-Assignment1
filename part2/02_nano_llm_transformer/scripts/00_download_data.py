"""CRISP-DM Phase 2 (data collection step): download raw source data.

TinyStories validation split (~22.5MB, 27,630 stories) is used as the full
pretraining corpus rather than the 2GB train split — plenty of data at this
model's scale (see RESEARCH_REPORT.md), and much faster to fetch.
"""

import json
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

FILES = {
    "tinystories.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "alpaca_data.json": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
}


def main():
    for filename, url in FILES.items():
        dest = RAW / filename
        if dest.exists():
            print(f"Already have {filename}, skipping")
            continue
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)

    n_stories = (RAW / "tinystories.txt").read_text(encoding="utf-8").count("<|endoftext|>")
    n_alpaca = len(json.loads((RAW / "alpaca_data.json").read_text(encoding="utf-8")))
    print(f"TinyStories: {n_stories:,} stories")
    print(f"Alpaca: {n_alpaca:,} instruction/response pairs")
    print("Done. Next: python scripts/01_eda.py")


if __name__ == "__main__":
    main()
