"""CRISP-DM Phase 2: Data Understanding.

Profiles both raw corpora at the text level (before any tokenizer exists):
length distributions, vocabulary richness, and structural stats that justify
Phase 3's tokenizer vocab size and context-length choices.
"""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOCS = ROOT / "docs" / "eda"
DOCS.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"[A-Za-z']+")


def analyze_tinystories():
    text = (RAW / "tinystories.txt").read_text(encoding="utf-8")
    stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
    lengths_chars = [len(s) for s in stories]
    lengths_words = [len(WORD_RE.findall(s)) for s in stories]

    vocab = Counter()
    for s in stories[:5000]:  # sample for vocab richness (full corpus vocab computed at tokenizer stage)
        vocab.update(w.lower() for w in WORD_RE.findall(s))

    return {
        "n_stories": len(stories),
        "total_chars": len(text),
        "avg_chars_per_story": sum(lengths_chars) / len(lengths_chars),
        "avg_words_per_story": sum(lengths_words) / len(lengths_words),
        "max_words_per_story": max(lengths_words),
        "unique_words_sample_5000_stories": len(vocab),
        "top_20_words": vocab.most_common(20),
    }, lengths_words


def analyze_alpaca():
    data = json.loads((RAW / "alpaca_data.json").read_text(encoding="utf-8"))
    instr_lens = [len(WORD_RE.findall(d["instruction"])) for d in data]
    output_lens = [len(WORD_RE.findall(d["output"])) for d in data]
    with_input = sum(1 for d in data if d["input"].strip())

    return {
        "n_examples": len(data),
        "n_with_input_field": with_input,
        "avg_instruction_words": sum(instr_lens) / len(instr_lens),
        "avg_output_words": sum(output_lens) / len(output_lens),
        "max_output_words": max(output_lens),
    }, output_lens


def main():
    ts_stats, ts_word_lens = analyze_tinystories()
    alpaca_stats, alpaca_output_lens = analyze_alpaca()

    findings = {"tinystories": ts_stats, "alpaca": alpaca_stats}
    with open(DOCS / "eda_findings.json", "w") as f:
        json.dump(findings, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist([w for w in ts_word_lens if w <= 400], bins=60, color="#6366f1")
    axes[0].set_title(f"TinyStories: words/story (n={ts_stats['n_stories']:,})")
    axes[0].set_xlabel("words")

    axes[1].hist([w for w in alpaca_output_lens if w <= 300], bins=60, color="#22c55e")
    axes[1].set_title(f"Alpaca: words/response (n={alpaca_stats['n_examples']:,})")
    axes[1].set_xlabel("words")

    plt.tight_layout()
    plt.savefig(DOCS / "eda_overview.png", dpi=130)
    plt.close(fig)

    print(json.dumps(findings, indent=2))
    print(f"\nWrote -> {DOCS / 'eda_findings.json'}, {DOCS / 'eda_overview.png'}")


if __name__ == "__main__":
    main()
