"""CRISP-DM Phase 3: Data Preparation.

1. Trains a byte-level BPE tokenizer (vocab_size=8192, justified by EDA's
   ~6.4K-word vocabulary in a 5K-story sample — TinyStories' restricted
   vocabulary means we don't need GPT-2's 50K vocab) on the TinyStories corpus.
2. Tokenizes TinyStories into a single flat uint16 token stream, split
   train/val, saved as memmap-friendly .bin files (nanoGPT-style) for
   pretraining.
3. Tokenizes Alpaca into instruction-formatted sequences for SFT, with a
   loss mask marking which tokens are "response" (trainable) vs "prompt"
   (context only, not backpropagated through) — standard SFT practice, since
   we want the model to learn to *generate* responses, not to predict the
   fixed instruction template.
"""

import json
from pathlib import Path

import numpy as np
from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TOKENIZER_DIR = ROOT / "data" / "tokenizer"
PROCESSED.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 8192
CONTEXT_LENGTH = 384
VAL_FRACTION = 0.02

ALPACA_TEMPLATE_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"
ALPACA_TEMPLATE_WITH_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


def train_tokenizer():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(RAW / "tinystories.txt")],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )
    tokenizer.save_model(str(TOKENIZER_DIR))
    with open(TOKENIZER_DIR / "config.json", "w") as f:
        json.dump({"vocab_size": VOCAB_SIZE, "context_length": CONTEXT_LENGTH}, f, indent=2)
    print(f"Trained tokenizer, vocab_size={tokenizer.get_vocab_size()} -> {TOKENIZER_DIR}")
    return tokenizer


def prepare_pretrain_data(tokenizer):
    text = (RAW / "tinystories.txt").read_text(encoding="utf-8")
    stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]

    eot_id = tokenizer.token_to_id("<|endoftext|>")
    all_ids = []
    for story in stories:
        ids = tokenizer.encode(story).ids
        all_ids.extend(ids)
        all_ids.append(eot_id)

    arr = np.array(all_ids, dtype=np.uint16)
    n_val = int(len(arr) * VAL_FRACTION)
    train_arr, val_arr = arr[:-n_val], arr[-n_val:]

    train_arr.tofile(PROCESSED / "pretrain_train.bin")
    val_arr.tofile(PROCESSED / "pretrain_val.bin")

    print(f"Pretrain tokens: train={len(train_arr):,}  val={len(val_arr):,}")
    return {"n_train_tokens": int(len(train_arr)), "n_val_tokens": int(len(val_arr)), "n_stories": len(stories)}


def prepare_sft_data(tokenizer):
    data = json.loads((RAW / "alpaca_data.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(42)
    rng.shuffle(data)
    subset = data  # full Alpaca set — see DESIGN_DOC §5 for why the earlier 8K-example subset was insufficient
    n_val = 1000
    train_examples, val_examples = subset[n_val:], subset[:n_val]

    pad_id = tokenizer.token_to_id("<|pad|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    def encode_split(examples):
        input_ids_list, loss_mask_list = [], []
        for ex in examples:
            prompt = (
                ALPACA_TEMPLATE_WITH_INPUT.format(instruction=ex["instruction"], input=ex["input"])
                if ex["input"].strip()
                else ALPACA_TEMPLATE_NO_INPUT.format(instruction=ex["instruction"])
            )
            prompt_ids = tokenizer.encode(prompt).ids
            response_ids = tokenizer.encode(ex["output"]).ids + [eot_id]

            ids = (prompt_ids + response_ids)[:CONTEXT_LENGTH]
            mask = ([0] * len(prompt_ids) + [1] * len(response_ids))[:CONTEXT_LENGTH]

            pad_len = CONTEXT_LENGTH - len(ids)
            ids = ids + [pad_id] * pad_len
            mask = mask + [0] * pad_len

            input_ids_list.append(ids)
            loss_mask_list.append(mask)
        return np.array(input_ids_list, dtype=np.uint16), np.array(loss_mask_list, dtype=np.uint8)

    train_ids, train_mask = encode_split(train_examples)
    val_ids, val_mask = encode_split(val_examples)

    np.save(PROCESSED / "sft_train_ids.npy", train_ids)
    np.save(PROCESSED / "sft_train_mask.npy", train_mask)
    np.save(PROCESSED / "sft_val_ids.npy", val_ids)
    np.save(PROCESSED / "sft_val_mask.npy", val_mask)

    print(f"SFT examples: train={len(train_ids):,}  val={len(val_ids):,}  (of {len(data):,} available, subset for time budget)")
    return {"n_train_examples": len(train_ids), "n_val_examples": len(val_ids), "n_available_total": len(data)}


def main():
    tokenizer = train_tokenizer()
    pretrain_stats = prepare_pretrain_data(tokenizer)
    sft_stats = prepare_sft_data(tokenizer)

    summary = {
        "vocab_size": VOCAB_SIZE,
        "context_length": CONTEXT_LENGTH,
        "pretrain": pretrain_stats,
        "sft": sft_stats,
    }
    with open(ROOT / "docs" / "eda" / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
