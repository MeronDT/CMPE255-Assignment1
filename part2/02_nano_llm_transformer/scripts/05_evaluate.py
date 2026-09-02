"""CRISP-DM Phase 5: Evaluation.

Turns the base + SFT training histories into the charts an ML engineer
actually wants (loss curves, perplexity, LR schedule, throughput), and
generates qualitative sample completions from both checkpoints so coherence
can be judged by eye, not just by loss.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tokenizers import ByteLevelBPETokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.nanollama import NanoLlama, NanoLlamaConfig

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "eda"
CHECKPOINTS = ROOT / "checkpoints"
TOKENIZER_DIR = ROOT / "data" / "tokenizer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_PROMPTS = ["Once upon a time,", "The little cat", "One day, a boy named Tom"]
SFT_PROMPTS = [
    "What is the capital of France?",
    "Give three tips for staying healthy.",
    "Write a short poem about the ocean.",
]
SFT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def load_tokenizer():
    return ByteLevelBPETokenizer(str(TOKENIZER_DIR / "vocab.json"), str(TOKENIZER_DIR / "merges.txt"))


def load_model(run_tag):
    ckpt = torch.load(CHECKPOINTS / f"{run_tag}.pt", map_location=DEVICE, weights_only=False)
    cfg = NanoLlamaConfig(**ckpt["config"])
    model = NanoLlama(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def generate_samples(model, tokenizer, prompts, eot_id, max_new=120, strip_prompt=False):
    samples = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt).ids], device=DEVICE)
        prompt_len = ids.shape[1]
        out = model.generate(ids, max_new_tokens=max_new, temperature=0.8, top_p=0.95, eot_id=eot_id)
        tokens = out[0, prompt_len:].tolist() if strip_prompt else out[0].tolist()
        if tokens and tokens[-1] == eot_id:
            tokens = tokens[:-1]
        text = tokenizer.decode(tokens).strip()
        samples.append({"prompt": prompt, "completion": text})
    return samples


def plot_history(run_tag, title, color):
    with open(DOCS / f"train_history_{run_tag}.json") as f:
        hist = json.load(f)
    steps = hist["steps"]
    iters = [s["iter"] for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    fig.suptitle(title, fontweight="bold")

    axes[0].plot(iters, [s["train_loss"] for s in steps], label="train", color=color)
    axes[0].plot(iters, [s["val_loss"] for s in steps], label="val", color="#ef4444", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("iteration")
    axes[0].legend(fontsize=8)

    axes[1].plot(iters, [s["val_perplexity"] for s in steps], color=color)
    axes[1].set_yscale("log")
    axes[1].set_title("Validation Perplexity (log scale)")
    axes[1].set_xlabel("iteration")

    axes[2].plot(iters, [s["lr"] for s in steps], color="#f59e0b")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("iteration")

    plt.tight_layout()
    out_path = DOCS / f"evaluation_{run_tag}.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    return hist, out_path


def main():
    tokenizer = load_tokenizer()
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    base_hist, base_png = plot_history("base", "NanoLlama Base Pretraining (TinyStories)", "#6366f1")
    print(f"Wrote {base_png}")

    sft_hist, sft_png = plot_history("sft", "NanoLlama SFT (Alpaca instructions)", "#22c55e")
    print(f"Wrote {sft_png}")

    base_model, _ = load_model("base")
    base_samples = generate_samples(base_model, tokenizer, BASE_PROMPTS, eot_id, max_new=100)

    sft_model, _ = load_model("sft")
    sft_prompts_formatted = [SFT_TEMPLATE.format(instruction=p) for p in SFT_PROMPTS]
    sft_samples_raw = generate_samples(sft_model, tokenizer, sft_prompts_formatted, eot_id, max_new=120, strip_prompt=True)
    sft_samples = [{"prompt": SFT_PROMPTS[i], "completion": sft_samples_raw[i]["completion"]} for i in range(len(SFT_PROMPTS))]

    final_base = base_hist["steps"][-1]
    final_sft = sft_hist["steps"][-1]

    summary = {
        "base": {
            "final_train_loss": final_base["train_loss"],
            "final_val_loss": final_base["val_loss"],
            "final_val_perplexity": final_base["val_perplexity"],
            "total_train_seconds": base_hist["total_train_seconds"],
            "n_params": base_hist["n_params"],
            "samples": base_samples,
        },
        "sft": {
            "final_train_loss": final_sft["train_loss"],
            "final_val_loss": final_sft["val_loss"],
            "final_val_perplexity": final_sft["val_perplexity"],
            "total_train_seconds": sft_hist["total_train_seconds"],
            "samples": sft_samples,
        },
    }
    with open(DOCS / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Base model samples ===")
    for s in base_samples:
        print(f"PROMPT: {s['prompt']}\n  -> {s['completion']}\n")
    print("=== SFT model samples ===")
    for s in sft_samples:
        print(f"PROMPT: {s['prompt']}\n  -> {s['completion']}\n")

    print(f"\nBase: val_loss={final_base['val_loss']:.4f} ppl={final_base['val_perplexity']:.2f}")
    print(f"SFT:  val_loss={final_sft['val_loss']:.4f} ppl={final_sft['val_perplexity']:.2f}")


if __name__ == "__main__":
    main()
