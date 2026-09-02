"""CRISP-DM Phase 4a: Modeling — base (pretraining) run.

Trains NanoLlama from scratch on TinyStories token stream. AdamW + cosine LR
schedule with linear warmup, bf16 autocast (Ampere+ native support, no
GradScaler needed), gradient clipping. Full step-by-step history (loss, val
loss, lr, tokens/sec, GPU memory) logged to JSON for the admin dashboard —
not just the final numbers.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.nanollama import NanoLlama, NanoLlamaConfig

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
DOCS = ROOT / "docs" / "eda"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # calibrated against this GPU's 4GB VRAM ceiling — see DESIGN_DOC §5
MAX_ITERS = 3000
WARMUP_ITERS = 200
LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 150
EVAL_ITERS = 40
LOG_INTERVAL = 25


def load_bin(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, context_length, batch_size, device):
    ix = torch.randint(len(data) - context_length - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def lr_at(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup) / max(1, (max_iters - warmup))
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, eval_iters):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, cfg.context_length, BATCH_SIZE, DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main(max_iters=MAX_ITERS, config_overrides=None, run_tag="base"):
    cfg = NanoLlamaConfig()
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)

    train_data = load_bin(PROCESSED / "pretrain_train.bin")
    val_data = load_bin(PROCESSED / "pretrain_val.bin")

    model = NanoLlama(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    history = {
        "run_tag": run_tag,
        "config": cfg.__dict__,
        "n_params": n_params,
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "max_iters": max_iters,
        "steps": [],
    }

    t_start = time.time()
    tokens_seen = 0
    for it in range(max_iters + 1):
        lr = lr_at(it, WARMUP_ITERS, max_iters, LR, MIN_LR)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % EVAL_INTERVAL == 0 or it == max_iters:
            losses = estimate_loss(model, train_data, val_data, cfg, EVAL_ITERS)
            elapsed = time.time() - t_start
            mem_mb = torch.cuda.memory_allocated() / 1e6 if DEVICE == "cuda" else 0
            entry = {
                "iter": it,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "val_perplexity": float(np.exp(losses["val"])),
                "lr": lr,
                "elapsed_sec": elapsed,
                "tokens_seen": tokens_seen,
                "tokens_per_sec": tokens_seen / elapsed if elapsed > 0 else 0,
                "gpu_mem_mb": mem_mb,
            }
            history["steps"].append(entry)
            print(f"[{run_tag}] iter {it}/{max_iters} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} "
                  f"val_ppl={entry['val_perplexity']:.2f} lr={lr:.2e} tok/s={entry['tokens_per_sec']:.0f} mem={mem_mb:.0f}MB")

        if it == max_iters:
            break

        x, y = get_batch(train_data, cfg.context_length, BATCH_SIZE, DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        tokens_seen += x.numel()

    history["total_train_seconds"] = time.time() - t_start
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, CHECKPOINTS / f"{run_tag}.pt")
    with open(DOCS / f"train_history_{run_tag}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[{run_tag}] done in {history['total_train_seconds']:.1f}s -> {CHECKPOINTS / f'{run_tag}.pt'}")
    return history


if __name__ == "__main__":
    main()
