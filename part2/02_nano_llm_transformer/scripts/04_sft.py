"""CRISP-DM Phase 4b: Modeling — Supervised Fine-Tuning.

Continues training from the pretrained base checkpoint on Alpaca instruction/
response pairs, with loss computed only on response tokens (loss_mask from
02_prepare_data.py) — the model shouldn't be trained to "predict" the fixed
instruction template, only to generate good responses. Lower LR than
pretraining (standard SFT practice: adapt, don't overwrite, what pretraining
learned) and far fewer steps (small instruction set, easy to overfit).
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # calibrated against this GPU's 4GB VRAM ceiling for the 4-layer model — see DESIGN_DOC §5
MAX_ITERS = 4500  # ~2.8 epochs over the full 51K-example Alpaca train split (previously 600 steps over an 8K
# subset was <1.3 epochs — nowhere near enough exposure for instruction-tuning, root cause of low-quality
# generations reported in RESEARCH_REPORT.md; see the follow-up audit note there for the full diagnosis)
WARMUP_ITERS = 300
LR = 1e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
EVAL_INTERVAL = 225
LOG_INTERVAL = 25


def get_batch(ids, mask, batch_size, device):
    idx = np.random.randint(0, len(ids), size=batch_size)
    x = torch.from_numpy(ids[idx, :-1].astype(np.int64))
    y = torch.from_numpy(ids[idx, 1:].astype(np.int64))
    m = torch.from_numpy(mask[idx, 1:].astype(np.int64))
    return x.to(device), y.to(device), m.to(device)


def lr_at(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup) / max(1, (max_iters - warmup))
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model, ids, mask, eval_batches=10):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y, m = get_batch(ids, mask, BATCH_SIZE, DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            _, loss = model(x, y, loss_mask=m)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main(base_checkpoint="base", max_iters=MAX_ITERS, run_tag="sft"):
    train_ids = np.load(PROCESSED / "sft_train_ids.npy")
    train_mask = np.load(PROCESSED / "sft_train_mask.npy")
    val_ids = np.load(PROCESSED / "sft_val_ids.npy")
    val_mask = np.load(PROCESSED / "sft_val_mask.npy")

    ckpt = torch.load(CHECKPOINTS / f"{base_checkpoint}.pt", map_location=DEVICE, weights_only=False)
    cfg = NanoLlamaConfig(**ckpt["config"])
    model = NanoLlama(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    history = {"run_tag": run_tag, "base_checkpoint": base_checkpoint, "config": cfg.__dict__, "max_iters": max_iters, "steps": []}
    t_start = time.time()

    for it in range(max_iters + 1):
        lr = lr_at(it, WARMUP_ITERS, max_iters, LR, MIN_LR)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % EVAL_INTERVAL == 0 or it == max_iters:
            train_loss = estimate_loss(model, train_ids, train_mask)
            val_loss = estimate_loss(model, val_ids, val_mask)
            elapsed = time.time() - t_start
            entry = {
                "iter": it, "train_loss": train_loss, "val_loss": val_loss,
                "val_perplexity": float(np.exp(val_loss)), "lr": lr, "elapsed_sec": elapsed,
                "gpu_mem_mb": torch.cuda.memory_allocated() / 1e6 if DEVICE == "cuda" else 0,
            }
            history["steps"].append(entry)
            print(f"[{run_tag}] iter {it}/{max_iters} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={entry['val_perplexity']:.2f} lr={lr:.2e}")

        if it == max_iters:
            break

        x, y, m = get_batch(train_ids, train_mask, BATCH_SIZE, DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            _, loss = model(x, y, loss_mask=m)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    history["total_train_seconds"] = time.time() - t_start
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, CHECKPOINTS / f"{run_tag}.pt")
    with open(DOCS / f"train_history_{run_tag}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[{run_tag}] done in {history['total_train_seconds']:.1f}s -> {CHECKPOINTS / f'{run_tag}.pt'}")
    return history


if __name__ == "__main__":
    main()
