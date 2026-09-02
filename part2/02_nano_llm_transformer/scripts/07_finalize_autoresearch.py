"""Retrain AutoResearch's recommended configuration at the FULL training
budget (3000 steps, full context length) and compare against the currently
deployed base model on real validation data — mirroring the taxi project's
honest search-then-finalize pattern. Redeploys only if it's actually better.
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

FULL_CONTEXT_LENGTH = 384
MAX_ITERS = 3000
WARMUP_ITERS = 200
MIN_LR_FRACTION = 0.1
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 300
EVAL_ITERS = 40


def load_bin(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, context_length, batch_size, device):
    ix = np.random.randint(0, len(data) - context_length - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i : i + context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_at(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup) / max(1, (max_iters - warmup))
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def main():
    with open(DOCS / "autoresearch_history.json") as f:
        autoresearch = json.load(f)
    with open(DOCS / "train_history_base.json") as f:
        current_history = json.load(f)

    rec = autoresearch["final_recommended_config"]
    cfg = NanoLlamaConfig(d_model=rec["d_model"], n_layers=rec["n_layers"], n_heads=rec["n_heads"], context_length=FULL_CONTEXT_LENGTH)
    lr = rec["lr"]
    batch_size = rec["batch_size"]
    warmup_iters = max(WARMUP_ITERS, int(rec["warmup_frac"] * MAX_ITERS))
    min_lr = lr * MIN_LR_FRACTION

    current_val_loss = current_history["steps"][-1]["val_loss"]

    train_data = load_bin(PROCESSED / "pretrain_train.bin")
    val_data = load_bin(PROCESSED / "pretrain_val.bin")

    model = NanoLlama(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    print(f"Finalizing AutoResearch config: d_model={cfg.d_model} n_layers={cfg.n_layers} n_heads={cfg.n_heads} "
          f"({n_params/1e6:.1f}M params), lr={lr:.2e} batch_size={batch_size} warmup_iters={warmup_iters}")

    t0 = time.time()
    final_val_loss = None
    for it in range(MAX_ITERS + 1):
        cur_lr = lr_at(it, warmup_iters, MAX_ITERS, lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(EVAL_ITERS):
                    x, y = get_batch(val_data, cfg.context_length, batch_size, DEVICE)
                    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                        _, loss = model(x, y)
                    losses.append(loss.item())
            final_val_loss = float(np.mean(losses))
            model.train()
            print(f"  iter {it}/{MAX_ITERS} val_loss={final_val_loss:.4f} val_ppl={np.exp(final_val_loss):.2f}")

        if it == MAX_ITERS:
            break

        x, y = get_batch(train_data, cfg.context_length, batch_size, DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    elapsed = time.time() - t0
    redeployed = final_val_loss < current_val_loss

    print(f"\ncurrent production val_loss={current_val_loss:.4f}  autoresearch-config val_loss={final_val_loss:.4f} "
          f"-> {'REDEPLOYED' if redeployed else 'kept existing production model'}")

    if redeployed:
        torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, CHECKPOINTS / "base.pt")

    finalize_report = {
        "autoresearch_config": rec,
        "n_params": n_params,
        "train_seconds": elapsed,
        "previous_val_loss": current_val_loss,
        "finalized_val_loss": final_val_loss,
        "redeployed": redeployed,
    }
    with open(DOCS / "autoresearch_finalize.json", "w") as f:
        json.dump(finalize_report, f, indent=2)
    print(f"Wrote {DOCS / 'autoresearch_finalize.json'}")


if __name__ == "__main__":
    main()
