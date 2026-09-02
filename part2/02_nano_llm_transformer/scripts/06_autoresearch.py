"""AutoResearch: 4-phase autonomous search over NanoLlama's design space, in
the same spirit as the taxi project's AutoResearch pipeline (see
../01_nyc_taxi_trip_prediction/scripts/05_autoresearch.py) — adapted here for
an LLM instead of gradient-boosted trees. Every trial is a short proxy
training run (short step budget, same small corpus) so the whole search
finishes in minutes; the winning configuration is then trained for the real
step budget in scripts/07_finalize_autoresearch.py.

Phase 1 — Architecture-primitive tournament: empirically test each "state of
  the art primitive" claim from RESEARCH_REPORT.md against its pre-2020
  alternative (RoPE vs. learned position embeddings, SwiGLU vs. GELU MLP,
  RMSNorm vs. LayerNorm), rather than just asserting modern = better.
Phase 2 — Model-shape search: width vs. depth trade-off at a fixed
  (approximate) parameter budget.
Phase 3 — Hyperparameter hill-climbing (literal greedy local search, not
  random/grid): from a seed learning-rate/batch-size/warmup config, evaluate
  every one-step neighbor, move to the best improving one, repeat to a local
  optimum.
Phase 4 — Model soup (Wortsman et al. 2022, "Model soups: averaging weights
  of multiple fine-tuned models improves accuracy without increasing
  inference time"): weight-average the top-2 checkpoints from Phase 3 and
  check whether the soup beats either ingredient.
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.nanollama import NanoLlama, NanoLlamaConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"

PROXY_ITERS = 250
PROXY_BATCH = 24  # calibrated against this GPU's 4GB VRAM ceiling — see DESIGN_DOC §5
PROXY_CONTEXT = 256
PROXY_EVAL_ITERS = 20


def load_bin(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, context_length, batch_size, device):
    ix = np.random.randint(0, len(data) - context_length - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i : i + context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def proxy_train(cfg: NanoLlamaConfig, train_data, val_data, lr=3e-4, iters=PROXY_ITERS, batch_size=PROXY_BATCH):
    torch.manual_seed(42)
    model = NanoLlama(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    t0 = time.time()

    for it in range(iters):
        frac = it / iters
        cur_lr = lr * min(1.0, (it + 1) / max(1, int(0.1 * iters)))
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        x, y = get_batch(train_data, cfg.context_length, batch_size, DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_losses = []
        for _ in range(PROXY_EVAL_ITERS):
            x, y = get_batch(val_data, cfg.context_length, batch_size, DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                _, loss = model(x, y)
            val_losses.append(loss.item())
    val_loss = float(np.mean(val_losses))
    elapsed = time.time() - t0
    del model, optimizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return val_loss, elapsed


# ---------------------------------------------------------------------------
# Phase 1: Architecture-primitive tournament
# ---------------------------------------------------------------------------
def phase1_architecture_tournament(train_data, val_data):
    variants = {
        "nanollama (rope+rmsnorm+swiglu)": {},
        "learned_pos_embeddings": {"pos_type": "learned"},
        "layernorm_instead_of_rmsnorm": {"norm_type": "layernorm"},
        "gelu_mlp_instead_of_swiglu": {"mlp_type": "gelu"},
    }
    results = []
    for name, overrides in variants.items():
        cfg = NanoLlamaConfig(context_length=PROXY_CONTEXT, d_model=384, n_layers=6, n_heads=6, **overrides)
        val_loss, elapsed = proxy_train(cfg, train_data, val_data)
        n_params = cfg.n_params()
        results.append({"variant": name, "val_loss": val_loss, "val_perplexity": float(np.exp(val_loss)), "n_params": n_params, "seconds": round(elapsed, 1)})
        print(f"    [arch-tournament] {name}: val_loss={val_loss:.4f} ppl={np.exp(val_loss):.2f} ({elapsed:.1f}s)")
    winner = min(results, key=lambda r: r["val_loss"])
    return results, winner["variant"]


# ---------------------------------------------------------------------------
# Phase 2: Width vs. depth search at ~fixed parameter budget
# ---------------------------------------------------------------------------
def phase2_shape_search(train_data, val_data):
    shapes = [
        {"d_model": 256, "n_layers": 12, "n_heads": 8},   # deep & narrow
        {"d_model": 384, "n_layers": 6, "n_heads": 6},     # balanced
        {"d_model": 512, "n_layers": 4, "n_heads": 8},     # wide & shallow
        {"d_model": 512, "n_layers": 8, "n_heads": 8},     # deployed default
    ]
    results = []
    for shape in shapes:
        cfg = NanoLlamaConfig(context_length=PROXY_CONTEXT, **shape)
        val_loss, elapsed = proxy_train(cfg, train_data, val_data)
        n_params = cfg.n_params()
        label = f"d{shape['d_model']}_L{shape['n_layers']}_h{shape['n_heads']}"
        results.append({"shape": label, **shape, "val_loss": val_loss, "val_perplexity": float(np.exp(val_loss)), "n_params": n_params, "seconds": round(elapsed, 1)})
        print(f"    [shape-search] {label} ({n_params/1e6:.1f}M params): val_loss={val_loss:.4f} ppl={np.exp(val_loss):.2f}")
    winner = min(results, key=lambda r: r["val_loss"])
    return results, winner["shape"], {k: winner[k] for k in ("d_model", "n_layers", "n_heads")}


# ---------------------------------------------------------------------------
# Phase 3: Hyperparameter hill-climbing (greedy local search)
# ---------------------------------------------------------------------------
PARAM_STEPS = {"lr_log10": [-0.3, 0.3], "batch_size": [-8, 8], "warmup_frac": [-0.03, 0.03]}
PARAM_BOUNDS = {"lr_log10": (-4.3, -2.7), "batch_size": (8, 32), "warmup_frac": (0.02, 0.25)}


def clip(param, value):
    lo, hi = PARAM_BOUNDS[param]
    return max(lo, min(hi, value))


def eval_hparam_config(cfg, params, train_data, val_data):
    lr = 10 ** params["lr_log10"]
    batch_size = int(params["batch_size"])
    val_loss, elapsed = proxy_train(cfg, train_data, val_data, lr=lr, batch_size=batch_size)
    return val_loss, elapsed


def phase3_hill_climb(cfg, seed_params, train_data, val_data, max_iters=6):
    path = []
    current = dict(seed_params)
    current_loss, _ = eval_hparam_config(cfg, current, train_data, val_data)
    path.append({"iteration": 0, "params": dict(current), "val_loss": current_loss, "move": "seed"})
    print(f"    [hill-climb] seed val_loss={current_loss:.4f} params={current}")

    for it in range(1, max_iters + 1):
        best_candidate, best_loss = None, current_loss
        for param, steps in PARAM_STEPS.items():
            for step in steps:
                candidate = dict(current)
                candidate[param] = clip(param, round(current[param] + step, 4))
                if candidate == current:
                    continue
                loss, _ = eval_hparam_config(cfg, candidate, train_data, val_data)
                if loss < best_loss:
                    best_candidate, best_loss = candidate, loss

        if best_candidate is None:
            print(f"    [hill-climb] iteration {it}: no improving neighbor -> local optimum reached")
            break
        current, current_loss = best_candidate, best_loss
        path.append({"iteration": it, "params": dict(current), "val_loss": current_loss, "move": "accepted"})
        print(f"    [hill-climb] iteration {it}: val_loss={current_loss:.4f} params={current}")

    return current, current_loss, path


# ---------------------------------------------------------------------------
# Phase 4: Model soup (weight averaging)
# ---------------------------------------------------------------------------
def phase4_model_soup(cfg, params_a, params_b, train_data, val_data):
    torch.manual_seed(42)
    model_a = NanoLlama(cfg).to(DEVICE)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=10 ** params_a["lr_log10"], weight_decay=0.1, betas=(0.9, 0.95))
    for it in range(PROXY_ITERS):
        lr = (10 ** params_a["lr_log10"]) * min(1.0, (it + 1) / max(1, int(0.1 * PROXY_ITERS)))
        for g in opt_a.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_data, cfg.context_length, int(params_a["batch_size"]), DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            _, loss = model_a(x, y)
        opt_a.zero_grad(set_to_none=True)
        loss.backward()
        opt_a.step()

    del opt_a  # free model_a's optimizer state before model_b trains — both models'
    if DEVICE == "cuda":  # weights alone (no optimizer) safely co-reside under 4GB, but
        torch.cuda.empty_cache()  # two full Adam optimizer states at once would not.

    torch.manual_seed(123)
    model_b = NanoLlama(cfg).to(DEVICE)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=10 ** params_b["lr_log10"], weight_decay=0.1, betas=(0.9, 0.95))
    for it in range(PROXY_ITERS):
        lr = (10 ** params_b["lr_log10"]) * min(1.0, (it + 1) / max(1, int(0.1 * PROXY_ITERS)))
        for g in opt_b.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_data, cfg.context_length, int(params_b["batch_size"]), DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            _, loss = model_b(x, y)
        opt_b.zero_grad(set_to_none=True)
        loss.backward()
        opt_b.step()

    def eval_model(model):
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(PROXY_EVAL_ITERS):
                x, y = get_batch(val_data, cfg.context_length, PROXY_BATCH, DEVICE)
                with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                    _, loss = model(x, y)
                losses.append(loss.item())
        return float(np.mean(losses))

    loss_a, loss_b = eval_model(model_a), eval_model(model_b)

    soup = NanoLlama(cfg).to(DEVICE)
    with torch.no_grad():
        for p_soup, p_a, p_b in zip(soup.parameters(), model_a.parameters(), model_b.parameters()):
            p_soup.copy_(0.5 * p_a + 0.5 * p_b)
    loss_soup = eval_model(soup)

    del model_a, model_b, soup
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "model_a_val_loss": loss_a, "model_b_val_loss": loss_b, "soup_val_loss": loss_soup,
        "soup_beats_both": loss_soup < min(loss_a, loss_b),
    }


def main():
    train_data = load_bin(PROCESSED / "pretrain_train.bin")
    val_data = load_bin(PROCESSED / "pretrain_val.bin")

    print("Phase 1: Architecture-primitive tournament")
    arch_results, arch_winner = phase1_architecture_tournament(train_data, val_data)

    print("\nPhase 2: Width vs. depth shape search")
    shape_results, shape_winner_label, shape_winner_dims = phase2_shape_search(train_data, val_data)

    print("\nPhase 3: Hyperparameter hill-climbing")
    cfg = NanoLlamaConfig(context_length=PROXY_CONTEXT, **shape_winner_dims)
    seed_params = {"lr_log10": np.log10(3e-4), "batch_size": 24, "warmup_frac": 0.08}
    best_params, best_loss, hillclimb_path = phase3_hill_climb(cfg, seed_params, train_data, val_data)

    print("\nPhase 4: Model soup (top hill-climb config x seed config)")
    soup_result = phase4_model_soup(cfg, best_params, seed_params, train_data, val_data)
    print(f"    [soup] model_a={soup_result['model_a_val_loss']:.4f} model_b={soup_result['model_b_val_loss']:.4f} "
          f"soup={soup_result['soup_val_loss']:.4f} beats_both={soup_result['soup_beats_both']}")

    report = {
        "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
        "methodology": "4-phase AutoResearch: architecture tournament -> shape search -> hyperparameter hill-climbing -> model soup",
        "proxy_budget": {"iters": PROXY_ITERS, "batch_size": PROXY_BATCH, "context_length": PROXY_CONTEXT},
        "phase1_architecture_tournament": {"results": arch_results, "winner": arch_winner},
        "phase2_shape_search": {"results": shape_results, "winner": shape_winner_label, "winner_dims": shape_winner_dims},
        "phase3_hill_climbing": {"seed_params": seed_params, "path": hillclimb_path, "best_params": best_params, "best_val_loss": best_loss},
        "phase4_model_soup": soup_result,
        "final_recommended_config": {**shape_winner_dims, "lr": 10 ** best_params["lr_log10"], "batch_size": int(best_params["batch_size"]), "warmup_frac": best_params["warmup_frac"]},
    }
    with open(DOCS / "autoresearch_history.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote AutoResearch telemetry -> {DOCS / 'autoresearch_history.json'}")


if __name__ == "__main__":
    main()
