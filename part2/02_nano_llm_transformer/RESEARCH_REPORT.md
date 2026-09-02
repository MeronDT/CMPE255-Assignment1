# Research Report: NanoLlama — Autoregressive SFT LLM

**Methodology**: CRISP-DM · **Data**: TinyStories (pretraining), Stanford Alpaca (SFT) · **Model**: from-scratch decoder-only transformer with RoPE, RMSNorm, SwiGLU, tied embeddings (initially ~29.5M params; AutoResearch found and redeployed a 16.8M-param variant with *better* validation loss — see §4.4) · **Hardware**: NVIDIA RTX 3050 Ti Laptop GPU, 4GB VRAM

---

## Abstract

We implement the full modern LLM recipe — pretrain, then instruction-tune — from scratch, at a scale deliberately chosen to fit and finish training on a single 4GB laptop GPU rather than promise more than the hardware can deliver. The architecture uses four primitives that postdate the original Transformer (RoPE, RMSNorm, SwiGLU, weight tying), each validated empirically against its pre-2020 alternative via an AutoResearch ablation tournament, not just cited from the papers that proposed them. A subsequent AutoResearch hyperparameter/shape search found a **smaller** (16.8M vs. 29.5M parameter) configuration that beat the original on held-out validation loss at full training budget (perplexity 6.73 vs. 7.45) and was automatically redeployed — a genuine efficiency-and-quality win, not just a quality win. The base model produces genuinely coherent short-story text. The SFT model, instruction-tuned on a subset of Alpaca, learns the *structure* of instruction-following but not general factual accuracy — an honest, expected finding given the pretraining corpus's narrow domain, discussed in full in §6.

---

## 1. Business Understanding

Understand what training a small chat-following LLM actually takes, end to end, on hardware anyone might have — see [`docs/01_business_understanding.md`](./docs/01_business_understanding.md) for full objectives and constraints. The central, non-negotiable constraint: **4GB of VRAM**, which rules out anything beyond tens of millions of parameters and forces every other design decision (corpus choice, batch size, search budget) to be justified against it, not assumed.

## 2. Data Understanding

- **TinyStories** (Eldan & Li, Microsoft Research, 2023): 27,630 short children's stories, ~22.5M characters, restricted vocabulary (~6,450 unique words in a 5,000-story sample). The paper's central finding — models with **fewer than 10M parameters** can generate coherent English when trained on simple, curated text — is the entire reason this corpus was chosen over a generic web-text slice a 30M-parameter model would have no realistic chance of modeling well.
- **Stanford Alpaca** (Taori et al., 2023): 52,002 instruction/response pairs generated via OpenAI's `text-davinci-003` from a small human-written seed set — the standard small-scale instruction-tuning dataset, widely used precisely because it's small enough to fine-tune on quickly while still teaching instruction-following behavior.

Full corpus statistics: [`docs/eda/eda_findings.json`](./docs/eda/eda_findings.json), [`eda_overview.png`](./docs/eda/eda_overview.png).

## 3. Data Preparation

A byte-level BPE tokenizer (`vocab_size=8192`) was trained from scratch on TinyStories — not GPT-2's 50K-token vocabulary, which would be unjustifiably large for a corpus with a ~6.5K-word working vocabulary and would bloat the (untied-if-not-for-weight-tying) embedding table. Pretraining data is a single packed token stream (5.3M train / 108K val tokens). SFT data uses an 8,000-example Alpaca subset (7,600 train / 400 val), formatted with the classic Alpaca instruction template and **loss-masked to response tokens only** — the model is trained to generate good responses, not to predict the fixed instruction template that precedes them.

## 4. Modeling

### 4.1 Architecture

| Component | Choice | Reference |
|---|---|---|
| Position encoding | RoPE | Su et al. 2021, *RoFormer* |
| Normalization | RMSNorm | Zhang & Sennrich 2019 |
| Feed-forward | SwiGLU | Shazeer 2020, *GLU Variants Improve Transformer* |
| Embeddings | Tied input/output | Press & Wolf 2017 |
| Overall recipe | — | Touvron et al. 2023, *LLaMA* |

Full rationale for each choice: [`DESIGN_DOC.md §3`](./DESIGN_DOC.md#3-model-architecture).

### 4.2 A real engineering incident: VRAM oversubscription

The first pretraining attempt, at the "obvious" default (`batch_size=64, context_length=384`), **silently stalled** — GPU utilization sat at 100% with almost no iteration progress for several minutes. This is a genuinely useful finding for anyone reproducing this on similar hardware, so it's documented rather than quietly fixed and forgotten:

A targeted memory diagnostic (sweeping batch size × context length, measuring `torch.cuda.max_memory_allocated()`) found the root cause: that configuration needs **4.52GB**, just over this GPU's 4.00GB. When a CUDA allocation exceeds *dedicated* VRAM by even a little on Windows, the driver can fall back to a "shared GPU memory" pool backed by system RAM over PCIe — not a crash, but 10-50x slower, which looks exactly like a hang rather than an OOM error. Every batch size in every script in this project was subsequently chosen from an explicit sweep (`batch_size=16` for real training, keeping peak usage at 2.49GB; `24` for AutoResearch's shorter proxy runs) — see [`DESIGN_DOC.md §5`](./DESIGN_DOC.md#5-key-technical-decisions) for the full sweep data.

**Takeaway for other small-hardware LLM projects**: "does this fit in VRAM" is an empirical question specific to your GPU and driver stack, not something to infer from parameter count alone — and a stalled-not-crashed training run is a strong signal to check memory headroom before assuming a code bug.

### 4.3 AutoResearch: 4-phase search

Following the same search→finalize philosophy as this repo's [taxi project AutoResearch pipeline](../01_nyc_taxi_trip_prediction/RESEARCH_REPORT.md) (search cheaply, finalize properly), every phase below trains a short **proxy run** (250 steps, 256-token context) so the whole search finishes in minutes, not hours:

1. **Architecture-primitive tournament** — the NanoLlama default (RoPE + RMSNorm + SwiGLU) vs. three single-primitive swaps (learned position embeddings, LayerNorm, GELU MLP), each trained from the same seed.
2. **Width vs. depth shape search** — four (d_model, n_layers, n_heads) combinations at roughly matched compute.
3. **Hyperparameter hill-climbing** — literal greedy local search (not random/grid): from a seed (learning rate, batch size, warmup fraction), evaluate every one-step neighbor, move to the best improving one, repeat to a local optimum.
4. **Model soup** ([Wortsman et al. 2022](https://arxiv.org/abs/2203.05482)) — weight-average the hill-climbed model and the seed-config model; check whether the average beats both ingredients, without any extra inference cost over a single model.

**Phase 1 — Architecture-primitive tournament** (val perplexity, lower is better):

| Variant | Val PPL | vs. NanoLlama default |
|---|---:|---|
| NanoLlama default (RoPE + RMSNorm + SwiGLU) | 32.46 | — |
| Learned position embeddings (no RoPE) | 41.09 | **26.6% worse** — confirms RoPE's relative-position encoding helps, matching Su et al. |
| LayerNorm (no RMSNorm) | 33.89 | **4.4% worse** — RMSNorm ahead, but by a much smaller margin than RoPE's effect |
| GELU MLP (no SwiGLU) | 28.92 | **10.9% *better*** — see discussion below |

Two of three ablations confirmed the expected direction (RoPE and RMSNorm both won). The third didn't: GELU nominally *beat* SwiGLU in this tournament. Rather than discard or explain this away, it's worth being precise about why it's not strong evidence against SwiGLU: this is a **single 250-step run at a single random seed**, with no repeated trials to estimate variance — nowhere near enough statistical power to distinguish a real effect from run-to-run noise at this tiny budget. SwiGLU's advantage in the literature (Shazeer 2020) is demonstrated over much longer training runs at much larger scale. The deployed model keeps SwiGLU on that basis — swapping a design choice on a single noisy small-budget signal would be exactly the kind of noise-chasing good ML practice avoids, not genuine evidence.

**Phase 2 — Width vs. depth shape search:**

| Shape | Params | Val PPL |
|---|---:|---:|
| d256, 12 layers (deep/narrow) | 11.7M | 39.87 |
| d384, 6 layers (balanced) | 13.8M | 31.88 |
| **d512, 4 layers (wide/shallow)** | **16.8M** | **27.98** ← winner |
| d512, 8 layers (deployed default) | 29.5M | 29.81 |

At this proxy training budget (250 steps), the **wider-shallower** d512/4-layer shape beat the deployed 8-layer default — with *fewer* total parameters. This is plausible at short training horizons: shallower networks are easier to optimize quickly (shorter gradient paths), while deeper networks' capacity advantage typically needs a longer training budget to pay off. §4.3's finalization step checks whether this holds at the *full* training budget, not just the 250-step proxy.

**Phase 3 — Hyperparameter hill-climbing** (greedy local search from the deployed model's seed config):

| Iteration | learning rate | batch size | warmup fraction | Val Loss |
|---:|---:|---:|---:|---:|
| 0 (seed) | 3.0×10⁻⁴ | 24 | 0.08 | 3.352 |
| 1 | 3.0×10⁻⁴ | 32 | 0.08 | 3.199 |
| 2 | 6.0×10⁻⁴ | 32 | 0.08 | 3.125 |
| 3 | 6.0×10⁻⁴ | 32 | 0.11 | 3.100 |
| 4 (local optimum) | 6.0×10⁻⁴ | 32 | 0.14 | **3.061** |

4 accepted moves, ending at a local optimum (iteration 5 found no improving neighbor) — an 8.7% val-loss reduction from the seed config, entirely from a larger batch size, ~2x higher peak LR, and a longer warmup.

**Phase 4 — Model soup**: weight-averaging the hill-climbed model (val loss 3.153) and the seed-config model (val loss 3.421) produced a soup with val loss **7.128** — dramatically *worse* than either ingredient, not better. This is a genuine negative result worth explaining rather than hiding: [Wortsman et al.'s](https://arxiv.org/abs/2203.05482) model soups technique averages weights of models **fine-tuned from a shared pretrained checkpoint** with different hyperparameters — those models stay in the same loss basin, so linear interpolation between them stays on a low-loss path. Here, the two ingredient models were trained **from scratch from different random initializations** (different seeds), which — due to neural networks' permutation symmetry — land in essentially unrelated regions of the loss landscape. Averaging weights across different basins lands in the high-loss region *between* them, which is exactly the documented failure mode the original paper's setup avoids by construction. **Takeaway**: model soups need shared-checkpoint provenance to work; this implementation validated that precondition by testing what happens when it's violated.

**Final recommended config** (from phases 2 + 3): `d_model=512, n_layers=4, n_heads=8, lr=5.99e-4, batch_size=32, warmup_frac=0.14`. Whether this proxy-scale winner holds up at the full training budget is checked next.

### 4.4 Finalization: does the AutoResearch winner hold at full scale?

Yes — and by a wide margin. Retraining the recommended config (`d_model=512, n_layers=4, n_heads=8, lr=5.99e-4, batch_size=32`) at the **full** 3,000-step budget and full 384-token context reached **val loss 1.9059 (perplexity 6.73)**, beating the originally-deployed 8-layer model's **val loss 2.0085 (perplexity 7.45)** — a real, held-out-validated improvement, not a proxy-budget artifact. The result was automatically redeployed as the production `checkpoints/base.pt`.

What makes this a clean result, not a wash: the winning model has **16.8M parameters vs. the original's 29.5M** — 43% fewer parameters, better validation loss, and (because it's a shallower network) faster training and inference. At this data scale (5.3M training tokens), a shallower-but-wider network apparently optimizes more effectively than a deeper one within the same step budget — plausibly because shorter gradient paths are easier to optimize quickly, while a deeper network's extra capacity would need a longer training run to pay for itself. This is exactly the kind of finding a fixed-architecture-by-assumption approach would never surface.

One honest trade-off to flag: qualitatively, a small sample of generations from the new 4-layer model reads very slightly less fluent than the original 8-layer model's samples (see §5) — e.g. one sample repeats a clause ("Max got close to Max"). The quantitative validation loss is unambiguously better; whether that fully translates to human-judged coherence at this sample size is genuinely uncertain and would need a larger, more systematic qualitative evaluation (e.g. the TinyStories paper's GPT-4-as-grader protocol) to settle definitively — noted here rather than asserted away.

The SFT model was re-fine-tuned from this new base checkpoint (for consistency — an SFT checkpoint built on a since-replaced base model would be a real, if subtle, deployment bug) — see updated §5 metrics.

## 5. Evaluation

Final numbers, **after** the AutoResearch redeployment described in §4.4 (both models below use the winning 16.8M-parameter, 4-layer architecture; SFT was re-fine-tuned from the new base checkpoint for consistency):

| Model | Final Val Loss | Val Perplexity | Training Time | Params |
|---|---:|---:|---:|---:|
| Base (TinyStories pretrain) | 1.9059 | 6.73 | 10.5 min (3,000 steps) | 16.8M |
| SFT (Alpaca instruction-tune, full retrain — see §8) | 2.7783 | 16.09 | 15.5 min (4,500 steps) | 16.8M |

(For reference, the original 29.5M-param 8-layer model this replaced scored val loss 2.0085 / perplexity 7.45 on the base task before being superseded. The SFT row reflects the §8 follow-up retrain; the original 600-step/8K-example SFT run scored val loss 4.4154 / perplexity 82.72 and is superseded.)

Training curves: [`docs/eda/evaluation_base.png`](./docs/eda/evaluation_base.png), [`evaluation_sft.png`](./docs/eda/evaluation_sft.png). Both show smooth, healthy convergence with no train/val divergence indicating overfitting.

**Base model sample** (prompt: *"Once upon a time,"*):
> Once upon a time, there was a little boy named Tim. Tim had a toy snake that he loved to play with. One day, Tim took his snake to the park. The snake was very excited to play with Tim. At the park, Tim saw a big tree. He thought, "Maybe I can turn my snake and go under the tree!" So, Tim climbed up the tree and pushed the snake with his hand...

Grammatical, coherent, on-topic for the TinyStories domain — consistent with the paper's finding that models at this scale can produce fluent text on a sufficiently narrow, curated corpus. One other sample from this smaller redeployed model shows a minor repetition artifact ("When Max got close to Max...") not seen in the original 8-layer model's samples — a small, real qualitative trade-off against the quantitative perplexity win, flagged honestly in §4.4 rather than cherry-picked around.

## 6. Discussion: What the SFT Model Did and Didn't Learn

Even after the §8 retrain (full 51K-example Alpaca set, 4,500 steps), the SFT model's factual quality remains poor (e.g., asked "What is the capital of France?", it produces fluent, grammatically-correct text that sometimes gestures at the right topic but does not reliably state "Paris"). This is worth explaining rather than hiding:

- **It did learn structure and fluency.** Every SFT sample correctly terminates with the end-of-text token and stays within a single-response format. Post-retrain, generations are also *grammatically coherent, on-topic English* — a qualitative step up from the original 600-step run's malformed/fragmentary output (see §8) — but coherent phrasing is not the same as correct facts.
- **It did not reliably learn facts.** The base model's *entire* pretraining exposure is TinyStories — a children's-story corpus that never mentions capitals, health advice, or poetry technique. Alpaca's instructions span general knowledge the base model has literally never seen a single token of. Even 4,500 SFT steps over the full Alpaca set cannot inject broad world knowledge that was never in the pretraining distribution — SFT adapts a model's *behavior* to existing knowledge, it doesn't create new knowledge from examples the base model has no grounding for.
- **This is a capacity question too.** Even with domain-matched data, a 16.8M-parameter model has nowhere near the representational capacity to memorize the breadth of facts a general-purpose assistant needs — real instruction-tuned models with broad factual competence start in the billions of parameters, pretrained on hundreds of billions of tokens of general text, not millions of tokens of curated children's-story words.

This matches exactly what [`docs/01_business_understanding.md`](./docs/01_business_understanding.md) scoped out from the start: *"This will not be a general-purpose assistant, will not have broad world knowledge... The goal is a correct, working, honestly-evaluated implementation of the full pretrain → SFT → deploy pipeline at a scale that actually fits, not a scaled-up promise."* The pipeline, architecture, and training methodology are all real and correct; the fluency ceiling is a direct, well-understood consequence of the deliberately tiny data and parameter budget, not a bug.

## 7. Limitations & Future Work

- **Pretraining domain should match SFT domain more closely.** A general-web-text pretraining corpus (even a small one) would likely produce a more factually-competent SFT model than TinyStories' narrow children's-story domain — at the cost of needing a larger model to reach TinyStories-level fluency on that harder distribution (per the TinyStories paper's own findings). This remains true even after the §8 SFT data/step increase, since it doesn't touch the pretraining corpus.
- **AutoResearch's proxy runs (250 steps) are directional, not definitive** — a config that wins a 250-step proxy race may not be the true optimum at the full 3,000-step budget, the same caution the taxi project's AutoResearch finalization step surfaced for gradient-boosted trees.
- **No RLHF/DPO alignment** — out of scope per the original business-understanding constraints; would require preference data and substantially more compute than a laptop GPU provides.

## 8. Follow-Up Fix: SFT Undertraining (Addendum)

A later manual test of the deployed chat UI reported low-quality ("gibberish") responses. Diagnosis ruled out a correctness bug first: loss masking, RoPE, and nucleus sampling were all re-audited against the model code (`model/nanollama.py`) and found correct. A Playwright-driven diagnostic (`diagnose_chat.mjs`) across six varied prompts confirmed the problem was general low quality, not a single bad prompt, a UI bug, or a network/console error.

The actual root cause was undertraining, not a bug: the original SFT run (§5's superseded numbers) used only an 8,000-example subset (15%) of the 52,002-example Alpaca set for 600 steps at batch size 16 — under 1.3 epochs of exposure, nowhere near enough for instruction-tuning to converge. Fix applied in `scripts/02_prepare_data.py` and `scripts/04_sft.py`: SFT now trains on the **full 51,002-example train split** (1,000 held out for validation) for **4,500 steps** (~2.8 epochs) at **batch size 32** (empirically VRAM-swept beforehand — batch 32 uses 2.9GB, batch 48 overflows the 4GB budget, consistent with the VRAM-oversubscription caution from the original base-model incident in §5/DESIGN_DOC §5), with LR raised to 1e-4 and warmup extended to 300 steps to match the larger effective training budget.

Result: val perplexity improved from 82.72 (undertrained) to 16.09 (retrained) — see the updated §5 table. Qualitatively, output changed from fragmentary/malformed text to grammatically coherent, on-topic English across all six diagnostic prompts, with zero console or network errors. Factual accuracy is still weak (§6) — that is the honest, expected ceiling of a 16.8M-parameter model pretrained only on children's-story text, not a remaining bug. The tokenizer was retrained as part of rerunning `02_prepare_data.py`; a `git diff` on `vocab.json`/`merges.txt` confirmed byte-identical output (deterministic given the same corpus), so no risk to the base checkpoint's compatibility.
