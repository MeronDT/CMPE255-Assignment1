# CRISP-DM Phase 1: Business Understanding

## Business Objectives

Understand, from first principles and by building one, what it actually takes to train a small autoregressive language model into a chat-following assistant — the same two-stage recipe (pretrain, then instruction-tune) behind every modern LLM product, just scaled down to run entirely on a single laptop GPU instead of a data-center cluster.

## Data Mining Goals

1. **Pretraining**: train a decoder-only transformer ("NanoLlama") from scratch to model English text, using architecture primitives from current research rather than the original 2017 Transformer defaults:
   - **RoPE** (rotary position embeddings — Su et al. 2021) instead of learned/absolute positional embeddings.
   - **RMSNorm** (Zhang & Sennrich 2019) instead of LayerNorm.
   - **SwiGLU** feed-forward blocks (Shazeer 2020) instead of ReLU/GELU MLPs.
   - **Weight-tied** input/output embeddings (Press & Wolf 2017).

   This is the same primitive set LLaMA (Touvron et al. 2023) popularized — hence "NanoLlama."
2. **Supervised Fine-Tuning (SFT)**: continue training the pretrained model on instruction/response pairs so it follows a chat-style prompt format, turning a raw text-completion model into something that behaves like an assistant.
3. **AutoResearch**: rather than asserting these architecture choices are good, empirically test them — an ablation tournament (RoPE vs. learned positions, SwiGLU vs. GELU, RMSNorm vs. LayerNorm), a hyperparameter hill-climbing search, and weight-averaging ("model soup") of top checkpoints — all logged, all reproducible, all comparable against the papers that proposed each technique.

## Success Criteria

- **Statistical**: falling training/validation loss and held-out perplexity for both the base and SFT model; ablation results directionally consistent with what each cited paper reports (e.g., RoPE/SwiGLU should beat the older defaults, even if the margin is smaller at this tiny scale than at GPT-3 scale).
- **Qualitative**: the base model should produce locally-coherent, grammatical English continuations (the TinyStories bar — see Constraints); the SFT model should visibly follow instruction-format prompts rather than just continuing them as free text.
- **Deployable**: a live chat API + UI, not just a notebook — plus a dashboard exposing every metric an ML/AI engineer would actually want before trusting a model (loss curves, param count, tokens/sec, GPU memory, sample generations, the full AutoResearch search history).

## Constraints

- **Hardware**: a single laptop GPU — NVIDIA RTX 3050 Ti Laptop, **4GB VRAM**. This bounds model size (tens of millions, not billions, of parameters) and batch size, and rules out anything resembling a "real" chat assistant's fluency.
- **Data scale matches model scale, deliberately.** [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, Microsoft Research, 2023) showed that models with as few as 1–33M parameters can generate fluent, coherent English when trained on a *curated, simple* corpus (short children's stories using a restricted vocabulary) — rather than on generic web text, which requires much larger models to produce coherent output. We follow that finding directly: pretrain on TinyStories, not on a slice of a huge generic corpus that a 30M-parameter model has no realistic chance of fitting well.
- **SFT data**: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (Taori et al. 2023), 52,002 instruction/response pairs — the standard, widely-cited small-scale instruction-tuning dataset.
- **Time budget**: every training run (base pretrain, SFT, and every AutoResearch trial) is sized to finish in minutes, not hours, on this GPU — see [`DESIGN_DOC.md §5`](../DESIGN_DOC.md#5-key-technical-decisions) for the specific step/token budgets and why they're sufficient to demonstrate the method honestly.

## Explicitly Out of Scope

This will not be a general-purpose assistant, will not have broad world knowledge, and will not be safety-tuned (RLHF/DPO) — those require both far more data and far more compute than a laptop GPU can provide. The goal is a correct, working, honestly-evaluated implementation of the full pretrain → SFT → deploy pipeline at a scale that actually fits, not a scaled-up promise.
