"""NanoLlama: a small decoder-only transformer using current-generation
primitives (RoPE, RMSNorm, SwiGLU, tied embeddings) instead of the original
2017 Transformer defaults. Architecture choices are validated empirically
against ablation alternatives in scripts/06_autoresearch.py, not just
asserted — see RESEARCH_REPORT.md.

References:
  RoPE:     Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  RMSNorm:  Zhang & Sennrich 2019, "Root Mean Square Layer Normalization"
  SwiGLU:   Shazeer 2020, "GLU Variants Improve Transformer"
  Tied embeddings: Press & Wolf 2017, "Using the Output Embedding to Improve Language Models"
  Overall recipe: Touvron et al. 2023, "LLaMA: Open and Efficient Foundation Language Models"
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NanoLlamaConfig:
    vocab_size: int = 8192
    context_length: int = 384
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_type: str = "rmsnorm"  # "rmsnorm" | "layernorm" — ablated in autoresearch
    mlp_type: str = "swiglu"  # "swiglu" | "gelu" — ablated in autoresearch
    pos_type: str = "rope"  # "rope" | "learned" — ablated in autoresearch
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mlp_hidden_dim(self) -> int:
        # SwiGLU convention (LLaMA): keep param count close to a 4x-GELU MLP
        # despite having 3 weight matrices instead of 2.
        raw = int(2 * 4 * self.d_model / 3)
        multiple_of = 32
        return multiple_of * ((raw + multiple_of - 1) // multiple_of)

    def n_params(self, non_embedding: bool = False) -> int:
        emb = self.vocab_size * self.d_model
        attn = 4 * self.d_model * self.d_model
        mlp = 3 * self.d_model * self.mlp_hidden_dim if self.mlp_type == "swiglu" else 2 * self.d_model * self.mlp_hidden_dim
        per_layer = attn + mlp
        total = self.n_layers * per_layer + (0 if (self.tie_embeddings and non_embedding) else emb)
        return total


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim: int, context_length: int, theta: float, device):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(context_length, device=device).float()
    freqs = torch.outer(t, freqs)  # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim)
    T = x.shape[-2]
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.pos_type = cfg.pos_type
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pos_type == "rope":
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        h = cfg.mlp_hidden_dim
        self.gate = nn.Linear(cfg.d_model, h, bias=False)
        self.up = nn.Linear(cfg.d_model, h, bias=False)
        self.down = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GELUMLP(nn.Module):
    """Ablation baseline: the pre-SwiGLU standard MLP."""

    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        h = cfg.mlp_hidden_dim
        self.fc1 = nn.Linear(cfg.d_model, h, bias=False)
        self.fc2 = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def make_norm(cfg: NanoLlamaConfig, dim: int):
    return RMSNorm(dim) if cfg.norm_type == "rmsnorm" else nn.LayerNorm(dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        self.norm1 = make_norm(cfg, cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg, cfg.d_model)
        self.mlp = SwiGLU(cfg) if cfg.mlp_type == "swiglu" else GELUMLP(cfg)

    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class NanoLlama(nn.Module):
    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model) if cfg.pos_type == "learned" else None
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = make_norm(cfg, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        self._rope_cache = None
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, device):
        if self._rope_cache is None or self._rope_cache[0].device != device:
            self._rope_cache = precompute_rope(self.cfg.head_dim, self.cfg.context_length, self.cfg.rope_theta, device)
        return self._rope_cache

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, loss_mask: torch.Tensor = None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))

        rope_cos, rope_sin = (self._rope(idx.device) if self.cfg.pos_type == "rope" else (None, None))
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)
        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            if loss_mask is not None:
                per_token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
                mask = loss_mask.view(-1).float()
                loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_p: float = 0.95, eot_id: int = None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumulative > top_p).float().argmax(dim=-1, keepdim=True)
            mask = torch.arange(sorted_probs.size(-1), device=idx.device).unsqueeze(0) > cutoff
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            next_sorted = torch.multinomial(sorted_probs, 1)
            next_id = sorted_idx.gather(-1, next_sorted)
            idx = torch.cat([idx, next_id], dim=1)

            if eot_id is not None and next_id.item() == eot_id:
                break
        return idx
