"""Loads the SFT checkpoint and serves chat completions. The model was
fine-tuned on single-turn Alpaca instruction/response pairs (no multi-turn
conversation data), so — honestly, not silently — only the most recent user
message is used as the instruction; earlier turns are shown in the UI for
context but are not fed back into the model.
"""

import time
from functools import lru_cache
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.nanollama import NanoLlama, NanoLlamaConfig

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
TOKENIZER_DIR = ROOT / "data" / "tokenizer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SFT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


class ChatService:
    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer(str(TOKENIZER_DIR / "vocab.json"), str(TOKENIZER_DIR / "merges.txt"))
        self.eot_id = self.tokenizer.token_to_id("<|endoftext|>")

        ckpt = torch.load(CHECKPOINTS / "sft.pt", map_location=DEVICE, weights_only=False)
        self.cfg = NanoLlamaConfig(**ckpt["config"])
        self.model = NanoLlama(self.cfg).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def chat(self, last_user_message: str, temperature: float, top_p: float, max_new_tokens: int) -> dict:
        prompt = SFT_TEMPLATE.format(instruction=last_user_message)
        ids = torch.tensor([self.tokenizer.encode(prompt).ids], device=DEVICE)
        prompt_len = ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, eot_id=self.eot_id)
        elapsed = time.time() - t0

        new_tokens = out[0, prompt_len:].tolist()
        if new_tokens and new_tokens[-1] == self.eot_id:
            new_tokens = new_tokens[:-1]
        reply = self.tokenizer.decode(new_tokens).strip()
        n_generated = len(new_tokens)

        return {
            "reply": reply,
            "generation_ms": elapsed * 1000,
            "tokens_generated": n_generated,
            "tokens_per_sec": n_generated / elapsed if elapsed > 0 else 0,
        }


@lru_cache(maxsize=1)
def get_service() -> "ChatService":
    return ChatService()
