"""TinyGPT: a small decoder-only transformer, written so every tensor is nameable.

Deliberately plain: no fused kernels, no flash attention, no torch.compile. The
point of this repo is to interrogate a training step, and you cannot interrogate
what you cannot see. Every intermediate is given a name and a shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128   # T: max context length in tokens
    n_layer: int = 4        # L: number of transformer blocks
    n_head: int = 4         # H: attention heads per block
    n_embd: int = 128       # C: residual stream width
    dropout: float = 0.0    # kept at 0 so gradients are deterministic
    bias: bool = True

    @property
    def head_dim(self) -> int:
        """D: width of one attention head. C = H * D."""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        return self.n_embd // self.n_head


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, spelled out in explicit matmuls."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # causal mask, registered as a buffer so it moves with .to(device)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x, tracer=None, pad_mask=None):
        B, T, C = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim

        qkv = self.c_attn(x)                                    # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)                           # each (B, T, C)
        # (B, T, C) -> (B, H, T, D): split the channel axis into heads, then put
        # the head axis next to batch so each head is an independent T x D matrix.
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))  # (B, H, T, T)
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        if pad_mask is not None:
            # pad_mask: (B, T) True where the key position is a real token.
            att = att.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                                             # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)        # re-merge heads
        y = self.resid_dropout(self.c_proj(y))

        if tracer is not None:
            tracer.log("attn.qkv", qkv, "B=batch, T=positions, 3C=q|k|v packed side by side")
            tracer.log("attn.q", q, "B=batch, H=heads, T=query positions, D=head width")
            tracer.log("attn.k", k, "B=batch, H=heads, T=key positions, D=head width")
            tracer.log("attn.v", v, "B=batch, H=heads, T=value positions, D=head width")
            tracer.log("attn.scores", att, "B=batch, H=heads, T=query pos, T=key pos (row i = what token i attends to)")
            tracer.log("attn.out_heads", y, "B=batch, T=positions, C=heads re-merged back into the residual width")
        return y


class MLP(nn.Module):
    """Position-wise feed-forward: widen 4x, GELU, project back."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, tracer=None):
        h = self.c_fc(x)                                        # (B, T, 4C)
        h = F.gelu(h)
        out = self.dropout(self.c_proj(h))                      # (B, T, C)
        if tracer is not None:
            tracer.log("mlp.hidden", h, "B=batch, T=positions, 4C=widened feature space (per position, independent)")
            tracer.log("mlp.out", out, "B=batch, T=positions, C=projected back to residual width")
        return out


class Block(nn.Module):
    """Pre-norm transformer block: x = x + attn(ln(x)); x = x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, tracer=None, pad_mask=None):
        x = x + self.attn(self.ln_1(x), tracer=tracer, pad_mask=pad_mask)
        if tracer is not None:
            tracer.log("block.after_attn", x, "B=batch, T=positions, C=residual stream after the attention add")
        x = x + self.mlp(self.ln_2(x), tracer=tracer)
        if tracer is not None:
            tracer.log("block.after_mlp", x, "B=batch, T=positions, C=residual stream after the MLP add")
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying: the embedding matrix and the unembedding are one tensor
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        # scaled init on residual projections (GPT-2 recipe)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()   # wte is tied to lm_head, so it is counted once
        return n

    def forward(self, idx, targets=None, tracer=None, pad_mask=None, loss_reduction="mean"):
        """
        idx:     (B, T) int64 token ids
        targets: (B, T) int64 next-token ids, -100 where the position is ignored
        loss_reduction: "mean" -> average over unignored positions
                        "sum"  -> total over unignored positions (and also return the count)
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)

        tok_emb = self.wte(idx)                                 # (B, T, C)
        pos_emb = self.wpe(pos)                                 # (T, C)
        x = self.drop(tok_emb + pos_emb)                        # (B, T, C) by broadcast over B

        if tracer is not None:
            tracer.log("input.idx", idx, "B=batch, T=positions; each entry is a token id in [0, vocab)")
            tracer.log("embed.tok", tok_emb, "B=batch, T=positions, C=embedding width (what the token is)")
            tracer.log("embed.pos", pos_emb, "T=positions, C=embedding width (where the token is); broadcast over B")
            tracer.log("embed.sum", x, "B=batch, T=positions, C=residual stream at layer 0")

        for i, block in enumerate(self.blocks):
            sub = tracer.scoped(f"block{i}") if tracer is not None else None
            x = block(x, tracer=sub, pad_mask=pad_mask)

        x = self.ln_f(x)                                        # (B, T, C)
        logits = self.lm_head(x)                                # (B, T, V)

        if tracer is not None:
            tracer.log("final.ln", x, "B=batch, T=positions, C=residual stream after the final LayerNorm")
            tracer.log("logits", logits, "B=batch, T=positions, V=vocab; logits[b,t,:] scores the token that follows position t")

        loss = None
        n_tokens = None
        if targets is not None:
            flat_logits = logits.view(-1, logits.size(-1))      # (B*T, V)
            flat_targets = targets.reshape(-1)                  # (B*T,)
            n_tokens = int((flat_targets != -100).sum().item())
            loss = F.cross_entropy(
                flat_logits, flat_targets,
                ignore_index=-100,
                reduction="sum" if loss_reduction == "sum" else "mean",
            )
            if tracer is not None:
                tracer.log("loss.flat_logits", flat_logits, "B*T=every position flattened into one batch, V=vocab")
                tracer.log("loss.flat_targets", flat_targets, "B*T=one correct next-token id per position (-100 = ignored)")
                tracer.log("loss", loss, "scalar: cross-entropy, in nats per token")

        return logits, loss, n_tokens


# ---------------------------------------------------------------- device

def pick_device(prefer: str | None = None) -> torch.device:
    """cuda > mps > cpu. `prefer` overrides, so a cell can pin something to CPU.

    Q2 and the Q3 control stay on CPU on purpose: they assert bit-for-bit
    identities that a GPU's reduction order is free to break without anything
    being wrong.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device) -> None:
    """Block until queued work on `device` has actually finished.

    CUDA and MPS launch asynchronously: without this, `time.perf_counter()`
    around a forward pass measures how long it took to *enqueue* the work, not
    to do it. Every timing number in Q5 would be fiction.
    """
    d = torch.device(device)
    if d.type == "cuda":
        torch.cuda.synchronize()
    elif d.type == "mps":
        torch.mps.synchronize()
