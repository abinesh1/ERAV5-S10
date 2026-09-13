"""A real training loop, instrumented to report on itself.

Nothing exotic: AdamW, linear warmup then cosine decay, global-norm clipping.
The only unusual thing is that it records, at every single step, the quantities
the rest of this repo asks questions about - loss, pre-clip grad norm, step time,
tokens consumed - rather than printing a loss every N steps and throwing the
rest away.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from .model import GPTConfig, TinyGPT, pick_device, sync


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 16
    block_size: int = 128
    lr: float = 3e-3
    min_lr_frac: float = 0.1
    warmup: int = 30
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 1337
    log_every: int = 25
    device: str | None = None    # None = auto (cuda > mps > cpu)


@dataclass
class StepRecord:
    step: int
    loss: float           # loss on THIS step's random training batch
    grad_norm: float      # global L2 norm over all parameter grads, BEFORE clipping
    lr: float
    dt: float             # wall-clock seconds for the whole step
    tokens: int
    probe_loss: float = float("nan")   # loss on a FIXED batch, same weights as `loss`


@dataclass
class TrainLog:
    records: list = field(default_factory=list)

    def add(self, r: StepRecord):
        self.records.append(r)

    def col(self, name):
        return [getattr(r, name) for r in self.records]


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to min_lr_frac * lr."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if step >= cfg.steps:
        return cfg.lr * cfg.min_lr_frac
    prog = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * coeff)


def global_grad_norm(model) -> float:
    """L2 norm of the concatenation of every parameter gradient.

    Accumulated on-device and read back once. Calling float() per parameter
    would force one host sync per tensor, which on a GPU costs more than the
    norm itself and would show up as "gradnorm" time in the Q5 breakdown.
    """
    sq = [p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None]
    if not sq:
        return 0.0
    return float(torch.stack(sq).sum().sqrt())


def build(dataset, cfg: TrainConfig, model_cfg: GPTConfig | None = None):
    torch.manual_seed(cfg.seed)
    mcfg = model_cfg or GPTConfig(vocab_size=dataset.vocab_size, block_size=cfg.block_size)
    model = TinyGPT(mcfg).to(pick_device(cfg.device))
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )
    return model, opt


def train(dataset, cfg: TrainConfig, model=None, opt=None, verbose=True,
          probe=False) -> tuple[TinyGPT, TrainLog]:
    """Fixed-length training run. Returns the model and a full per-step log.

    probe=True also evaluates a FIXED held-out batch every step, at the same
    weights the training loss was computed at. That matters for the grad-norm
    study: the per-step training loss uses a different random batch each time,
    so its wobble is mostly batch difficulty, not model change. The probe holds
    the data constant so a change in it can only come from the weights.

    It costs an extra forward pass per step, so it is off by default and the MFU
    timings never see it.
    """
    if model is None or opt is None:
        model, opt = build(dataset, cfg)
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(cfg.seed)
    log = TrainLog()
    model.train()

    probe_x = probe_y = None
    if probe:
        pg = torch.Generator().manual_seed(cfg.seed + 991)
        probe_x, probe_y = dataset.fixed_batch(cfg.batch_size, cfg.block_size,
                                               split="val", generator=pg, device=dev)

    for step in range(cfg.steps):
        lr = lr_at(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr

        sync(dev)
        t0 = time.perf_counter()
        x, y = dataset.fixed_batch(cfg.batch_size, cfg.block_size, generator=gen, device=dev)
        opt.zero_grad(set_to_none=True)
        _, loss, ntok = model(x, targets=y)

        probe_val = float("nan")
        if probe_x is not None:
            with torch.no_grad():
                _, pl, _ = model(probe_x, targets=probe_y)
                probe_val = float(pl)

        loss.backward()
        gnorm = global_grad_norm(model)          # measured BEFORE clipping
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sync(dev)
        dt = time.perf_counter() - t0

        log.add(StepRecord(step, float(loss.detach()), gnorm, lr, dt, ntok, probe_val))
        if verbose and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            print(f"step {step:4d} | loss {float(loss.detach()):.4f} | "
                  f"gnorm {gnorm:7.4f} | lr {lr:.2e} | {dt*1e3:6.1f} ms")

    return model, log
