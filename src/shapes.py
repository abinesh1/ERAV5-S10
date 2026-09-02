"""Q1 - print every tensor shape in one training step, and say what each axis means.

A step is not just the forward pass. It is forward, loss, backward, and the
optimizer update, and every one of those stages has tensors with shapes worth
knowing. This module walks all four.

Axis letters used throughout:
    B  batch          rows processed together
    T  time/position  token slots along the sequence
    C  channels       residual stream width (n_embd)
    H  heads          attention heads
    D  head width     C / H
    V  vocab          number of distinct tokens
    L  layers         transformer blocks
"""

from __future__ import annotations

import torch

from .model import GPTConfig, TinyGPT


class ShapeTracer:
    """Collects (name, shape, dtype, meaning) as the forward pass runs."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.rows: list[tuple[str, str, str, str]] = []
        self._root = self

    def scoped(self, sub: str) -> "ShapeTracer":
        child = ShapeTracer(prefix=f"{self.prefix}{sub}.")
        child.rows = self.rows          # share the same sink
        child._root = self._root
        return child

    def log(self, name: str, t: torch.Tensor, meaning: str):
        shape = "scalar" if t.dim() == 0 else "(" + ", ".join(str(s) for s in t.shape) + ")"
        self.rows.append((f"{self.prefix}{name}", shape, str(t.dtype).replace("torch.", ""), meaning))

    def table(self, title: str) -> str:
        if not self.rows:
            return ""
        w0 = max(len(r[0]) for r in self.rows)
        w1 = max(len(r[1]) for r in self.rows)
        w2 = max(len(r[2]) for r in self.rows)
        out = [title, "=" * len(title), ""]
        out.append(f"{'tensor'.ljust(w0)}  {'shape'.ljust(w1)}  {'dtype'.ljust(w2)}  what the dimensions mean")
        out.append(f"{'-'*w0}  {'-'*w1}  {'-'*w2}  {'-'*56}")
        for n, s, d, m in self.rows:
            out.append(f"{n.ljust(w0)}  {s.ljust(w1)}  {d.ljust(w2)}  {m}")
        return "\n".join(out)


def _fmt(shape) -> str:
    return "(" + ", ".join(str(s) for s in shape) + ")" if len(shape) else "scalar"


def parameter_table(model: TinyGPT, only_first_block: bool = True) -> str:
    """Every learnable tensor, its shape, and what its axes index."""
    cfg = model.cfg
    meanings = {
        "wte.weight":  f"V={cfg.vocab_size} rows (one embedding per token), C={cfg.n_embd} channels. TIED to lm_head.weight",
        "wpe.weight":  f"T={cfg.block_size} rows (one embedding per position), C={cfg.n_embd} channels",
        "ln_f.weight": f"C={cfg.n_embd} per-channel gain of the final LayerNorm",
        "ln_f.bias":   f"C={cfg.n_embd} per-channel shift of the final LayerNorm",
        "lm_head.weight": f"V={cfg.vocab_size} out, C={cfg.n_embd} in. Same storage as wte.weight",
    }
    per_block = {
        "ln_1.weight": f"C={cfg.n_embd} gain, pre-attention LayerNorm",
        "ln_1.bias":   f"C={cfg.n_embd} shift, pre-attention LayerNorm",
        "attn.c_attn.weight": f"3C={3*cfg.n_embd} out (q|k|v stacked), C={cfg.n_embd} in - one matmul makes all three",
        "attn.c_attn.bias":   f"3C={3*cfg.n_embd} biases, one per q/k/v channel",
        "attn.c_proj.weight": f"C={cfg.n_embd} out, C={cfg.n_embd} in - mixes the heads back together",
        "attn.c_proj.bias":   f"C={cfg.n_embd} output biases",
        "ln_2.weight": f"C={cfg.n_embd} gain, pre-MLP LayerNorm",
        "ln_2.bias":   f"C={cfg.n_embd} shift, pre-MLP LayerNorm",
        "mlp.c_fc.weight":   f"4C={4*cfg.n_embd} out, C={cfg.n_embd} in - widen",
        "mlp.c_fc.bias":     f"4C={4*cfg.n_embd} biases",
        "mlp.c_proj.weight": f"C={cfg.n_embd} out, 4C={4*cfg.n_embd} in - narrow back",
        "mlp.c_proj.bias":   f"C={cfg.n_embd} biases",
    }
    rows = []
    for name, p in model.named_parameters():
        if name.startswith("blocks."):
            idx, rest = name.split(".", 2)[1], name.split(".", 2)[2]
            if only_first_block and idx != "0":
                continue
            meaning = per_block.get(rest, "")
            if only_first_block:
                meaning += f"   [x{cfg.n_layer} blocks]"
        else:
            meaning = meanings.get(name, "")
        rows.append((name, _fmt(p.shape), f"{p.numel():,}", meaning))

    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    w2 = max(len(r[2]) for r in rows)
    head = "PARAMETERS (learnable tensors)"
    out = [head, "=" * len(head), ""]
    if only_first_block:
        out.append(f"blocks.1..{cfg.n_layer-1} are identical to blocks.0 and are omitted.\n")
    out.append(f"{'parameter'.ljust(w0)}  {'shape'.ljust(w1)}  {'count'.rjust(w2)}  meaning of each axis")
    out.append(f"{'-'*w0}  {'-'*w1}  {'-'*w2}  {'-'*56}")
    for n, s, c, m in rows:
        out.append(f"{n.ljust(w0)}  {s.ljust(w1)}  {c.rjust(w2)}  {m}")
    out.append("")
    out.append(f"total parameters: {model.num_params():,}")
    out.append(f"  (wte.weight and lm_head.weight are the SAME tensor - counted once)")
    return "\n".join(out)


def gradient_table(model: TinyGPT, only_first_block: bool = True) -> str:
    """After backward(), every parameter has a grad of exactly its own shape."""
    rows = []
    mismatch = 0
    for name, p in model.named_parameters():
        if only_first_block and name.startswith("blocks.") and name.split(".")[1] != "0":
            continue
        g = p.grad
        if g is None:
            rows.append((name, "None", "-", "no gradient (unused in this step)"))
            continue
        same = tuple(g.shape) == tuple(p.shape)
        mismatch += (not same)
        rows.append((name, _fmt(g.shape), str(g.dtype).replace("torch.", ""),
                     "same shape as the parameter" if same else "!! SHAPE MISMATCH !!"))
    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    head = "GRADIENTS (after loss.backward())"
    out = [head, "=" * len(head), "",
           "dL/dW has the shape of W, always: one partial derivative per scalar knob.",
           ""]
    out.append(f"{'parameter.grad'.ljust(w0)}  {'shape'.ljust(w1)}  note")
    out.append(f"{'-'*w0}  {'-'*w1}  {'-'*40}")
    for n, s, d, m in rows:
        out.append(f"{n.ljust(w0)}  {s.ljust(w1)}  {m}")
    out.append("")
    out.append(f"shape mismatches: {mismatch}  (must be 0)")
    return "\n".join(out)


def optimizer_state_table(model: TinyGPT, opt) -> str:
    """AdamW carries two extra tensors per parameter, each the parameter's shape."""
    rows = []
    for name, p in model.named_parameters():
        st = opt.state.get(p, {})
        if not st:
            continue
        for key, meaning in (("exp_avg", "1st moment m: EMA of the gradient, elementwise"),
                             ("exp_avg_sq", "2nd moment v: EMA of the gradient SQUARED, elementwise")):
            if key in st:
                rows.append((f"{name}.{key}", _fmt(st[key].shape), meaning))
        if name.startswith("blocks.") and name.split(".")[1] != "0":
            continue
    shown = [r for r in rows if not (r[0].startswith("blocks.") and r[0].split(".")[1] != "0")]
    w0 = max(len(r[0]) for r in shown)
    w1 = max(len(r[1]) for r in shown)
    head = "OPTIMIZER STATE (AdamW, after opt.step())"
    out = [head, "=" * len(head), "",
           "AdamW stores m and v per parameter, both shaped like the parameter.",
           "That is why Adam costs 3x the memory of the weights alone (W + m + v).",
           ""]
    out.append(f"{'state tensor'.ljust(w0)}  {'shape'.ljust(w1)}  meaning")
    out.append(f"{'-'*w0}  {'-'*w1}  {'-'*52}")
    for n, s, m in shown:
        out.append(f"{n.ljust(w0)}  {s.ljust(w1)}  {m}")
    total_state = sum(st[k].numel() for p, st in opt.state.items()
                      for k in ("exp_avg", "exp_avg_sq") if k in st)
    out.append("")
    out.append(f"optimizer state elements (all blocks): {total_state:,} "
               f"= 2 x {model.num_params():,} parameters")
    return "\n".join(out)


def trace_one_step(dataset, cfg: GPTConfig | None = None, batch_size: int = 4,
                   seq_len: int = 16, seed: int = 0) -> str:
    """Run exactly one full training step with tracing on, return the whole report."""
    torch.manual_seed(seed)
    cfg = cfg or GPTConfig(vocab_size=dataset.vocab_size, block_size=128,
                           n_layer=4, n_head=4, n_embd=128)
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    x, y = dataset.fixed_batch(batch_size, seq_len, generator=gen)

    B, T = x.shape
    header = [
        "ONE TRAINING STEP, EVERY TENSOR",
        "=" * 31,
        "",
        f"B = {B}   batch: independent sequences processed together",
        f"T = {T}   time/position: token slots within each sequence",
        f"C = {cfg.n_embd}  channels: width of the residual stream (n_embd)",
        f"H = {cfg.n_head}   heads: attention heads per block",
        f"D = {cfg.head_dim}  head width: C / H",
        f"V = {cfg.vocab_size}  vocab: distinct characters in the corpus",
        f"L = {cfg.n_layer}   layers: transformer blocks",
        "",
    ]

    tracer = ShapeTracer()
    opt.zero_grad(set_to_none=True)
    _, loss, _ = model(x, targets=y, tracer=tracer)
    fwd = tracer.table("FORWARD PASS (activations)")

    loss.backward()
    grads = gradient_table(model)
    opt.step()
    state = optimizer_state_table(model, opt)
    params = parameter_table(model, only_first_block=True)

    note = (
        "\nNote on block0 vs block1..3: the tracer records every block, but blocks\n"
        "1-3 repeat block0's shapes exactly, so only block0 is shown above in the\n"
        "parameter and gradient tables. The forward table below lists all of them.\n"
    )
    return "\n\n".join(["\n".join(header), params, note, fwd, grads, state])
