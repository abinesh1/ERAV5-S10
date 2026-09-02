"""Q2 - verify a gradient by hand.

backward() reports dL/dw for every weight w. That claim is checkable without
trusting autograd at all: move w by a hair, watch the loss, divide.

    central difference:   dL/dw  ~=  [ L(w+h) - L(w-h) ] / 2h

The central form is used rather than the forward form because its truncation
error is O(h^2) rather than O(h) - it buys several extra digits for free.

Two things have to be right for the digits to show up:

  1. float64. In float32 the loss carries ~7 significant digits, and
     L(w+h) - L(w-h) cancels most of them away. There is no point asking for
     "several decimals" of agreement from a float32 loss.
  2. A deterministic loss. Dropout is 0 and the batch is fixed, so L is a
     genuine function of w alone.

The h-sweep at the bottom is the honest part: it shows the classic error valley,
truncation error falling as h^2 from the right, round-off error rising as 1/h
from the left, and a best h in between. That valley is the evidence that the
agreement is real and not a coincidence of one lucky step size.
"""

from __future__ import annotations

import math

import torch

from .model import GPTConfig, TinyGPT


def _loss_at(model, x, y, param, flat_index, value):
    """Set one scalar of one parameter to `value`, return the loss. No grads."""
    with torch.no_grad():
        flat = param.view(-1)
        old = flat[flat_index].item()
        flat[flat_index] = value
    with torch.no_grad():
        _, loss, _ = model(x, targets=y)
    with torch.no_grad():
        flat[flat_index] = old
    return float(loss)


def check_one_weight(dataset, param_name="blocks.0.mlp.c_fc.weight", flat_index=None,
                     batch_size=4, seq_len=16, seed=0, h=1e-6, verbose=True):
    """Full hand-check of a single weight. Returns a dict of the numbers."""
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=dataset.vocab_size, block_size=64,
                    n_layer=2, n_head=4, n_embd=64, dropout=0.0)
    model = TinyGPT(cfg).double()          # float64: this is the whole ballgame
    model.eval()                            # no dropout, deterministic loss
    gen = torch.Generator().manual_seed(seed)
    x, y = dataset.fixed_batch(batch_size, seq_len, generator=gen)

    params = dict(model.named_parameters())
    assert param_name in params, f"{param_name} not found; have {list(params)[:5]}..."
    p = params[param_name]

    # --- what backward() says -------------------------------------------------
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(x, targets=y)
    loss.backward()
    g = p.grad.view(-1)

    if flat_index is None:
        # pick the weight with the largest |grad| so the relative error is
        # meaningful - checking a weight whose gradient is ~0 proves nothing
        flat_index = int(g.abs().argmax())

    analytic = float(g[flat_index])
    w0 = float(p.detach().view(-1)[flat_index])
    base_loss = float(loss.detach())

    # --- what the loss surface says ------------------------------------------
    Lp = _loss_at(model, x, y, p, flat_index, w0 + h)
    Lm = _loss_at(model, x, y, p, flat_index, w0 - h)
    numeric = (Lp - Lm) / (2 * h)
    forward_only = (Lp - base_loss) / h

    abs_err = abs(numeric - analytic)
    rel_err = abs_err / max(abs(analytic), 1e-30)
    digits = -math.log10(rel_err) if rel_err > 0 else float("inf")

    row, col = divmod(flat_index, p.shape[-1]) if p.dim() == 2 else (0, flat_index)

    result = dict(
        param_name=param_name, flat_index=flat_index, row=row, col=col,
        shape=tuple(p.shape), w0=w0, h=h, base_loss=base_loss,
        loss_plus=Lp, loss_minus=Lm, delta_loss=Lp - Lm,
        analytic=analytic, numeric=numeric, forward_only=forward_only,
        abs_err=abs_err, rel_err=rel_err, matching_digits=digits,
        dtype=str(p.dtype),
    )

    if verbose:
        print(f"weight under test : {param_name}[{row}, {col}]   (shape {tuple(p.shape)}, {p.dtype})")
        print(f"  w                          = {w0:+.17f}")
        print(f"  h (nudge)                  = {h:.1e}")
        print(f"  L(w)                       = {base_loss:.17f}")
        print(f"  L(w+h)                     = {Lp:.17f}")
        print(f"  L(w-h)                     = {Lm:.17f}")
        print(f"  L(w+h) - L(w-h)            = {Lp-Lm:+.3e}")
        print()
        print(f"  numeric  [L(w+h)-L(w-h)]/2h = {numeric:+.17f}")
        print(f"  analytic backward() says    = {analytic:+.17f}")
        print(f"  absolute difference         = {abs_err:.3e}")
        print(f"  relative difference         = {rel_err:.3e}")
        print(f"  => they agree to {digits:.1f} decimal digits")
        print()
        print(f"  (forward difference only, for contrast: {forward_only:+.12f}, "
              f"rel err {abs(forward_only-analytic)/abs(analytic):.3e})")
    return result


def sweep_h(dataset, param_name="blocks.0.mlp.c_fc.weight", flat_index=None,
            hs=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12),
            batch_size=4, seq_len=16, seed=0, verbose=True):
    """Sweep the step size to expose the truncation / round-off valley."""
    rows = []
    idx = flat_index
    for h in hs:
        r = check_one_weight(dataset, param_name, idx, batch_size, seq_len, seed,
                             h=h, verbose=False)
        idx = r["flat_index"]           # pin the same weight across the sweep
        rows.append(r)
    if verbose:
        print(f"{'h':>10}  {'numeric':>22}  {'rel error':>11}  {'digits':>6}  regime")
        print(f"{'-'*10}  {'-'*22}  {'-'*11}  {'-'*6}  {'-'*26}")
        best = min(rows, key=lambda r: r["rel_err"])
        for r in rows:
            if r["h"] > best["h"]:
                regime = "truncation error dominates"
            elif r["h"] < best["h"]:
                regime = "round-off error dominates"
            else:
                regime = "<-- best"
            print(f"{r['h']:10.0e}  {r['numeric']:+22.15f}  {r['rel_err']:11.2e}  "
                  f"{r['matching_digits']:6.1f}  {regime}")
    return rows


def autograd_gradcheck(dataset, seed=0):
    """Independent corroboration: torch's own gradcheck on the whole loss.

    This differentiates the loss w.r.t. every parameter numerically, not just the
    one we checked by hand. It is the same idea, done exhaustively.
    """
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=dataset.vocab_size, block_size=16,
                    n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = TinyGPT(cfg).double()
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    x, y = dataset.fixed_batch(2, 8, generator=gen)

    names = [n for n, _ in model.named_parameters()]
    params = [p.detach().clone().requires_grad_(True) for _, p in model.named_parameters()]

    def f(*ps):
        return torch.func.functional_call(model, dict(zip(names, ps)), (x,), {"targets": y})[1]

    ok = torch.autograd.gradcheck(f, tuple(params), eps=1e-6, atol=1e-7, rtol=1e-4)
    return bool(ok), len(names)
