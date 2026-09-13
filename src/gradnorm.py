"""Q4 - log the grad norm every step, and find a step where it moved first.

Why it should lead at all
-------------------------
The loss at step t is a property of the weights BEFORE step t's update. The grad
norm at step t is a property of the slope at those same weights, and it is what
sizes the update that produces the weights measured at step t+1. So the causal
chain is:

    grad norm at t  ->  size of update at t  ->  loss at t+1

The gradient is a derivative and the loss is a level. A derivative moves first
almost by construction. This module tries to catch that happening in a real run.

Honesty constraints
-------------------
1. Every baseline here is CAUSAL - an EMA of strictly past values. Using a
   centred window would let the future leak into "was this a surprise", which
   would manufacture the very lead the exercise is asking about.
2. The loss is measured on a different random batch each step, so it is noisy.
   A single anecdote could be luck. The lagged cross-correlation at the bottom
   is the check on that: if the grad norm really leads, correlation must peak at
   a POSITIVE lag, and that is a statement about the whole run, not one step.
"""

from __future__ import annotations

import math


def causal_z(series, alpha=0.15, warmup=10):
    """z-score of each point against an EMA mean/std of everything BEFORE it.

    Returns z[t] = (x[t] - mean_{<t}) / std_{<t}, with z[t]=0 during warmup.
    Nothing at index >= t is ever consulted.
    """
    z = [0.0] * len(series)
    mean = series[0]
    var = 0.0
    for t, x in enumerate(series):
        if t >= warmup:
            std = math.sqrt(max(var, 1e-12))
            z[t] = (x - mean) / max(std, 1e-9)
        d = x - mean
        mean += alpha * d
        var = (1 - alpha) * (var + alpha * d * d)
    return z


def find_lead_steps(log, g_thresh=2.0, quiet=1.0, react=1.5, horizon=3,
                    alpha=0.15, warmup=10, loss_key="loss"):
    """Steps where the grad norm jumped, the loss did NOT, and then the loss did.

    A step t qualifies when all three hold:
      (a) |z_gradnorm[t]| >= g_thresh      the gradient was surprising at t
      (b) |z_loss[t]|     <= quiet         the loss was NOT surprising at t
      (c) max |z_loss[t+k]| >= react       the loss became surprising within
          for k in 1..horizon               the next few steps
    """
    loss = log.col(loss_key)
    gn = log.col("grad_norm")
    zl = causal_z(loss, alpha, warmup)
    zg = causal_z(gn, alpha, warmup)

    hits = []
    for t in range(warmup, len(loss) - horizon):
        if abs(zg[t]) < g_thresh:
            continue
        if abs(zl[t]) > quiet:
            continue
        future = [abs(zl[t + k]) for k in range(1, horizon + 1)]
        if max(future) < react:
            continue
        k_star = int(max(range(1, horizon + 1), key=lambda k: abs(zl[t + k])))
        hits.append(dict(
            step=t, z_grad=zg[t], z_loss=zl[t],
            grad_norm=gn[t], loss=loss[t],
            reacts_at=t + k_star, lag=k_star,
            z_loss_react=zl[t + k_star], loss_react=loss[t + k_star],
            delta_gn=gn[t] - gn[t - 1], delta_loss=loss[t] - loss[t - 1],
            delta_loss_react=loss[t + k_star] - loss[t + k_star - 1],
            _zl=zl, _zg=zg,
        ))
    return hits, zl, zg


def lag_correlation(log, max_lag=6, alpha=0.15, warmup=10, loss_key="loss"):
    """corr( |z_gradnorm[t]| , |z_loss[t+lag]| ) for a range of lags.

    Negative lag asks whether the loss leads the grad norm. Positive lag asks
    the reverse. Where the peak sits is the answer, and it is computed over the
    whole run rather than cherry-picked.
    """
    loss, gn = log.col(loss_key), log.col("grad_norm")
    zl = [abs(v) for v in causal_z(loss, alpha, warmup)]
    zg = [abs(v) for v in causal_z(gn, alpha, warmup)]
    n = len(zl)

    def corr(a, b):
        if len(a) < 3:
            return float("nan")
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a)
        vb = sum((x - mb) ** 2 for x in b)
        if va <= 0 or vb <= 0:
            return float("nan")
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return cov / math.sqrt(va * vb)

    out = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = zg[warmup:n - lag] if lag else zg[warmup:]
            b = zl[warmup + lag:]
        else:
            a = zg[warmup - lag:]
            b = zl[warmup:n + lag]
        m = min(len(a), len(b))
        out.append(dict(lag=lag, corr=corr(a[:m], b[:m]), n=m))
    return out


def describe_hit(h, log, loss_key="loss") -> str:
    loss, gn = log.col(loss_key), log.col("grad_norm")
    t, r = h["step"], h["reacts_at"]
    lines = [
        f"step {t}: the gradient moved, the loss did not.",
        "",
        f"{'step':>6}  {'grad norm':>10}  {'d(gn)':>9}  {'z(gn)':>7}   {'loss':>8}  {'d(loss)':>9}  {'z(loss)':>8}",
        "-" * 74,
    ]
    zl, zg = h["_zl"], h["_zg"]
    for s in range(max(0, t - 3), min(len(loss), r + 3)):
        mark = "  <-- grad norm jumps here" if s == t else (
               "  <-- loss reacts here" if s == r else "")
        dgn = gn[s] - gn[s - 1] if s else 0.0
        dls = loss[s] - loss[s - 1] if s else 0.0
        lines.append(f"{s:>6}  {gn[s]:>10.4f}  {dgn:>+9.4f}  {zg[s]:>+7.2f}   "
                     f"{loss[s]:>8.4f}  {dls:>+9.4f}  {zl[s]:>+8.2f}{mark}")
    lines += [
        "",
        f"grad norm at step {t}: {h['grad_norm']:.4f}  (z = {h['z_grad']:+.2f}, "
        f"a {abs(h['z_grad']):.1f}-sigma surprise against its own past)",
        f"loss      at step {t}: {h['loss']:.4f}  (z = {h['z_loss']:+.2f}, "
        f"well inside the noise - nothing visible had happened yet)",
        f"loss      at step {r}: {h['loss_react']:.4f}  (z = {h['z_loss_react']:+.2f}, "
        f"moving {h['delta_loss_react']:+.4f} in one step)",
        "",
        f"lag from gradient signal to loss response: {h['lag']} step(s).",
    ]
    return "\n".join(lines)


def rank_hits(hits):
    """Best evidence first: a big gradient surprise, a quiet loss, a sharp reaction."""
    return sorted(hits, key=lambda h: -(abs(h["z_grad"]) * abs(h["z_loss_react"])
                                        / max(abs(h["z_loss"]), 0.25)))


def instability_run(dataset, lr=1.5e-2, steps=150, seed=1337, device=None):
    """Corroboration at large amplitude: same mechanism, turned up until it is loud.

    Run without gradient clipping at a learning rate past the stable range. The
    lead is the same one the stable run shows in miniature; here the gradient
    spike is several times the local baseline and impossible to argue with.
    """
    from .train import TrainConfig, train
    cfg = TrainConfig(steps=steps, lr=lr, warmup=10, grad_clip=0.0, seed=seed, device=device)
    _, log = train(dataset, cfg, verbose=False, probe=True)
    return log


def _events(driver, follower, g_thresh, quiet, react, horizon, alpha, warmup):
    """Count steps where `driver` was surprising, `follower` was not, and then was."""
    zd = causal_z(driver, alpha, warmup)
    zf = causal_z(follower, alpha, warmup)
    n = 0
    for t in range(warmup, len(driver) - horizon):
        if abs(zd[t]) < g_thresh or abs(zf[t]) > quiet:
            continue
        if max(abs(zf[t + k]) for k in range(1, horizon + 1)) >= react:
            n += 1
    return n


def lead_symmetry_test(dataset, seeds=(1337, 7, 21, 99, 2024), steps=250,
                       g_thresh=2.0, quiet=1.0, react=1.5, horizon=3,
                       alpha=0.15, warmup=10, verbose=True, device=None):
    """Does the grad norm lead the loss more often than the loss leads the grad norm?

    A single anecdote proves nothing, and the lagged correlation on a stable run
    is buried in noise. This asks the question as a SYMMETRY test instead, which
    is much harder to fool:

      forward : grad norm surprises at t, loss is quiet at t, loss moves by t+3
      reverse : loss surprises at t, grad norm is quiet at t, grad norm moves by t+3

    Both directions use the identical detector and identical thresholds, so any
    imbalance between them is a property of the training dynamics rather than of
    the test. If the two counts come out level, the honest conclusion is that
    this run gives no evidence of a lead, and that is what gets reported.
    """
    from .train import TrainConfig, train

    rows = []
    for seed in seeds:
        cfg = TrainConfig(steps=steps, seed=seed, device=device)
        _, log = train(dataset, cfg, verbose=False, probe=True)
        gn = log.col("grad_norm")
        pl = log.col("probe_loss")
        fwd = _events(gn, pl, g_thresh, quiet, react, horizon, alpha, warmup)
        rev = _events(pl, gn, g_thresh, quiet, react, horizon, alpha, warmup)
        rows.append(dict(seed=seed, forward=fwd, reverse=rev))
        if verbose:
            print(f"  seed {seed:>5}: grad-norm-leads {fwd:>3}   loss-leads {rev:>3}")

    F = sum(r["forward"] for r in rows)
    R = sum(r["reverse"] for r in rows)

    # Exact two-sided binomial test against the null "the two directions are
    # equally likely". 18-vs-10 sounds convincing until you ask this; a ratio
    # with no p-value attached is how you end up believing noise.
    n = F + R
    if n:
        c = math.comb
        tail = [c(n, k) for k in range(n + 1)]
        obs = min(c(n, F), c(n, R))
        p_two = sum(t for t in tail if t <= obs) / (2 ** n)
        p_two = min(1.0, p_two)
    else:
        p_two = float("nan")

    if verbose:
        print(f"  {'-'*44}")
        print(f"  {'total':>10}: grad-norm-leads {F:>3}   loss-leads {R:>3}")
        if n == 0:
            verdict = "no events either way - test inconclusive"
        elif F > R:
            verdict = (f"grad norm leads {F/max(R,1):.2f}x more often "
                       f"({F/n:.0%} of {n} events)")
        elif R > F:
            verdict = (f"loss leads MORE often ({R/n:.0%}) - "
                       f"no support for the grad-norm-leads claim")
        else:
            verdict = "exactly level - no evidence of a lead in either direction"
        print(f"  verdict: {verdict}")
        if n:
            sig = "significant" if p_two < 0.05 else "NOT significant at the 5% level"
            print(f"  exact two-sided binomial test vs 50/50: p = {p_two:.3f}  ->  {sig}")
            if p_two >= 0.05:
                print(f"  so: the lean is in the predicted direction, but {n} events is too")
                print(f"      few to call it. Reported as suggestive, not established.")
    return dict(rows=rows, forward=F, reverse=R, n=n, p_value=p_two,
                ratio=(F / R if R else float("inf")) if F else 0.0)
