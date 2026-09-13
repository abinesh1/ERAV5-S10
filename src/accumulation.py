"""Q3 - break gradient accumulation on purpose, and measure the damage.

The bug
-------
You want the gradient of the mean loss over a big batch, but the big batch does
not fit, so you split it into N micro-batches and accumulate. The tempting line
is:

    for mb in micro_batches:
        loss = model(mb).mean()      # mean over the tokens IN THIS micro-batch
        (loss / N).backward()        # "average of the averages"

That is right if and only if every micro-batch holds the same number of tokens.
With variable-length sequences it is wrong, and wrong in a specific way: it
gives every micro-batch an equal vote regardless of how much data is in it.

Write out the weight each individual token ends up with:

    correct        w_i = 1 / sum_m(n_m)          same for every token
    mean-of-means  w_i = 1 / (N * n_m)           depends on which micro-batch it landed in

So a token in a short micro-batch counts for sum(n)/(N*n_m) times as much as it
should. Short micro-batches shout; long ones get muffled.

Why Adam does not save you
--------------------------
Adam is invariant to a global rescale of the gradient, so if this were only a
scale error it would mostly wash out. It is not. It is a REWEIGHTING BETWEEN
EXAMPLES, which rotates the gradient vector. Cosine similarity below is the
number that matters, not the norm ratio.
"""

from __future__ import annotations

import copy

import torch

from .data import skewed_lengths
from .model import GPTConfig, TinyGPT, pick_device
from .train import lr_at, TrainConfig


def make_micro_batches(dataset, lengths_per_micro, generator=None, device=None):
    return [dataset.varlen_batch(L, generator=generator, device=device)
            for L in lengths_per_micro]


def accumulate_grads(model, micro_batches, mode: str):
    """Run one accumulated step's worth of backward passes. Returns reported loss.

    mode="correct"       : sum of token losses / total tokens across ALL micro-batches
    mode="mean_of_means" : (1/N) * sum of per-micro-batch mean losses
    """
    model.zero_grad(set_to_none=True)
    N = len(micro_batches)
    total_tokens = sum(mb[3] for mb in micro_batches)

    reported = 0.0
    for x, y, mask, n_tok in micro_batches:
        if mode == "correct":
            _, loss_sum, _ = model(x, targets=y, pad_mask=mask, loss_reduction="sum")
            contrib = loss_sum / total_tokens
        elif mode == "mean_of_means":
            _, loss_mean, _ = model(x, targets=y, pad_mask=mask, loss_reduction="mean")
            contrib = loss_mean / N
        else:
            raise ValueError(mode)
        contrib.backward()
        reported += float(contrib.detach())
    return reported, total_tokens


def flat_grad(model) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()
                      if p.grad is not None])


def compare_one_step(dataset, n_micro=4, micro_bs=8, lo=16, hi=128, seed=0, verbose=True):
    """Same weights, same data, two reductions. How different are the gradients?"""
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=dataset.vocab_size, block_size=hi,
                    n_layer=4, n_head=4, n_embd=128, dropout=0.0)
    model = TinyGPT(cfg)
    model.eval()

    gen = torch.Generator().manual_seed(seed)
    lengths = skewed_lengths(n_micro, micro_bs, lo, hi, gen)
    micro = make_micro_batches(dataset, lengths, gen)
    counts = [mb[3] for mb in micro]
    total = sum(counts)
    N = len(micro)

    rep_c, _ = accumulate_grads(model, micro, "correct")
    g_correct = flat_grad(model).clone()
    rep_b, _ = accumulate_grads(model, micro, "mean_of_means")
    g_broken = flat_grad(model).clone()

    # accumulate in float64: a 0.8M-element float32 dot product loses enough
    # precision to report a cosine slightly above 1.0, which reads as a bug.
    gc64, gb64 = g_correct.double(), g_broken.double()
    cos = float(torch.dot(gc64, gb64) / (gc64.norm() * gb64.norm()))
    rel_l2 = float((gb64 - gc64).norm() / gc64.norm())
    norm_ratio = float(gb64.norm() / gc64.norm())
    angle_deg = float(torch.rad2deg(torch.arccos(torch.clamp(
        torch.tensor(cos), -1.0, 1.0))))

    weights = [total / (N * n) for n in counts]

    if verbose:
        print(f"micro-batches: {N}, sequences each: {micro_bs}, total real tokens: {total}")
        print()
        print(f"{'micro':>5}  {'tokens':>7}  {'share of batch':>14}  {'token weight vs correct':>24}")
        print(f"{'-'*5}  {'-'*7}  {'-'*14}  {'-'*24}")
        for m, (n, w) in enumerate(zip(counts, weights)):
            print(f"{m:5d}  {n:7d}  {n/total:13.1%}  {w:23.3f}x")
        print()
        print(f"  longest/shortest micro-batch token ratio : {max(counts)/min(counts):.2f}x")
        print(f"  over/under-weighting spread              : {max(weights)/min(weights):.2f}x")
        print()
        print(f"reported loss, correct       : {rep_c:.6f}")
        print(f"reported loss, mean-of-means : {rep_b:.6f}   (delta {rep_b-rep_c:+.6f})")
        print()
        print("gradient comparison at identical weights:")
        print(f"  cosine similarity  : {cos:.6f}   (1.0 would mean same direction)")
        print(f"  angle between them : {angle_deg:.3f} degrees")
        print(f"  relative L2 error  : {rel_l2:.4%}")
        print(f"  norm ratio         : {norm_ratio:.4f}   (Adam largely absorbs THIS part)")
    return dict(counts=counts, total=total, weights=weights, cos=cos,
                rel_l2=rel_l2, norm_ratio=norm_ratio, angle_deg=angle_deg,
                reported_correct=rep_c, reported_broken=rep_b)


def control_equal_lengths(dataset, n_micro=4, micro_bs=8, L=64, seed=0, verbose=True):
    """Control: with EQUAL token counts the two reductions must agree exactly.

    If this does not come out at cosine 1.0, the bug being demonstrated is not
    the bug being claimed.
    """
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=dataset.vocab_size, block_size=L,
                    n_layer=4, n_head=4, n_embd=128, dropout=0.0)
    model = TinyGPT(cfg)
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    lengths = [[L] * micro_bs for _ in range(n_micro)]
    micro = make_micro_batches(dataset, lengths, gen)

    accumulate_grads(model, micro, "correct")
    gc = flat_grad(model).clone()
    accumulate_grads(model, micro, "mean_of_means")
    gb = flat_grad(model).clone()

    gc64, gb64 = gc.double(), gb.double()
    cos = float(torch.dot(gc64, gb64) / (gc64.norm() * gb64.norm()))
    max_abs = float((gb - gc).abs().max())
    if verbose:
        print(f"CONTROL - all micro-batches exactly {L*micro_bs} tokens:")
        print(f"  cosine similarity      : {cos:.10f}")
        print(f"  max |elementwise diff| : {max_abs:.3e}  (0.0 = bit-for-bit identical)")
        print("  => identical, as the algebra requires. The bug needs unequal lengths.")
    return dict(cos=cos, max_abs=max_abs)


@torch.no_grad()
def eval_loss(model, dataset, n_batches=20, batch_size=16, block_size=128, seed=999):
    """Honest yardstick: correct token-weighted loss on held-out data.

    Both runs are scored with THIS, regardless of what they optimised, so the
    curves are comparable.
    """
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    tot_loss, tot_tok = 0.0, 0
    for _ in range(n_batches):
        x, y = dataset.fixed_batch(batch_size, block_size, split="val", generator=gen,
                                   device=dev)
        _, loss_sum, n = model(x, targets=y, loss_reduction="sum")
        tot_loss += float(loss_sum)
        tot_tok += n
    model.train()
    return tot_loss / tot_tok


def train_both(dataset, steps=200, n_micro=4, micro_bs=8, lo=16, hi=128,
               lr=3e-3, seed=1337, eval_every=10, verbose=True, device=None):
    """Two identical models, identical data stream, different reductions.

    Same init, same seed, same micro-batches at every step. The ONLY difference
    is how the micro-batch losses are combined.

    This one runs on the accelerator when there is one - it is 300 steps x 2
    models x 4 micro-batches and nothing here asserts a bit-exact identity.
    """
    dev = pick_device(device)
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=dataset.vocab_size, block_size=hi,
                    n_layer=4, n_head=4, n_embd=128, dropout=0.0)
    base = TinyGPT(cfg).to(dev)

    models = {"correct": copy.deepcopy(base), "mean_of_means": copy.deepcopy(base)}
    opts = {k: torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95),
                                 weight_decay=0.1) for k, m in models.items()}
    tcfg = TrainConfig(steps=steps, lr=lr, warmup=max(1, steps // 10))

    hist = {k: {"step": [], "val": [], "reported": []} for k in models}

    gen = torch.Generator().manual_seed(seed)
    for step in range(steps):
        lengths = skewed_lengths(n_micro, micro_bs, lo, hi, gen)
        micro = make_micro_batches(dataset, lengths, gen, device=dev)
        lr_now = lr_at(step, tcfg)

        for mode, model in models.items():
            for g in opts[mode].param_groups:
                g["lr"] = lr_now
            rep, _ = accumulate_grads(model, micro, mode)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opts[mode].step()
            if step % eval_every == 0 or step == steps - 1:
                hist[mode]["step"].append(step)
                hist[mode]["val"].append(eval_loss(model, dataset))
                hist[mode]["reported"].append(rep)

        if verbose and (step % (eval_every * 5) == 0 or step == steps - 1):
            c = hist["correct"]["val"][-1]
            b = hist["mean_of_means"]["val"][-1]
            print(f"step {step:4d} | val(correct) {c:.4f} | val(mean-of-means) {b:.4f} "
                  f"| gap {b-c:+.4f}")
    return hist
