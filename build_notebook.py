#!/usr/bin/env python3
"""Generate notebook.ipynb. Kept as a script so the notebook is reproducible."""
import json, os

def _lines(src):
    """nbformat wants source as a list of lines that CONCATENATE back to the
    original - so every line but the last needs its trailing newline kept."""
    text = src.strip()
    return [ln + "\n" for ln in text.split("\n")[:-1]] + [text.split("\n")[-1]]

_n = [0]
def _id():
    _n[0] += 1
    return f"cell{_n[0]:02d}"

def md(src):   return {"cell_type": "markdown", "id": _id(), "metadata": {},
                       "source": _lines(src)}
def code(src): return {"cell_type": "code", "id": _id(), "metadata": {},
                       "execution_count": None, "outputs": [], "source": _lines(src)}

cells = [
md(r"""
# ERA V5 - Session 10: making a training loop tell the truth about itself

A small GPT and a real training loop, interrogated six ways:

| # | Question | What it takes to answer honestly |
|---|----------|----------------------------------|
| 1 | Every tensor shape in a step, and what each dimension means | trace all four stages, not just the forward |
| 2 | Verify one gradient by hand | float64, a central difference, and a step-size sweep |
| 3 | Break gradient accumulation on purpose | micro-batches of genuinely different lengths |
| 4 | Log the grad norm; find a step where it moved before the loss | causal baselines only, and a fixed probe batch |
| 5 | Compute MFU and account for the distance to 40% | measure the gap, don't narrate it |
| 6 | 0.1 in fp32, bf16 and fp8 E4M3, showing the bits | exact rational arithmetic, checked against silicon |

Run top to bottom. Roughly 8 minutes on 4 CPU cores.
"""),
code("""
import sys, subprocess
try:
    import torch
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch", "numpy", "matplotlib"], check=True)
    import torch

import matplotlib.pyplot as plt
from src.data import load_text, CharDataset

text, prov = load_text()
ds = CharDataset(text, prov)
print(f"torch {torch.__version__}, {torch.get_num_threads()} threads")
print(f"corpus: {prov}")
print(f"{len(text):,} chars, vocab {ds.vocab_size}, sha256[:16] {ds.sha256()}")
"""),

md(r"""
---
## Q1. Every tensor shape in one training step

A step is forward, loss, backward, and the optimizer update. All four have tensors
with shapes worth knowing, so all four are walked.

Axis letters: **B** batch, **T** position, **C** channels (`n_embd`), **H** heads,
**D** head width (`C/H`), **V** vocab, **L** layers.
"""),
code("""
from src.shapes import trace_one_step
print(trace_one_step(ds, batch_size=4, seq_len=16))
"""),

md(r"""
---
## Q2. Verify one gradient by hand

`backward()` claims a value for $\partial L/\partial w$. That claim is checkable without
trusting autograd at all - move $w$ a hair and watch the loss:

$$\frac{\partial L}{\partial w} \approx \frac{L(w+h) - L(w-h)}{2h}$$

The **central** difference is used because its truncation error is $O(h^2)$ rather than
$O(h)$, which buys several digits for free. Two things have to be right for those digits
to appear: **float64** (a float32 loss has ~7 significant digits, and the subtraction
cancels most of them away) and a **deterministic** loss (dropout 0, fixed batch).
"""),
code("""
from src.gradcheck import check_one_weight
_ = check_one_weight(ds)
"""),
md(r"""
One step size proves nothing on its own - it could be a lucky $h$. Sweeping $h$ shows the
classic error valley: truncation error falling as $h^2$ from the right, round-off error
rising as $1/h$ from the left. Watch the relative error fall by exactly **100x per decade**
on the way down. That factor is the $O(h^2)$ law, measured rather than asserted.
"""),
code("""
from src.gradcheck import sweep_h, autograd_gradcheck
rows = sweep_h(ds)
ok, n = autograd_gradcheck(ds)
print(f"\\ntorch.autograd.gradcheck over all {n} parameter tensors: {ok}")
"""),

md(r"""
---
## Q3. Break gradient accumulation on purpose

Split a big batch into $N$ micro-batches and accumulate. The tempting line is

```python
loss = model(mb).mean()   # mean over the tokens in THIS micro-batch
(loss / N).backward()     # "average of the averages"
```

That is correct **iff every micro-batch holds the same number of tokens**. Write out the
weight each individual token ends up with:

| reduction | weight of token $i$ in micro-batch $m$ |
|---|---|
| correct | $1 / \sum_m n_m$ - the same for every token |
| mean-of-means | $1 / (N\, n_m)$ - depends on which micro-batch it landed in |

So a token in a short micro-batch counts $\sum_m n_m / (N n_m)$ times as much as it should.
**Short micro-batches shout; long ones get muffled.**

Adam will not save you. Adam is invariant to a *global rescale* of the gradient, but this
is a *reweighting between examples*, which rotates the gradient. The number that matters
below is the cosine similarity, not the norm ratio.
"""),
code("""
from src.accumulation import compare_one_step
cmp1 = compare_one_step(ds)
"""),
md("""
**The control.** If the claim is "this bug needs unequal lengths", then equal lengths must
make the two reductions identical. If this does not come out bit-for-bit, the bug being
demonstrated is not the bug being claimed.
"""),
code("""
from src.accumulation import control_equal_lengths
_ = control_equal_lengths(ds)
"""),
md("""
**Both curves together.** Two identical models, same init, same seed, same micro-batches
at every step. The only difference is how the micro-batch losses are combined. Both are
scored with the *same correct* token-weighted held-out loss, so the comparison is
apples-to-apples.
"""),
code("""
from src.accumulation import train_both
hist = train_both(ds, steps=300, eval_every=10)
"""),
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist["correct"]["step"], hist["correct"]["val"],
           color="#0F766E", lw=2, label="correct: sum(loss) / sum(tokens)")
ax[0].plot(hist["mean_of_means"]["step"], hist["mean_of_means"]["val"],
           color="#B45309", lw=2, label="broken: mean of the means")
ax[0].set_xlabel("step"); ax[0].set_ylabel("held-out loss (nats/token)")
ax[0].set_title("Both scored the same correct way"); ax[0].legend(frameon=False)

gap = [b - c for b, c in zip(hist["mean_of_means"]["val"], hist["correct"]["val"])]
ax[1].plot(hist["correct"]["step"], gap, color="#B45309", lw=2)
ax[1].fill_between(hist["correct"]["step"], 0, gap, color="#B45309", alpha=0.15)
ax[1].axhline(0, color="#64748B", ls="--", lw=1)
ax[1].set_xlabel("step"); ax[1].set_ylabel("loss gap (broken - correct)")
ax[1].set_title("The gap, not my word for it")
for a in ax: a.grid(alpha=0.25)
plt.tight_layout(); plt.show()
print(f"final gap: {gap[-1]:+.4f} nats/token")
"""),

md(r"""
---
## Q4. Log the grad norm every step; find a step where it moved first

The loss at step $t$ is a property of the weights *before* step $t$'s update. The grad norm
at step $t$ sizes the update that produces the weights measured at step $t+1$:

$$\text{grad norm at } t \;\longrightarrow\; \text{update at } t \;\longrightarrow\; \text{loss at } t+1$$

A derivative moves before a level, almost by construction.

Two honesty constraints. **(1)** Every baseline is *causal* - an EMA of strictly past
values. A centred window would let the future leak into "was this a surprise", which would
manufacture the very lead we are testing for. **(2)** The per-step training loss uses a
different random batch each step, so its wobble is mostly batch difficulty, not model
change. A **fixed probe batch** is evaluated every step so that a change in it can only
come from the weights.
"""),
code("""
from src.train import TrainConfig, train
from src.gradnorm import find_lead_steps, rank_hits, describe_hit, lag_correlation

_, log = train(ds, TrainConfig(steps=400, log_every=100), verbose=True, probe=True)
hits, zl, zg = find_lead_steps(log, loss_key="probe_loss")
print(f"\\nlead-step candidates: {len(hits)}")
print()
print(describe_hit(rank_hits(hits)[0], log, loss_key="probe_loss"))
"""),
md("""
A single step could be luck. The lagged cross-correlation is the check on that - it is a
statement about the whole run rather than one cherry-picked step. Read the result
honestly, whichever way it comes out.
"""),
code("""
for r in lag_correlation(log, loss_key="probe_loss"):
    print(f"  lag {r['lag']:+d}: {r['corr']:+.4f}  " + "#" * int(max(0, r['corr']) * 60))
"""),
code("""
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
st = log.col("step")
ax[0].plot(st, log.col("grad_norm"), color="#7C3AED", lw=0.9)
ax[0].set_ylabel("grad norm (pre-clip)")
ax[1].plot(st, log.col("probe_loss"), color="#0F766E", lw=1.2, label="fixed probe batch")
ax[1].plot(st, log.col("loss"), color="#64748B", lw=0.6, alpha=0.55, label="training batch")
ax[1].set_ylabel("loss"); ax[1].set_xlabel("step"); ax[1].legend(frameon=False)
b = rank_hits(hits)[0]
for a in ax:
    a.axvline(b["step"], color="#7C3AED", ls="--", lw=1.2)
    a.axvline(b["reacts_at"], color="#B45309", ls=":", lw=1.2)
    a.grid(alpha=0.25)
plt.tight_layout(); plt.show()
"""),
md(r"""
### The test designed to settle it either way

A single step could be luck, and the lagged correlation above is buried in noise. So ask the
question as a **symmetry test** instead, which is much harder to fool — run the *identical*
detector in both directions with *identical* thresholds:

- *forward*: grad norm surprises at $t$, loss is quiet at $t$, loss moves by $t+3$
- *reverse*: loss surprises at $t$, grad norm is quiet at $t$, grad norm moves by $t+3$

Any imbalance is then a property of the training dynamics rather than of the test. And a
ratio with no p-value attached is how you end up believing noise, so it reports an exact
two-sided binomial test against a 50/50 null.
"""),
code("""
from src.gradnorm import lead_symmetry_test
sym = lead_symmetry_test(ds, steps=250)
"""),
md("""
**Corroboration at large amplitude.** Same mechanism, turned up until it is loud: a
learning rate past the stable range, with clipping off.
"""),
code("""
from src.gradnorm import instability_run
ilog = instability_run(ds, steps=150)
ihits, _, _ = find_lead_steps(ilog, loss_key="probe_loss", warmup=8)
if ihits:
    print(describe_hit(rank_hits(ihits)[0], ilog, loss_key="probe_loss"))
"""),

md(r"""
### The honest conclusion

- The **anecdote is real** — the trace above shows it.
- The **mechanism is real**: the loss at step $t$ cannot reflect the update made at step $t$,
  so a lead of at least one step is forced by the ordering of operations, not by any
  empirical claim.
- Whether the lead is **systematic** is decided by the p-value printed above, not by the
  ratio. Read it before believing the ratio.

Why is it so hard to see in a healthy run? **When training is going well, nothing surprising
happens.** The grad norm is a leading indicator of *trouble* — spikes, divergence, a bad
batch, a learning rate past the stable range — and a stable run has little of that to lead.
"""),
md(r"""
---
## Q5. MFU, honestly

$$\text{MFU} = \frac{\text{model FLOPs required per second}}{\text{peak FLOPs the device can do}}$$

"Model FLOPs" is the arithmetic the *maths* demands, not the arithmetic the hardware
happened to run. Recomputation, padding and masked-away attention cost real time and earn
no credit - that is the point.

Per token, forward: weight matmuls contribute $2 N_{mm}$ where
$N_{mm} = L\cdot 12C^2 + VC$, and the parameter-free attention matmuls contribute
$4LTC$. Backward costs $2\times$ forward, so

$$\text{FLOPs/token} = 6 N_{mm} + 12\,L\,T\,C$$

Reported against **two** denominators, because only one of them is arguable: the
theoretical ISA peak, and the best throughput a large SGEMM actually reaches on this box.
"""),
code("""
from src.model import GPTConfig
from src.mfu import compute_mfu

cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128)
mfu = compute_mfu(ds, cfg, batch_size=16, seq_len=128, reps=20)
"""),
md("""
### Where does the distance to 40% go?

Three candidate explanations. Measure all three rather than picking a favourite.

**(a) Are the matmul shapes bad?** A matmul with a small `K` or `N` cannot saturate a
vector unit however good the kernel is.
"""),
code("""
from src.mfu import matmul_shape_census
peak = mfu["peak"]["gflops"]
census = matmul_shape_census(cfg, 16, 128, peak)
tot = sum(r["flops"] for r in census)
print(f"{'matmul':<30} {'GFLOP/s':>9} {'% of peak':>10} {'FLOP share':>11}")
print("-" * 64)
for r in census:
    print(f"{r['name']:<30} {r['gflops']:>9.1f} {r['frac_of_peak']:>9.1%} {r['flops']/tot:>10.1%}")
ceiling = tot / sum(r['flops']/(r['gflops']*1e9) for r in census) / 1e9
print(f"\\nFLOP-weighted matmul throughput: {ceiling:.1f} GFLOP/s = {ceiling/peak:.1%} of peak")
"""),
md("""
**(b) How much of the step is matmul at all?** Only `mm`/`addmm`/`bmm` earn model FLOPs.
Softmax, GELU, LayerNorm, the causal `masked_fill`, the transpose copies and the AdamW
update all cost wall clock and earn nothing.
"""),
code("""
from src.mfu import op_attribution
attr = op_attribution(ds, cfg)
print(f"matmul ops      : {attr['matmul_frac']:.1%} of step time")
print(f"everything else : {attr['other_frac']:.1%} of step time\\n")
for k, t, c in attr["rows"][:12]:
    tag = "  <- matmul" if k in {"aten::mm","aten::addmm","aten::bmm"} else ""
    print(f"  {k:<34} {t/attr['total_us']:>6.1%}  {c:>6} calls{tag}")
"""),
md("""
**(c) Is it the machine, or the model size?** These two hypotheses make opposite
predictions. If low MFU were something fundamental about this CPU, growing the model would
not help. If it is fixed per-step overhead that does not scale with $C^2$, growing the
model fixes it. That is a decidable question.
"""),
code("""
from src.mfu import size_sweep
sizes = size_sweep(ds)
"""),
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
labels = ["data","forward","backward","gradnorm","clip","opt"]
ax[0].barh(labels, [mfu["phases"][k]*1e3 for k in labels],
           color=["#64748B","#0F766E","#0F766E","#7C3AED","#7C3AED","#B45309"])
ax[0].invert_yaxis(); ax[0].set_xlabel("ms per step")
ax[0].set_title(f"step = {mfu['dt']*1e3:.1f} ms, MFU {mfu['mfu_measured_ceiling']:.1%}")
ax[1].plot([r["n_embd"] for r in sizes], [r["mfu"]*100 for r in sizes],
           "o-", color="#0F766E", lw=2)
ax[1].axhline(40, color="#B45309", ls="--", lw=1.2)
ax[1].set_xlabel("n_embd"); ax[1].set_ylabel("MFU %")
ax[1].set_title("MFU is a size problem, not a machine problem")
for a in ax: a.grid(alpha=0.25)
plt.tight_layout(); plt.show()
"""),

md(r"""
---
## Q6. 0.1 in fp32, bf16 and fp8 E4M3

0.1 is not representable in binary, for the same reason 1/3 is not representable in
decimal: $10 = 2\times 5$, and that factor of 5 is not a power of the base.

$$0.1 \times 2 = 0.2 \to 0,\quad 0.2\times 2 = 0.4 \to 0,\quad 0.4\times2=0.8\to0,\quad
0.8\times2=1.6\to1,\quad 0.6\times2=1.2\to1,\quad 0.2\ \text{again}\ \dots$$

$$0.1 = 0.0\overline{0011}_2 = 1.\overline{1001}_2 \times 2^{-4}$$

So in **all three** formats the sign is 0 and the unbiased exponent is $-4$. The only thing
that changes is how many bits of the repeating mantissa $0.6 = .\overline{1001}_2$ survive,
and which way the leftovers round.

Every derivation below is done in exact rational arithmetic (`fractions.Fraction`), never
in floating point - so the "by hand" answer cannot inherit an error from the thing it is
checking - and each is then verified against the real hardware bits.
"""),
code("""
from src.floats import report, precision_table, accumulation_demo
fl = report()
print("all hand-derived bit patterns match hardware:", all(r["bits_match"] for r in fl))
"""),
code("""
_ = precision_table()
"""),
md("""
### Which would I train in?

The one-off rounding error of a single 0.1 is not what decides this. What decides it is
that error **compounding**, which is exactly what gradient accumulation and optimizer
state do all day. Add 0.1 to itself 1000 times in each format:
"""),
code("""
_ = accumulation_demo()
"""),
md(r"""
bf16 stalls at 32 and fp8 at 2, and neither is a rounding curiosity - it is **swamping**.
Once the accumulator grows large enough that the addend falls below *half an ulp* of it,
`acc + x` rounds straight back to `acc` and the sum stops moving forever.

The stall values are predicted exactly by the ulp arithmetic. For a format with $m$ mantissa
bits and an accumulator whose exponent is $e$, one ulp is $2^{e-m}$:

| | $e$ | ulp $=2^{e-m}$ | half-ulp | is $0.1$ above it? | `acc + 0.1` |
|---|---|---|---|---|---|
| bf16 at 16 | 4 | $2^{-3}=0.125$ | 0.0625 | yes | 16.125, still moving |
| **bf16 at 32** | 5 | $2^{-2}=0.25$ | **0.125** | **no** | **32.0, dead** |
| fp8 at 1.875 | 0 | $2^{-3}=0.125$ | 0.0625 | yes | 2.0, still moving |
| **fp8 at 2** | 1 | $2^{-2}=0.25$ | **0.125** | **no** | **2.0, dead** |

Each format dies at exactly the value where its ulp doubles past $0.2$.

**The answer: bf16 for the matmuls, fp32 for anything that accumulates.**

- **Not fp32 everywhere.** It is the safe choice and roughly half the speed. On any tensor-core
  machine you are leaving most of the throughput unused for precision the gradients do not need.
- **bf16 over fp16** - and this is the reason bf16 exists. It keeps all 8 exponent bits, so it
  has the *same dynamic range as fp32* (76.5 decades in the table above) and trades mantissa
  bits instead. Gradients underflow long before they lose meaningful precision, so range is the
  scarce resource. That is also why bf16 needs no loss scaling and fp16 does.
- **Not fp8 E4M3 for the whole loop.** 4.5 decades of range and ~1.9 decimal digits is enough
  for a forward matmul with per-tensor scaling, which is exactly how it is used in practice -
  but master weights, the optimizer moments and the accumulation all stay in fp32. The 1000-add
  experiment above is what happens if you forget that.

The rule underneath all of it: **exponent bits buy you range, mantissa bits buy you precision,
and training runs out of range before it runs out of precision.**
"""),
md("""
---
## Summary

Run `python run_all.py` to regenerate every number and figure here into `artifacts/`,
and see `README.md` for the written answers.
"""),
]

nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out} ({len(cells)} cells)")
