#!/usr/bin/env python3
"""Regenerate every number and figure in the README.

    python run_all.py            # full run  (~8 minutes on 4 CPU cores)
    python run_all.py --quick    # smaller sweeps, for a fast sanity check

Everything the README claims is produced here. Nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.accumulation import (compare_one_step, control_equal_lengths, train_both)
from src.data import CharDataset, load_text
from src.floats import accumulation_demo, precision_table, report as floats_report
from src.gradcheck import autograd_gradcheck, check_one_weight, sweep_h
from src.gradnorm import (describe_hit, find_lead_steps, instability_run,
                          lag_correlation, lead_symmetry_test, rank_hits)
from src.model import GPTConfig
from src.mfu import (compute_mfu, device_info, matmul_shape_census,
                     op_attribution, size_sweep)
from src.model import pick_device
from src.shapes import trace_one_step
from src.train import TrainConfig, train

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ART, exist_ok=True)

# muted, colour-blind-safe pair used for every two-series comparison
C_GOOD, C_BAD, C_ACCENT, C_GREY = "#0F766E", "#B45309", "#7C3AED", "#64748B"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def capture(fn, *a, **kw):
    """Run fn, echo its stdout to the console, and return (text, value)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        val = fn(*a, **kw)
    text = buf.getvalue()
    sys.stdout.write(text)
    return text, val


def write(name, text):
    path = os.path.join(ART, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  -> artifacts/{name}")


def banner(n, title):
    line = "=" * 78
    print(f"\n{line}\n  Q{n}. {title}\n{line}")


def plot_accumulation(hist, cmp1):
    """Q3 figure. Split out so `--replot` can rebuild it from results.json."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    a = ax[0]
    a.plot(hist["correct"]["step"], hist["correct"]["val"], color=C_GOOD, lw=2,
           label="correct: sum(loss) / sum(tokens)")
    a.plot(hist["mean_of_means"]["step"], hist["mean_of_means"]["val"], color=C_BAD, lw=2,
           label="broken: mean of the means")
    a.set_xlabel("step"); a.set_ylabel("held-out loss (nats/token)")
    a.set_title("Both runs scored the same correct way", fontsize=10)
    # legend goes bottom-left: the curves descend left-to-right, so that corner is
    # the only empty one, and the zoom inset needs the top-right.
    a.legend(frameon=False, fontsize=8, loc="lower left")
    # the curves sit close together at full scale; zoom the tail so the gap is
    # something you can see rather than something you take on trust
    n_tail = max(3, len(hist["correct"]["step"]) // 3)
    ins = a.inset_axes([0.50, 0.52, 0.46, 0.36])
    ins.plot(hist["correct"]["step"][-n_tail:], hist["correct"]["val"][-n_tail:],
             color=C_GOOD, lw=1.6)
    ins.plot(hist["mean_of_means"]["step"][-n_tail:], hist["mean_of_means"]["val"][-n_tail:],
             color=C_BAD, lw=1.6)
    ins.set_title("last third, zoomed", fontsize=7, pad=2)
    ins.tick_params(labelsize=6); ins.grid(alpha=0.2)

    a = ax[1]
    gap = [b - c for b, c in zip(hist["mean_of_means"]["val"], hist["correct"]["val"])]
    a.plot(hist["correct"]["step"], gap, color=C_BAD, lw=2)
    a.axhline(0, color=C_GREY, lw=1, ls="--")
    a.fill_between(hist["correct"]["step"], 0, gap, color=C_BAD, alpha=0.15)
    a.set_xlabel("step"); a.set_ylabel("loss gap (broken - correct)")
    a.set_title(f"Cost of the bug: cos sim {cmp1['cos']:.4f}, "
                f"{cmp1['angle_deg']:.1f}$\\degree$ off", fontsize=10)
    fig.suptitle("Gradient accumulation with unequal micro-batch token counts", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "03_accumulation.png"), bbox_inches="tight")
    plt.close(fig)
    print("  -> artifacts/03_accumulation.png")


def plot_gradnorm(log, hit):
    """Q4 figure: the whole run, plus a zoom on the lead event.

    The full run is 400 steps and the event is two steps wide, so without the
    zoom panel the thing the question actually asks about is invisible.
    """
    steps = log.col("step") if hasattr(log, "col") else log["step"]
    gnorm = log.col("grad_norm") if hasattr(log, "col") else log["grad_norm"]
    probe = log.col("probe_loss") if hasattr(log, "col") else log["probe_loss"]
    tloss = log.col("loss") if hasattr(log, "col") else log["loss"]

    fig = plt.figure(figsize=(12, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], hspace=0.12, wspace=0.22)
    a0 = fig.add_subplot(gs[0, 0])
    a1 = fig.add_subplot(gs[1, 0], sharex=a0)
    z = fig.add_subplot(gs[:, 1])

    a0.plot(steps, gnorm, color=C_ACCENT, lw=0.9)
    a0.set_ylabel("grad norm (pre-clip)")
    a0.set_title("The whole run", fontsize=10)
    a0.tick_params(labelbottom=False)
    a1.plot(steps, probe, color=C_GOOD, lw=1.3, label="fixed probe batch")
    a1.plot(steps, tloss, color=C_GREY, lw=0.6, alpha=0.55, label="training batch (noisy)")
    a1.set_ylabel("loss"); a1.set_xlabel("step")
    a1.legend(frameon=False, fontsize=8)

    if hit is not None:
        t, r = hit["step"], hit["reacts_at"]
        for a in (a0, a1):
            a.axvline(t, color=C_ACCENT, ls="--", lw=1.0, alpha=0.8)
            a.axvline(r, color=C_BAD, ls=":", lw=1.0, alpha=0.8)

        lo, hi = max(0, t - 8), min(len(steps), r + 8)
        w = list(range(lo, hi))
        z.plot(w, [gnorm[i] for i in w], color=C_ACCENT, lw=1.8, marker="o", ms=3,
               label="grad norm")
        z.set_ylabel("grad norm (pre-clip)", color=C_ACCENT)
        z.tick_params(axis="y", labelcolor=C_ACCENT)
        z.set_xlabel("step")
        z2 = z.twinx()
        z2.plot(w, [probe[i] for i in w], color=C_GOOD, lw=1.8, marker="s", ms=3,
                label="probe loss")
        z2.set_ylabel("probe loss", color=C_GOOD)
        z2.tick_params(axis="y", labelcolor=C_GOOD)
        z2.grid(False)
        z.axvline(t, color=C_ACCENT, ls="--", lw=1.2)
        z.axvline(r, color=C_BAD, ls=":", lw=1.2)
        z.set_title(f"Zoom: grad norm jumps at {t} ({hit['z_grad']:+.1f}$\\sigma$),\n"
                    f"loss still quiet ({hit['z_loss']:+.1f}$\\sigma$), reacts at {r} "
                    f"({hit['z_loss_react']:+.1f}$\\sigma$)", fontsize=9)

    fig.suptitle("Grad norm logged every step, against the fixed-probe loss", fontsize=11)
    fig.savefig(os.path.join(ART, "04_gradnorm.png"), bbox_inches="tight")
    plt.close(fig)
    print("  -> artifacts/04_gradnorm.png")


def replot():
    """Rebuild figures from a previous run's results.json, without retraining."""
    with open(os.path.join(ART, "results.json")) as f:
        R = json.load(f)
    acc = R["accumulation"]
    plot_accumulation(acc["curves"], acc)
    if "gradnorm_series" in R:
        plot_gradnorm(R["gradnorm_series"], R.get("gradnorm_best_hit"))
    print("replotted from artifacts/results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild figures from artifacts/results.json, no retraining")
    ap.add_argument("--device", default=None,
                    help="cuda / mps / cpu. Default: best available.")
    args = ap.parse_args()
    if args.replot:
        return replot()
    Q = args.quick
    DEV = pick_device(args.device)

    t_start = time.time()
    results = {}

    text, prov = load_text()
    ds = CharDataset(text, prov)
    print(f"corpus     : {prov}")
    print(f"             {len(text):,} characters, vocab {ds.vocab_size}, sha256[:16] {ds.sha256()}")
    print(f"torch      : {torch.__version__}   threads {torch.get_num_threads()}")
    print(f"python     : {platform.python_version()}")
    print(f"device     : {DEV}  ({device_info(DEV)['name']})")
    print(f"             Q2 and the Q3 control are pinned to CPU - they assert")
    print(f"             bit-exact identities that a GPU may reorder.")
    results["env"] = dict(torch=torch.__version__, python=platform.python_version(),
                          threads=torch.get_num_threads(), corpus_sha=ds.sha256(),
                          corpus_chars=len(text), vocab=ds.vocab_size, provenance=prov,
                          device=str(DEV), device_name=device_info(DEV)["name"])

    # ---------------------------------------------------------------- Q1 shapes
    banner(1, "Every tensor shape in one training step")
    rep = trace_one_step(ds, batch_size=4, seq_len=16)
    print(rep[:1200] + "\n...(full table in artifacts/01_shapes.txt)...")
    write("01_shapes.txt", rep)

    # ------------------------------------------------------------- Q2 gradcheck
    banner(2, "Verify one gradient by hand")
    t1, r1 = capture(check_one_weight, ds)
    print()
    t2, sweep = capture(sweep_h, ds)
    print()
    ok, npar = autograd_gradcheck(ds)
    t3 = f"\ntorch.autograd.gradcheck over all {npar} parameter tensors: {ok}\n"
    print(t3.strip())
    write("02_gradcheck.txt", t1 + "\n\nSTEP-SIZE SWEEP\n" + t2 + t3)
    results["gradcheck"] = dict(
        param=r1["param_name"], row=r1["row"], col=r1["col"],
        analytic=r1["analytic"], numeric=r1["numeric"], rel_err=r1["rel_err"],
        digits=r1["matching_digits"],
        best=min(sweep, key=lambda r: r["rel_err"])["h"],
        best_digits=max(s["matching_digits"] for s in sweep),
        torch_gradcheck=ok, n_param_tensors=npar)

    # ------------------------------------------------------------ Q3 accumulate
    banner(3, "Break gradient accumulation on purpose")
    t1, cmp1 = capture(compare_one_step, ds)
    print()
    t2, ctrl = capture(control_equal_lengths, ds)
    print()
    steps = 60 if Q else 300
    print(f"training two identical models for {steps} steps, one per reduction ...")
    t3, hist = capture(train_both, ds, steps=steps, eval_every=10, device=DEV)
    write("03_accumulation.txt", t1 + "\n" + t2 + "\n\nTRAINING BOTH REDUCTIONS\n" + t3)

    plot_accumulation(hist, cmp1)

    final_gap = hist["mean_of_means"]["val"][-1] - hist["correct"]["val"][-1]
    results["accumulation"] = dict(
        counts=cmp1["counts"], total_tokens=cmp1["total"], weights=cmp1["weights"],
        cos=cmp1["cos"], angle_deg=cmp1["angle_deg"], rel_l2=cmp1["rel_l2"],
        norm_ratio=cmp1["norm_ratio"], control_cos=ctrl["cos"], control_max_abs=ctrl["max_abs"],
        steps=steps, final_correct=hist["correct"]["val"][-1],
        final_broken=hist["mean_of_means"]["val"][-1], final_gap=final_gap,
        curves=hist)

    # -------------------------------------------------------------- Q4 gradnorm
    banner(4, "Grad norm every step; find a step where it moved first")
    nsteps = 120 if Q else 400
    print(f"training {nsteps} steps with a fixed probe batch ...")
    _, log = train(ds, TrainConfig(steps=nsteps, log_every=max(1, nsteps // 4),
                                  device=str(DEV)), verbose=True, probe=True)
    hits, zl, zg = find_lead_steps(log, loss_key="probe_loss")
    print(f"\nlead-step candidates (grad norm surprised, loss did not, loss then moved): {len(hits)}")
    best_txt = ""
    if hits:
        best = rank_hits(hits)[0]
        best_txt = describe_hit(best, log, loss_key="probe_loss")
        print("\n" + best_txt)
        results["gradnorm_best_hit"] = {k: v for k, v in best.items() if not k.startswith("_")}

    print("\nlagged cross-correlation over the whole run:")
    lag_txt = io.StringIO()
    lags = lag_correlation(log, loss_key="probe_loss")
    for r in lags:
        bar = "#" * int(max(0, r["corr"]) * 60)
        line = f"  lag {r['lag']:+d}: {r['corr']:+.4f}  {bar}"
        print(line); lag_txt.write(line + "\n")

    print("\nsymmetry test - is the lead real, or does noise cut both ways?")
    sym_seeds = (1337, 7) if Q else (1337, 7, 21, 99, 2024)
    sym_txt, sym = capture(lead_symmetry_test, ds, device=DEV, seeds=sym_seeds,
                           steps=120 if Q else 250)
    results["gradnorm_symmetry"] = dict(forward=sym["forward"], reverse=sym["reverse"],
                                        ratio=sym["ratio"], rows=sym["rows"],
                                        n=sym["n"], p_value=sym["p_value"],
                                        seeds=list(sym_seeds))

    print("\ncorroboration at large amplitude (high LR, no clipping):")
    ilog = instability_run(ds, steps=60 if Q else 150, device=DEV)
    ign, ipl = ilog.col("grad_norm"), ilog.col("probe_loss")
    ihits, _, _ = find_lead_steps(ilog, loss_key="probe_loss", warmup=8)
    itxt = ""
    if ihits:
        ibest = rank_hits(ihits)[0]
        itxt = describe_hit(ibest, ilog, loss_key="probe_loss")
        print(itxt)
        results["gradnorm_instability_hit"] = {k: v for k, v in ibest.items() if not k.startswith("_")}
    write("04_gradnorm.txt",
          f"lead candidates: {len(hits)}\n\n{best_txt}\n\nLAGGED CROSS-CORRELATION\n"
          f"{lag_txt.getvalue()}\nSYMMETRY TEST\n{sym_txt}\nINSTABILITY RUN\n{itxt}\n")

    plot_gradnorm(log, rank_hits(hits)[0] if hits else None)
    results["gradnorm_series"] = {k: log.col(k) for k in
                                  ("step", "loss", "probe_loss", "grad_norm")}
    results["gradnorm"] = dict(steps=nsteps, n_hits=len(hits),
                               lags=[(r["lag"], r["corr"]) for r in lags],
                               instability_max_gnorm=max(ign),
                               instability_probe_max=max(ipl))

    # ------------------------------------------------------------------- Q5 MFU
    banner(5, "MFU, honestly")
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128)
    t1, mfu = capture(compute_mfu, ds, cfg, batch_size=16, seq_len=128, reps=20, device=DEV)
    print()
    peak = mfu["peak"]["gflops"]
    census = matmul_shape_census(cfg, 16, 128, peak, device=DEV)
    lines = [f"{'matmul':<30} {'M':>6} {'K':>5} {'N':>5} {'x':>2}  {'GFLOP/s':>8}  "
             f"{'% of peak':>9}  {'FLOP share':>10}",
             "-" * 90]
    tot = sum(r["flops"] for r in census)
    for r in census:
        lines.append(f"{r['name']:<30} {r['M']:>6} {r['K']:>5} {r['N']:>5} {r['count']:>2}  "
                     f"{r['gflops']:>8.1f}  {r['frac_of_peak']:>8.1%}  {r['flops']/tot:>9.1%}")
    t_ideal = sum(r["flops"] / (r["gflops"] * 1e9) for r in census)
    shape_ceiling = tot / t_ideal / 1e9
    shape_frac = shape_ceiling / peak
    verdict = ("  (so matmul SHAPE is not the problem here)" if shape_frac > 0.75 else
               f"  (shapes this small reach only {shape_frac:.0%} of the device's own GEMM\n"
               f"   ceiling, so SHAPE is a first-order part of the gap)")
    lines += ["", f"FLOP-weighted matmul throughput: {shape_ceiling:.1f} GFLOP/s "
                  f"= {shape_frac:.1%} of peak", verdict]
    census_txt = "\n".join(lines)
    print(census_txt)

    print()
    attr = op_attribution(ds, cfg, device=DEV)
    alines = [f"matmul ops (mm/addmm/bmm) : {attr['matmul_frac']:.1%} of step time",
              f"everything else          : {attr['other_frac']:.1%} of step time", "",
              f"(call counts are TOTALS over the {attr['reps']} profiled steps, not per-step)", "",
              f"{'operator':<36} {'self %':>7}  {'calls':>7}", "-" * 54]
    for k, t, c in attr["rows"][:18]:
        tag = "  <- matmul" if k in {"aten::mm", "aten::addmm", "aten::bmm"} else ""
        alines.append(f"{k:<36} {t/attr['total_us']:>6.1%}  {c:>7}{tag}")
    attr_txt = "\n".join(alines)
    print(attr_txt)

    print()
    sweep_cfgs = ((128, 4, 16, 128), (256, 4, 16, 128)) if Q else None
    t4, sizes = capture(size_sweep, ds, device=DEV,
                        **({"configs": sweep_cfgs} if Q else {}))
    write("05_mfu.txt", t1 + "\nMATMUL SHAPE CENSUS\n" + census_txt +
          "\n\nOPERATOR ATTRIBUTION\n" + attr_txt + "\n\nMFU vs MODEL SIZE\n" + t4)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    a = ax[0]
    labels = ["data", "forward", "backward", "gradnorm", "clip", "opt"]
    vals = [mfu["phases"][k] * 1e3 for k in labels]
    a.barh(labels, vals, color=[C_GREY, C_GOOD, C_GOOD, C_ACCENT, C_ACCENT, C_BAD])
    a.set_xlabel("ms per step"); a.invert_yaxis()
    a.set_title(f"Step time = {mfu['dt']*1e3:.1f} ms   "
                f"(MFU {mfu['mfu_measured_ceiling']:.1%})", fontsize=10)
    a = ax[1]
    xs = [r["n_embd"] for r in sizes]
    ys = [r["mfu"] * 100 for r in sizes]
    a.plot(xs, ys, "o-", color=C_GOOD, lw=2)
    a.axhline(40, color=C_BAD, ls="--", lw=1.2)
    a.annotate("40% target", xy=(xs[0], 40), xytext=(0, 5),
               textcoords="offset points", color=C_BAD, fontsize=8)
    for r in sizes:
        a.annotate(f"{r['params']/1e6:.1f}M", xy=(r["n_embd"], r["mfu"] * 100),
                   xytext=(0, -13), textcoords="offset points", ha="center",
                   fontsize=7, color=C_GREY)
    a.set_xlabel("n_embd (model width)"); a.set_ylabel("MFU %")
    a.set_title("MFU is a size problem, not a machine problem", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "05_mfu.png"), bbox_inches="tight")
    plt.close(fig)
    print("  -> artifacts/05_mfu.png")

    results["mfu"] = dict(
        device=str(DEV), device_name=mfu["device"]["name"],
        cores=mfu["device"].get("cores"), ghz=mfu["device"].get("ghz"),
        isa=mfu["device"].get("isa"),
        theoretical_peak_gflops=(mfu["device"]["theoretical_peak"] / 1e9
                                 if mfu["device"].get("theoretical_peak") else None),
        peak_by_dtype_gflops=mfu["peak"]["by_dtype"],
        mfu_vs_lowp_ceiling=mfu["mfu_vs_lowp_ceiling"],
        measured_peak_gflops=peak, flops_per_token=mfu["flops"]["per_token"],
        step_flops=mfu["step_flops"], step_ms=mfu["dt"] * 1e3,
        achieved_gflops=mfu["achieved_flops"] / 1e9,
        mfu_theoretical=mfu["mfu_theoretical"], mfu_measured=mfu["mfu_measured_ceiling"],
        tokens_per_s=mfu["tokens_per_s"],
        phases={k: v * 1e3 for k, v in mfu["phases"].items()},
        matmul_frac=attr["matmul_frac"], shape_ceiling_gflops=shape_ceiling,
        shape_ceiling_frac=shape_ceiling / peak,
        attn_share_of_fwd=mfu["flops"]["attn_share"],
        size_sweep=sizes)

    # ---------------------------------------------------------------- Q6 floats
    banner(6, "0.1 in fp32, bf16 and fp8 E4M3")
    t1, fl = capture(floats_report)
    t2, prec = capture(precision_table)
    print()
    t3, accd = capture(accumulation_demo)
    write("06_floats.txt", t1 + "\nPRECISION / RANGE\n" + t2 + "\n\nACCUMULATION\n" + t3)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    a = ax[0]
    names = [r["fmt"].name for r in fl]
    errs = [float(r["rel_err"]) * 100 for r in fl]
    bars = a.bar(names, errs, color=[C_GOOD, C_ACCENT, C_BAD])
    a.set_yscale("log"); a.set_ylabel("relative error storing 0.1 (%)")
    a.set_title("Cost of one rounding", fontsize=10)
    for b, e in zip(bars, errs):
        a.annotate(f"{e:.4g}%", xy=(b.get_x() + b.get_width() / 2, e),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    a = ax[1]
    got = [r["got"] for r in accd]
    bars = a.bar([r["name"] for r in accd], got, color=[C_GOOD, C_ACCENT, C_BAD])
    a.axhline(100.0, color=C_GREY, ls="--", lw=1.2)
    a.annotate("exact = 100.0", xy=(2.4, 100), xytext=(0, 4), textcoords="offset points",
               ha="right", fontsize=8, color=C_GREY)
    a.set_ylabel("result"); a.set_title("0.1 added to itself 1000 times", fontsize=10)
    for b, g in zip(bars, got):
        a.annotate(f"{g:.4g}", xy=(b.get_x() + b.get_width() / 2, g),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "06_floats.png"), bbox_inches="tight")
    plt.close(fig)
    print("  -> artifacts/06_floats.png")

    results["floats"] = dict(
        formats=[dict(name=r["fmt"].name, bits=r["fmt"].total_bits,
                      sign=r["sign_bits"], exp=r["exp_bits_str"], mant=r["mant_bits_str"],
                      hex=r["hex"], hw_hex=r["hw_hex"], bits_match=r["bits_match"],
                      stored=r["stored_float"], rel_err=float(r["rel_err"]),
                      exact=f"{r['stored'].numerator}/{r['stored'].denominator}")
                 for r in fl],
        precision=prec,
        accumulation=[dict(name=r["name"], got=r["got"], rel_err=r["rel_err"]) for r in accd],
        all_match=all(r["bits_match"] for r in fl))

    # ------------------------------------------------------------------- finish
    results["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(ART, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*78}")
    print(f"  done in {results['runtime_seconds']:.0f}s -> artifacts/results.json")
    print(f"{'='*78}")
    return results


if __name__ == "__main__":
    main()
