"""Q5 - Model FLOPs Utilization, computed honestly.

    MFU = (model FLOPs actually required per second) / (peak FLOPs the device can do)

"Model FLOPs" means the arithmetic the maths demands, not the arithmetic the
hardware happened to execute. Recomputation, padding waste and masked-away
attention all cost real time but earn no credit. That is the point: MFU is
supposed to punish them.

FLOPs per token
---------------
Two contributions, forward pass:

  (a) weight matmuls. Every parameter that sits in a matmul is touched once per
      token as a multiply-accumulate = 2 FLOPs.
          per block: c_attn 3C*C + c_proj C*C + c_fc 4C*C + c_proj 4C*C = 12*C^2
          plus the lm_head: V*C
      -> N_mm = L*12*C^2 + V*C

  (b) attention score/value matmuls. These involve NO parameters, and they grow
      with sequence length:
          QK^T   per token: 2*T*D per head * H heads = 2*T*C
          att@V  per token: 2*T*C
      -> 4*L*T*C

Backward costs 2x forward (one matmul for the input gradient, one for the weight
gradient), so:

    flops_per_token = 6*N_mm + 12*L*T*C

Embedding and position lookups are gathers, not matmuls: ~0 FLOPs. LayerNorms,
GELU, softmax and the residual adds are elementwise; they are real time but a
rounding error in FLOPs, and by convention they are excluded. Excluding them is
what makes MFU a *lower* bound on efficiency, which is the conservative
direction, so it is the right convention to keep.

Peak FLOPs
----------
Reported against two denominators, because only one of them is arguable:

  * theoretical  - from the ISA: AVX-512 gives 16 fp32 lanes, FMA counts 2
                   FLOPs, assume 2 FMA ports/core. Nominal clock. On a
                   virtualised part with unknown turbo and AVX-512 licence-based
                   downclocking, this number has real uncertainty.
  * measured     - the best sustained throughput a large square SGEMM actually
                   reaches on this machine. Nothing in this process is going to
                   beat a big BLAS matmul, so this is the practical ceiling.

MFU against the measured ceiling is the number worth acting on.
"""

from __future__ import annotations

import platform
import re
import time

import torch

from .model import GPTConfig, TinyGPT


# ---------------------------------------------------------------- FLOP counting

def flops_per_token(cfg: GPTConfig, seq_len: int | None = None) -> dict:
    T = seq_len or cfg.block_size
    C, L, V = cfg.n_embd, cfg.n_layer, cfg.vocab_size

    n_mm_block = 12 * C * C          # c_attn 3C^2 + c_proj C^2 + c_fc 4C^2 + mlp c_proj 4C^2
    n_mm = L * n_mm_block + V * C    # + tied lm_head
    fwd_weight = 2 * n_mm
    fwd_attn = 4 * L * T * C         # QK^T and att@V, both 2*T*C per layer
    fwd = fwd_weight + fwd_attn
    total = 3 * fwd                  # backward = 2x forward

    return dict(
        T=T, C=C, L=L, V=V,
        n_matmul_params=n_mm,
        fwd_weight=fwd_weight, fwd_attn=fwd_attn, fwd_total=fwd,
        per_token=total,
        attn_share=fwd_attn / fwd,
    )


def flops_per_step(cfg: GPTConfig, batch_size: int, seq_len: int) -> int:
    return flops_per_token(cfg, seq_len)["per_token"] * batch_size * seq_len


# ---------------------------------------------------------------- device peak

def cpu_info() -> dict:
    name, mhz = platform.processor() or "unknown", None
    flags = set()
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        m = re.search(r"model name\s*:\s*(.+)", txt)
        if m:
            name = m.group(1).strip()
        m = re.search(r"cpu MHz\s*:\s*([\d.]+)", txt)
        if m:
            mhz = float(m.group(1))
        m = re.search(r"^flags\s*:\s*(.+)$", txt, re.M)
        if m:
            flags = set(m.group(1).split())
    except OSError:
        pass
    m = re.search(r"@\s*([\d.]+)\s*GHz", name)
    ghz = float(m.group(1)) if m else (mhz / 1000 if mhz else None)

    if "avx512f" in flags:
        isa, lanes, fma_ports = "AVX-512", 16, 2
    elif "avx2" in flags and "fma" in flags:
        isa, lanes, fma_ports = "AVX2+FMA", 8, 2
    else:
        isa, lanes, fma_ports = "SSE/scalar", 4, 1

    cores = torch.get_num_threads()
    try:
        import subprocess
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        c = re.search(r"Core\(s\) per socket:\s*(\d+)", out)
        s = re.search(r"Socket\(s\):\s*(\d+)", out)
        if c and s:
            cores = int(c.group(1)) * int(s.group(1))
    except Exception:
        pass

    flops_per_cycle = lanes * 2 * fma_ports      # 2 FLOPs per FMA
    peak = (ghz * 1e9 * cores * flops_per_cycle) if ghz else None
    return dict(name=name, ghz=ghz, cores=cores, isa=isa, lanes=lanes,
                fma_ports=fma_ports, flops_per_cycle=flops_per_cycle,
                theoretical_peak=peak,
                has_amx=("amx_tile" in flags), has_avx512_bf16=("avx512_bf16" in flags))


def measured_peak(sizes=(1024, 2048), reps=8, warmup=3) -> dict:
    """Largest sustained SGEMM throughput this box reaches. The practical ceiling."""
    best = 0.0
    detail = []
    for n in sizes:
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        for _ in range(warmup):
            a @ b
        t0 = time.perf_counter()
        for _ in range(reps):
            a @ b
        dt = (time.perf_counter() - t0) / reps
        gf = (2 * n ** 3) / dt / 1e9
        detail.append(dict(n=n, seconds=dt, gflops=gf))
        best = max(best, gf)
    return dict(gflops=best, detail=detail)


# ---------------------------------------------------------------- measurement

def time_step_phases(dataset, cfg: GPTConfig, batch_size: int, seq_len: int,
                     reps: int = 20, warmup: int = 5) -> dict:
    """Time the step, split into phases, so the MFU gap can be attributed."""
    torch.manual_seed(0)
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(0)
    model.train()

    acc = dict(data=0.0, forward=0.0, backward=0.0, gradnorm=0.0, clip=0.0, opt=0.0, total=0.0)
    for i in range(warmup + reps):
        rec = i >= warmup
        t_all = time.perf_counter()

        t = time.perf_counter()
        x, y = dataset.fixed_batch(batch_size, seq_len, generator=gen)
        d_data = time.perf_counter() - t

        opt.zero_grad(set_to_none=True)
        t = time.perf_counter()
        _, loss, _ = model(x, targets=y)
        d_fwd = time.perf_counter() - t

        t = time.perf_counter()
        loss.backward()
        d_bwd = time.perf_counter() - t

        t = time.perf_counter()
        gn = sum(float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None)
        d_gn = time.perf_counter() - t

        t = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        d_clip = time.perf_counter() - t

        t = time.perf_counter()
        opt.step()
        d_opt = time.perf_counter() - t

        d_total = time.perf_counter() - t_all
        if rec:
            acc["data"] += d_data; acc["forward"] += d_fwd; acc["backward"] += d_bwd
            acc["gradnorm"] += d_gn; acc["clip"] += d_clip; acc["opt"] += d_opt
            acc["total"] += d_total

    return {k: v / reps for k, v in acc.items()}


def compute_mfu(dataset, cfg: GPTConfig, batch_size: int = 16, seq_len: int = 128,
                reps: int = 20, verbose: bool = True) -> dict:
    fl = flops_per_token(cfg, seq_len)
    step_flops = fl["per_token"] * batch_size * seq_len
    phases = time_step_phases(dataset, cfg, batch_size, seq_len, reps=reps)
    info = cpu_info()
    peak = measured_peak()

    dt = phases["total"]
    achieved = step_flops / dt
    mfu_theory = achieved / info["theoretical_peak"] if info["theoretical_peak"] else None
    mfu_measured = achieved / (peak["gflops"] * 1e9)
    tokens_per_s = batch_size * seq_len / dt

    res = dict(flops=fl, step_flops=step_flops, phases=phases, cpu=info, peak=peak,
               achieved_flops=achieved, mfu_theoretical=mfu_theory,
               mfu_measured_ceiling=mfu_measured, tokens_per_s=tokens_per_s,
               batch_size=batch_size, seq_len=seq_len, dt=dt)

    if verbose:
        print(f"device : {info['name']}")
        print(f"         {info['cores']} physical cores @ {info['ghz']} GHz, {info['isa']}"
              f" ({info['lanes']} fp32 lanes x 2 FLOPs/FMA x {info['fma_ports']} ports"
              f" = {info['flops_per_cycle']} FLOPs/cycle/core)")
        print(f"         AMX present: {info['has_amx']}, AVX-512 bf16: {info['has_avx512_bf16']}"
              f"   (neither is reachable from fp32 eager PyTorch)")
        print()
        print(f"theoretical fp32 peak     : {info['theoretical_peak']/1e9:8.1f} GFLOP/s")
        for d in peak["detail"]:
            print(f"measured SGEMM {d['n']}x{d['n']}    : {d['gflops']:8.1f} GFLOP/s")
        print(f"practical ceiling (best)  : {peak['gflops']:8.1f} GFLOP/s "
              f"= {peak['gflops']*1e9/info['theoretical_peak']:.1%} of theoretical")
        print()
        print(f"model FLOPs accounting (B={batch_size}, T={seq_len}):")
        print(f"  matmul parameters        : {fl['n_matmul_params']:,}")
        print(f"  fwd weight FLOPs/token   : {fl['fwd_weight']:,}")
        print(f"  fwd attention FLOPs/token: {fl['fwd_attn']:,}  ({fl['attn_share']:.1%} of forward)")
        print(f"  total FLOPs/token (f+b)  : {fl['per_token']:,}")
        print(f"  FLOPs per step           : {step_flops/1e9:.4f} GFLOP")
        print()
        print(f"measured step time        : {dt*1e3:.2f} ms  ({tokens_per_s:,.0f} tokens/s)")
        print(f"achieved                  : {achieved/1e9:8.2f} GFLOP/s")
        print()
        print(f"  MFU vs theoretical peak    : {mfu_theory:6.2%}")
        print(f"  MFU vs measured SGEMM peak : {mfu_measured:6.2%}")
        print()
        print("where the step time actually goes:")
        for k in ("data", "forward", "backward", "gradnorm", "clip", "opt"):
            print(f"  {k:9s} {phases[k]*1e3:7.2f} ms  {phases[k]/dt:6.1%}")
        acct = phases['total'] - sum(phases[k] for k in ('data','forward','backward','gradnorm','clip','opt'))
        print(f"  {'unaccounted':9s} {acct*1e3:7.2f} ms  {acct/dt:6.1%}")
    return res


def matmul_shape_census(cfg: GPTConfig, batch_size: int, seq_len: int, peak_gflops: float):
    """Every matmul in the forward pass, with the throughput its shape can reach.

    A matmul's efficiency is governed by its shape. (M,K)x(K,N) with a small K or
    N cannot saturate a vector unit no matter how good the kernel is. This is the
    single biggest reason a small model has low MFU, so it is worth measuring
    rather than asserting.
    """
    B, T, C, L, V = batch_size, seq_len, cfg.n_embd, cfg.n_layer, cfg.vocab_size
    shapes = [
        ("c_attn   (B*T,C)x(C,3C)", B * T, C, 3 * C, L),
        ("attn QK^T per head",      B * cfg.n_head * T, cfg.head_dim, T, L),
        ("attn AV   per head",      B * cfg.n_head * T, T, cfg.head_dim, L),
        ("attn c_proj (B*T,C)x(C,C)", B * T, C, C, L),
        ("mlp c_fc  (B*T,C)x(C,4C)", B * T, C, 4 * C, L),
        ("mlp c_proj (B*T,4C)x(4C,C)", B * T, 4 * C, C, L),
        ("lm_head  (B*T,C)x(C,V)",  B * T, C, V, 1),
    ]
    rows = []
    for name, M, K, N, count in shapes:
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        for _ in range(3):
            a @ b
        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            a @ b
        dt = (time.perf_counter() - t0) / reps
        gf = 2 * M * K * N / dt / 1e9
        rows.append(dict(name=name, M=M, K=K, N=N, count=count, gflops=gf,
                         frac_of_peak=gf / peak_gflops,
                         flops=2 * M * K * N * count))
    return rows


def op_attribution(dataset, cfg: GPTConfig, batch_size=16, seq_len=128, reps=5):
    """Operator-level attribution of step time: matmul vs everything else.

    Model FLOPs are earned ONLY by mm/addmm/bmm. Every other operator - softmax,
    GELU, LayerNorm, the causal masked_fill, the transpose copies, the AdamW
    update - costs wall clock and earns nothing. This measures that split
    instead of guessing at it.
    """
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(0)
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    x, y = dataset.fixed_batch(batch_size, seq_len, generator=gen)

    for _ in range(5):                     # warm up before profiling
        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=y)
        loss.backward()
        opt.step()

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(reps):
            opt.zero_grad(set_to_none=True)
            _, loss, _ = model(x, targets=y)
            loss.backward()
            opt.step()

    MATMUL = {"aten::mm", "aten::addmm", "aten::bmm", "aten::matmul"}
    rows, total = [], 0.0
    for e in prof.key_averages():
        t = e.self_cpu_time_total
        if t <= 0:
            continue
        total += t
        rows.append((e.key, t, e.count))
    rows.sort(key=lambda r: -r[1])
    mm_time = sum(t for k, t, _ in rows if k in MATMUL)
    return dict(rows=rows, total_us=total, matmul_us=mm_time, reps=reps,
                matmul_frac=mm_time / total, other_frac=1 - mm_time / total)


def size_sweep(dataset, configs=((128, 4, 16, 128), (192, 4, 16, 128), (256, 4, 16, 128),
                                 (384, 4, 16, 128), (512, 4, 16, 128)),
               reps=5, verbose=True):
    """MFU as a function of model size, holding the machine fixed.

    If low MFU were caused by something fundamental about this machine, growing
    the model would not help. If it is caused by fixed per-step overhead that
    does not scale with C^2, growing the model fixes it. This distinguishes the
    two hypotheses by experiment.

    Batch size, sequence length and depth are held FIXED across the sweep so that
    width is the only thing that varies. Letting T grow at the same time would
    confound the two, since the attention FLOP term scales with T as well.
    """
    out = []
    for (C, L, B, T) in configs:
        cfg = GPTConfig(vocab_size=dataset.vocab_size, block_size=T,
                        n_layer=L, n_head=max(4, C // 64), n_embd=C)
        torch.manual_seed(0)
        npar = TinyGPT(cfg).num_params()
        r = compute_mfu(dataset, cfg, batch_size=B, seq_len=T, reps=reps, verbose=False)
        out.append(dict(n_embd=C, n_layer=L, batch=B, seq=T, params=npar,
                        ms=r["dt"] * 1e3, gflops=r["achieved_flops"] / 1e9,
                        mfu=r["mfu_measured_ceiling"]))
    if verbose:
        print(f"{'n_embd':>6} {'layers':>6} {'B':>3} {'T':>4} {'params':>11} "
              f"{'ms/step':>8} {'GFLOP/s':>8} {'MFU':>7}")
        print("-" * 63)
        for r in out:
            print(f"{r['n_embd']:>6} {r['n_layer']:>6} {r['batch']:>3} {r['seq']:>4} "
                  f"{r['params']:>11,} {r['ms']:>8.1f} {r['gflops']:>8.1f} {r['mfu']:>7.1%}")
    return out
