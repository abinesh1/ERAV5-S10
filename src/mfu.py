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
The numerator above is settled by algebra. The DENOMINATOR is the arguable half,
and it is where most quoted MFU numbers quietly cheat, so every denominator that
could be defended is reported and labelled:

  * datasheet    - the vendor number for the part. On a CPU it is derived from
                   the ISA (AVX-512 gives 16 fp32 lanes, FMA counts 2 FLOPs, 2
                   FMA ports/core, nominal clock) and carries real uncertainty on
                   a virtualised part with unknown turbo and licence-based
                   downclocking. On a GPU it is read from a table of published
                   peaks. Optimistic by construction; nothing ever reaches it.
  * measured fp32 - the best a large square fp32 GEMM actually sustains on this
                   device. Nothing in this process is going to beat a big GEMM,
                   so this is the practical ceiling, and MFU against it is the
                   number worth acting on.
  * measured low-precision - the same measurement in bf16/fp16/tf32, i.e. on the
                   tensor cores. This is the denominator the published "40% MFU"
                   figures are quoted against. This model trains in fp32 and so
                   cannot reach those units at all - which is a fact about the
                   code, not about the hardware, and is worth measuring rather
                   than asserting.

These can differ by more than an order of magnitude on a modern GPU. An MFU
number without its denominator attached is not a measurement.
"""

from __future__ import annotations

import platform
import re
import time

import torch

from .model import GPTConfig, TinyGPT, pick_device, sync


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


# Vendor dense (non-sparse) peaks for the parts a Colab runtime actually hands
# out. fp32 = plain CUDA cores; tf32/bf16 = tensor cores. These are datasheet
# numbers and datasheets are optimistic, which is exactly why every one of them
# is reported next to a measured SGEMM below. When the two disagree, believe the
# measurement.
GPU_PEAK = {
    "T4":    dict(fp32=8.1e12,  tf32=None,   bf16=None,    lowp=65e12,
                  note="Turing: fp16 tensor cores, no bf16 and no TF32"),
    "V100":  dict(fp32=15.7e12, tf32=None,   bf16=None,    lowp=125e12,
                  note="Volta: fp16 tensor cores, no bf16 and no TF32"),
    "P100":  dict(fp32=9.3e12,  tf32=None,   bf16=None,    lowp=None,
                  note="Pascal: no tensor cores at all"),
    "A100":  dict(fp32=19.5e12, tf32=156e12, bf16=312e12,  lowp=312e12,
                  note="Ampere"),
    "L4":    dict(fp32=30.3e12, tf32=121e12, bf16=242e12,  lowp=242e12,
                  note="Ada Lovelace"),
    "H100":  dict(fp32=67e12,   tf32=495e12, bf16=989e12,  lowp=989e12,
                  note="Hopper SXM"),
}


def _gpu_info(dev) -> dict:
    name = torch.cuda.get_device_name(dev)
    props = torch.cuda.get_device_properties(dev)
    spec = next((v for k, v in GPU_PEAK.items() if k in name.upper()), None)
    return dict(
        kind="cuda", name=name, sm_count=props.multi_processor_count,
        capability=f"{props.major}.{props.minor}",
        total_mem_gb=props.total_memory / 1e9,
        bf16_supported=torch.cuda.is_bf16_supported(),
        theoretical_peak=(spec or {}).get("fp32"),
        peak_tf32=(spec or {}).get("tf32"),
        peak_bf16=(spec or {}).get("bf16"),
        peak_lowp=(spec or {}).get("lowp"),
        note=(spec or {}).get("note", "unknown part - no datasheet peak on file"),
        cores=props.multi_processor_count, ghz=None, isa="CUDA cores + tensor cores",
    )


def _mps_info() -> dict:
    return dict(
        kind="mps", name="Apple Silicon GPU (MPS)", cores=None, ghz=None,
        isa="Apple GPU ALUs", theoretical_peak=None, peak_tf32=None,
        peak_bf16=None, peak_lowp=None,
        note="no published dense FLOP/s peak - measured ceiling only",
    )


def device_info(device=None) -> dict:
    """What the thing we are about to measure actually is."""
    dev = device if isinstance(device, torch.device) else pick_device(device)
    if dev.type == "cuda":
        return _gpu_info(dev)
    if dev.type == "mps":
        return _mps_info()
    info = cpu_info()
    info.update(kind="cpu", peak_tf32=None, peak_bf16=None, peak_lowp=None,
                note="fp32 SIMD only - no matrix engine reachable from eager PyTorch")
    return info


def measured_peak(device=None, sizes=None, reps=8, warmup=3, dtypes=None) -> dict:
    """Largest sustained GEMM throughput this device reaches. The practical ceiling.

    Measured per dtype, because on a GPU "peak" is not one number. fp32 runs on
    the CUDA cores; tf32 and bf16 run on the tensor cores and are several times
    faster. The model here trains in fp32, so `gflops` - the headline ceiling
    MFU is divided by - is the fp32 one. The rest are reported so Q5 can put a
    measured price on the dtype choice instead of asserting one.
    """
    dev = device if isinstance(device, torch.device) else pick_device(device)
    if sizes is None:
        sizes = (2048, 4096) if dev.type != "cpu" else (1024, 2048)
    if dtypes is None:
        dtypes = ["fp32"]
        if dev.type == "cuda":
            dtypes += ["tf32"]
            dtypes += ["bf16"] if torch.cuda.is_bf16_supported() else ["fp16"]
        elif dev.type == "mps":
            dtypes += ["fp16"]

    was_tf32 = torch.backends.cuda.matmul.allow_tf32 if dev.type == "cuda" else None
    by_dtype, detail = {}, []
    try:
        for tag in dtypes:
            if dev.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = (tag == "tf32")
            td = {"fp32": torch.float32, "tf32": torch.float32,
                  "bf16": torch.bfloat16, "fp16": torch.float16}[tag]
            best_for_tag = 0.0
            for n in sizes:
                a = torch.randn(n, n, device=dev, dtype=td)
                b = torch.randn(n, n, device=dev, dtype=td)
                for _ in range(warmup):
                    a @ b
                sync(dev)
                t0 = time.perf_counter()
                for _ in range(reps):
                    a @ b
                sync(dev)
                dt = (time.perf_counter() - t0) / reps
                gf = (2 * n ** 3) / dt / 1e9
                detail.append(dict(dtype=tag, n=n, seconds=dt, gflops=gf))
                best_for_tag = max(best_for_tag, gf)
            by_dtype[tag] = best_for_tag
            del a, b
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if was_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = was_tf32

    return dict(gflops=by_dtype["fp32"], by_dtype=by_dtype, detail=detail,
                fp32_detail=[d for d in detail if d["dtype"] == "fp32"])


# ---------------------------------------------------------------- measurement

def time_step_phases(dataset, cfg: GPTConfig, batch_size: int, seq_len: int,
                     reps: int = 20, warmup: int = 5, device=None) -> dict:
    """Time the step, split into phases, so the MFU gap can be attributed.

    Two passes, on purpose. On a GPU the only way to time a phase is to
    synchronise at its boundary, and those barriers drain the pipeline and make
    the sum of the phases longer than a real step. So:

      pass 1 - one barrier at each end of the step, nothing inside. This is the
               honest step time, and the only one MFU is computed from.
      pass 2 - a barrier after every phase. Distorted in total, but it is what
               tells you which phase the time is in.

    On CPU the barriers are free and the two passes agree.
    """
    dev = device if isinstance(device, torch.device) else pick_device(device)
    torch.manual_seed(0)
    model = TinyGPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(0)
    model.train()

    def one_step(timed: bool):
        """Run a step. `timed` -> synchronise at every phase boundary."""
        d = {}
        mark = time.perf_counter

        t = mark()
        x, y = dataset.fixed_batch(batch_size, seq_len, generator=gen, device=dev)
        if timed: sync(dev); d["data"] = mark() - t; t = mark()

        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=y)
        if timed: sync(dev); d["forward"] = mark() - t; t = mark()

        loss.backward()
        if timed: sync(dev); d["backward"] = mark() - t; t = mark()

        sq = [p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None]
        float(torch.stack(sq).sum().sqrt())
        if timed: sync(dev); d["gradnorm"] = mark() - t; t = mark()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if timed: sync(dev); d["clip"] = mark() - t; t = mark()

        opt.step()
        if timed: sync(dev); d["opt"] = mark() - t
        return d

    PHASES = ("data", "forward", "backward", "gradnorm", "clip", "opt")

    for _ in range(warmup):                       # warm up caches / autotune
        one_step(timed=False)

    sync(dev)                                     # pass 1: clean total
    t0 = time.perf_counter()
    for _ in range(reps):
        one_step(timed=False)
    sync(dev)
    total = (time.perf_counter() - t0) / reps

    acc = {k: 0.0 for k in PHASES}                # pass 2: attribution
    for _ in range(reps):
        d = one_step(timed=True)
        for k in PHASES:
            acc[k] += d[k]
    phases = {k: v / reps for k, v in acc.items()}
    instrumented = sum(phases.values())

    # Report the phases rescaled onto the clean total, so the breakdown adds up
    # to the step time MFU was actually computed from.
    scale = total / instrumented if instrumented else 1.0
    out = {k: v * scale for k, v in phases.items()}
    out["total"] = total
    out["instrumented_total"] = instrumented
    out["barrier_overhead"] = instrumented - total
    return out


def compute_mfu(dataset, cfg: GPTConfig, batch_size: int = 16, seq_len: int = 128,
                reps: int = 20, verbose: bool = True, device=None) -> dict:
    dev = device if isinstance(device, torch.device) else pick_device(device)
    fl = flops_per_token(cfg, seq_len)
    step_flops = fl["per_token"] * batch_size * seq_len
    phases = time_step_phases(dataset, cfg, batch_size, seq_len, reps=reps, device=dev)
    info = device_info(dev)
    peak = measured_peak(dev)

    dt = phases["total"]
    achieved = step_flops / dt
    theo = info.get("theoretical_peak")
    mfu_theory = achieved / theo if theo else None
    mfu_measured = achieved / (peak["gflops"] * 1e9)
    tokens_per_s = batch_size * seq_len / dt

    # What the same run would score if the denominator were the tensor-core peak
    # the "40% MFU" figures are quoted against. Same numerator, honest label.
    lowp_ceiling = peak["by_dtype"].get("bf16") or peak["by_dtype"].get("fp16")
    mfu_lowp = achieved / (lowp_ceiling * 1e9) if lowp_ceiling else None

    res = dict(flops=fl, step_flops=step_flops, phases=phases, device=info, peak=peak,
               achieved_flops=achieved, mfu_theoretical=mfu_theory,
               mfu_measured_ceiling=mfu_measured, mfu_vs_lowp_ceiling=mfu_lowp,
               tokens_per_s=tokens_per_s, batch_size=batch_size, seq_len=seq_len,
               dt=dt, device_type=dev.type)

    if verbose:
        print(f"device : {info['name']}  [{dev.type}]")
        if info["kind"] == "cuda":
            print(f"         {info['sm_count']} SMs, compute capability {info['capability']}, "
                  f"{info['total_mem_gb']:.1f} GB, bf16 {'yes' if info['bf16_supported'] else 'no'}")
        elif info["kind"] == "cpu":
            clock = f"@ {info['ghz']} GHz" if info.get("ghz") else "(clock not reported)"
            print(f"         {info['cores']} physical cores {clock}, {info['isa']}"
                  f" ({info['lanes']} fp32 lanes x 2 FLOPs/FMA x {info['fma_ports']} ports"
                  f" = {info['flops_per_cycle']} FLOPs/cycle/core)")
        print(f"         {info['note']}")
        print()
        if theo:
            print(f"datasheet fp32 peak       : {theo/1e12:8.2f} TFLOP/s")
        for tag in ("tf32", "bf16", "lowp"):
            v = info.get(f"peak_{tag}")
            if v:
                label = "tensor-core peak" if tag == "lowp" else f"datasheet {tag} peak"
                print(f"{label:<26}: {v/1e12:8.2f} TFLOP/s")
        print()
        print("measured GEMM ceilings on this device (what it really reaches):")
        for tag, gf in peak["by_dtype"].items():
            frac = f"  = {gf*1e9/theo:6.1%} of datasheet fp32" if (theo and tag == "fp32") else ""
            print(f"  {tag:<5} {gf/1e3:8.2f} TFLOP/s{frac}")
        print(f"  -> the model trains in fp32, so its ceiling is "
              f"{peak['gflops']/1e3:.2f} TFLOP/s")
        if lowp_ceiling and lowp_ceiling > 1.5 * peak["gflops"]:
            unit = "tensor cores" if info["kind"] == "cuda" else "low-precision units"
            print(f"  -> leaving {lowp_ceiling/peak['gflops']:.1f}x on the table by not "
                  f"using the {unit} at all")
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
        if mfu_theory:
            print(f"  MFU vs datasheet fp32 peak   : {mfu_theory:6.2%}")
        print(f"  MFU vs measured fp32 ceiling : {mfu_measured:6.2%}   <- the honest one")
        if mfu_lowp and lowp_ceiling > 1.5 * peak["gflops"]:
            what = "tensor-core" if info["kind"] == "cuda" else "low-precision"
            print(f"  MFU vs measured {what:<13}: {mfu_lowp:6.2%}   <- the denominator "
                  f"the 40% figures use")
        print()
        print("where the step time actually goes:")
        for k in ("data", "forward", "backward", "gradnorm", "clip", "opt"):
            print(f"  {k:9s} {phases[k]*1e3:7.2f} ms  {phases[k]/dt:6.1%}")
        if phases.get("barrier_overhead", 0) > 0:
            print(f"  (phase barriers cost {phases['barrier_overhead']*1e3:.2f} ms on top of "
                  f"the {dt*1e3:.2f} ms step; shares above are rescaled onto the clean step)")
    return res


def matmul_shape_census(cfg: GPTConfig, batch_size: int, seq_len: int, peak_gflops: float,
                        device=None):
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
    dev = device if isinstance(device, torch.device) else pick_device(device)
    rows = []
    for name, M, K, N, count in shapes:
        a = torch.randn(M, K, device=dev)
        b = torch.randn(K, N, device=dev)
        for _ in range(3):
            a @ b
        sync(dev)
        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            a @ b
        sync(dev)
        dt = (time.perf_counter() - t0) / reps
        gf = 2 * M * K * N / dt / 1e9
        rows.append(dict(name=name, M=M, K=K, N=N, count=count, gflops=gf,
                         frac_of_peak=gf / peak_gflops,
                         flops=2 * M * K * N * count))
    return rows


def op_attribution(dataset, cfg: GPTConfig, batch_size=16, seq_len=128, reps=5, device=None):
    """Operator-level attribution of step time: matmul vs everything else.

    Model FLOPs are earned ONLY by mm/addmm/bmm. Every other operator - softmax,
    GELU, LayerNorm, the causal masked_fill, the transpose copies, the AdamW
    update - costs wall clock and earns nothing. This measures that split
    instead of guessing at it.
    """
    from torch.profiler import ProfilerActivity, profile

    dev = device if isinstance(device, torch.device) else pick_device(device)
    torch.manual_seed(0)
    model = TinyGPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    x, y = dataset.fixed_batch(batch_size, seq_len, generator=gen, device=dev)

    for _ in range(5):                     # warm up before profiling
        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=y)
        loss.backward()
        opt.step()

    acts = [ProfilerActivity.CPU]
    on_gpu = dev.type == "cuda"
    if on_gpu:
        acts.append(ProfilerActivity.CUDA)
    sync(dev)
    with profile(activities=acts) as prof:
        for _ in range(reps):
            opt.zero_grad(set_to_none=True)
            _, loss, _ = model(x, targets=y)
            loss.backward()
            opt.step()
        sync(dev)

    MATMUL = {"aten::mm", "aten::addmm", "aten::bmm", "aten::matmul"}
    rows, total = [], 0.0
    for e in prof.key_averages():
        # On a GPU the CPU time is launch overhead; the kernel time is what the
        # device actually spent. Attribute the one that matters for this device.
        if on_gpu:
            t = getattr(e, "self_device_time_total", None)
            if t is None:
                t = getattr(e, "self_cuda_time_total", 0.0)
        else:
            t = e.self_cpu_time_total
        if t <= 0:
            continue
        total += t
        rows.append((e.key, t, e.count))
    rows.sort(key=lambda r: -r[1])
    mm_time = sum(t for k, t, _ in rows if k in MATMUL)
    return dict(rows=rows, total_us=total, matmul_us=mm_time, reps=reps,
                device=dev.type, measures="device kernel time" if on_gpu else "CPU op time",
                matmul_frac=mm_time / total, other_frac=1 - mm_time / total)


# (n_embd, n_layer, batch, seq) per sweep point. Batch and sequence are FIXED
# within each list so width is the only thing that moves; letting T grow too
# would confound the result, since the attention FLOP term scales with T.
#
# A discrete GPU needs far more work in flight than a CPU before a matmul stops
# being launch-bound, so it gets a bigger fixed (B, T) and a longer reach in
# width. MPS deliberately does NOT get that list: Apple Silicon shares one
# memory pool with the OS, and n_embd=1024 at B=32 drives an 8 GB machine into
# swap - measured, not guessed. Pass `configs=SWEEP_GPU` explicitly on a Mac
# with memory to spare.
SWEEP_CPU = ((128, 4, 16, 128), (192, 4, 16, 128), (256, 4, 16, 128),
             (384, 4, 16, 128), (512, 4, 16, 128))
SWEEP_GPU = ((128, 4, 32, 256), (256, 4, 32, 256), (512, 4, 32, 256),
             (768, 4, 32, 256), (1024, 4, 32, 256))


def size_sweep(dataset, configs=None, reps=5, verbose=True, device=None):
    """MFU as a function of model size, holding the machine fixed.

    If low MFU were caused by something fundamental about this machine, growing
    the model would not help. If it is caused by fixed per-step overhead that
    does not scale with C^2, growing the model fixes it. This distinguishes the
    two hypotheses by experiment.

    Batch size, sequence length and depth are held FIXED across the sweep so that
    width is the only thing that varies. Letting T grow at the same time would
    confound the two, since the attention FLOP term scales with T as well.
    """
    dev = device if isinstance(device, torch.device) else pick_device(device)
    if configs is None:
        configs = SWEEP_GPU if dev.type == "cuda" else SWEEP_CPU
    out = []
    for (C, L, B, T) in configs:
        cfg = GPTConfig(vocab_size=dataset.vocab_size, block_size=T,
                        n_layer=L, n_head=max(4, C // 64), n_embd=C)
        torch.manual_seed(0)
        npar = TinyGPT(cfg).num_params()
        r = compute_mfu(dataset, cfg, batch_size=B, seq_len=T, reps=reps, verbose=False,
                        device=dev)
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
