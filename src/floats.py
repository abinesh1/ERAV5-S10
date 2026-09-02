"""Q6 - 0.1 written out by hand in fp32, bf16 and fp8 E4M3.

0.1 is not representable in binary. In decimal, 1/3 needs infinitely many
digits; in binary, 1/10 does too, for the same reason: 10 = 2 * 5, and the
factor of 5 is not a power of the base.

    0.1 x 2 = 0.2  ->  0
    0.2 x 2 = 0.4  ->  0
    0.4 x 2 = 0.8  ->  0
    0.8 x 2 = 1.6  ->  1   (keep 0.6)
    0.6 x 2 = 1.2  ->  1   (keep 0.2)
    0.2 x 2 = 0.4  ->  0   <-- 0.2 already seen, so it cycles from here

    0.1 = 0.0 0011 0011 0011 0011... = 0.0(0011) repeating

Normalise: 0.1 = 1.1001100110011...(0011 repeating) x 2^-4.

So for every one of the three formats the sign is 0 and the exponent is -4;
the ONLY thing that changes is how many bits of the repeating mantissa
0.6 = .1001 1001 1001... survive, and which way the leftovers round.

Every derivation below is done in exact rational arithmetic (fractions.Fraction),
never in floating point, so the "by hand" answer cannot inherit an error from the
thing it is checking. Each result is then verified against the real hardware bits.
"""

from __future__ import annotations

import struct
from fractions import Fraction

import torch


class Format:
    def __init__(self, name, exp_bits, mant_bits, bias=None, has_inf=True, note=""):
        self.name = name
        self.exp_bits = exp_bits
        self.mant_bits = mant_bits
        self.bias = bias if bias is not None else (1 << (exp_bits - 1)) - 1
        self.has_inf = has_inf
        self.note = note
        self.total_bits = 1 + exp_bits + mant_bits

    @property
    def max_exp(self):
        # e4m3fn steals the all-ones exponent for NaN only, so its top exponent
        # is still usable; IEEE formats reserve all-ones for inf/NaN.
        return (1 << self.exp_bits) - 1 - (1 if self.has_inf else 0) - self.bias


FP32 = Format("fp32", 8, 23, note="IEEE 754 binary32")
BF16 = Format("bf16", 8, 7, note="fp32 with 16 mantissa bits chopped off")
FP8_E4M3 = Format("fp8 E4M3", 4, 3, has_inf=False, note="OCP/NVIDIA e4m3fn: no inf, one NaN")


def encode(value: Fraction, fmt: Format) -> dict:
    """Round `value` into `fmt` using round-to-nearest-even. Exact arithmetic."""
    assert value > 0, "only the positive case is needed here"
    sign = 0

    # normalise to 1 <= m < 2
    e = 0
    m = Fraction(value)
    while m >= 2:
        m /= 2
        e += 1
    while m < 1:
        m *= 2
        e -= 1

    # the mantissa field stores the fractional part, scaled by 2^mant_bits
    frac = m - 1                                    # in [0, 1)
    scaled = frac * (1 << fmt.mant_bits)            # exact Fraction
    floor = scaled.numerator // scaled.denominator
    remainder = scaled - floor                      # exact, in [0, 1)

    # round to nearest, ties to even
    if remainder > Fraction(1, 2):
        rounded = floor + 1
    elif remainder < Fraction(1, 2):
        rounded = floor
    else:
        rounded = floor + (floor & 1)               # tie -> make it even

    if rounded == (1 << fmt.mant_bits):             # mantissa overflowed to 2.0
        rounded = 0
        e += 1

    biased = e + fmt.bias
    assert 1 <= biased < (1 << fmt.exp_bits), f"{value} out of range for {fmt.name}"

    stored = Fraction(1) * (1 + Fraction(rounded, 1 << fmt.mant_bits)) * Fraction(2) ** e

    bits = (sign << (fmt.exp_bits + fmt.mant_bits)) | (biased << fmt.mant_bits) | rounded
    return dict(
        fmt=fmt, sign=sign, exp_unbiased=e, exp_biased=biased, mantissa=rounded,
        remainder=remainder, rounded_up=(rounded != floor),
        sign_bits="0",
        exp_bits_str=format(biased, f"0{fmt.exp_bits}b"),
        mant_bits_str=format(rounded, f"0{fmt.mant_bits}b"),
        bits=bits,
        hex=format(bits, f"0{fmt.total_bits // 4}X"),
        stored=stored,
        stored_float=float(stored),
        abs_err=abs(stored - value),
        rel_err=abs(stored - value) / value,
    )


def hardware_bits(fmt_name: str, value: float = 0.1) -> tuple[int, float]:
    """What the actual silicon stores, to check the hand derivation against."""
    if fmt_name == "fp32":
        b = struct.unpack("<I", struct.pack("<f", value))[0]
        return b, struct.unpack("<f", struct.pack("<f", value))[0]
    if fmt_name == "bf16":
        t = torch.tensor([value], dtype=torch.bfloat16)
        return int(t.view(torch.int16).item()) & 0xFFFF, float(t.float())
    if fmt_name == "fp8 E4M3":
        t = torch.tensor([value], dtype=torch.float8_e4m3fn)
        return int(t.view(torch.uint8).item()), float(t.float())
    raise ValueError(fmt_name)


def report(value=Fraction(1, 10), verbose=True) -> list[dict]:
    results = []
    for fmt in (FP32, BF16, FP8_E4M3):
        r = encode(value, fmt)
        hw_bits, hw_val = hardware_bits(fmt.name, float(value))
        r["hw_bits"] = hw_bits
        r["hw_hex"] = format(hw_bits, f"0{fmt.total_bits // 4}X")
        r["hw_value"] = hw_val
        r["bits_match"] = (hw_bits == r["bits"])
        r["value_match"] = (abs(hw_val - r["stored_float"]) == 0.0)
        results.append(r)

    if verbose:
        print(f"decimal {float(value)} = exactly {value.numerator}/{value.denominator}")
        print(f"in binary: 0.0(0011) repeating  =  1.1001100110011... x 2^-4")
        print(f"so in every format below: sign = 0, unbiased exponent = -4,")
        print(f"and the mantissa is as many bits of 0.6 = .1001 1001 1001... as will fit.")
        print()
        for r in results:
            f = r["fmt"]
            print(f"--- {f.name}  ({f.total_bits} bits: 1 sign + {f.exp_bits} exponent "
                  f"+ {f.mant_bits} mantissa, bias {f.bias}) ---")
            print(f"  {f.note}")
            print(f"  exponent field : {r['exp_unbiased']} + {f.bias} = {r['exp_biased']} "
                  f"= {r['exp_bits_str']}")
            print(f"  mantissa field : {r['mant_bits_str']}"
                  f"   ({'rounded UP' if r['rounded_up'] else 'truncated'}; "
                  f"discarded tail = {float(r['remainder']):.4f} of an ulp)")
            print(f"  bit pattern    : {r['sign_bits']} {r['exp_bits_str']} {r['mant_bits_str']}"
                  f"   = 0x{r['hex']}")
            print(f"  hardware says  : 0x{r['hw_hex']}   -> {'MATCH' if r['bits_match'] else 'MISMATCH'}")
            print(f"  value stored   : {r['stored'].numerator}/{r['stored'].denominator}"
                  f" = {r['stored_float']!r}")
            print(f"  relative error : {float(r['rel_err']):.3e}"
                  f"   ({float(r['rel_err'])*100:.4f}%)")
            print()
    return results


def precision_table(verbose=True):
    """Machine epsilon, dynamic range, and what each buys you in training."""
    rows = []
    for fmt, dtype in ((FP32, torch.float32), (BF16, torch.bfloat16), (FP8_E4M3, torch.float8_e4m3fn)):
        eps = 2.0 ** (-fmt.mant_bits)                 # gap between 1.0 and the next value
        decimal_digits = fmt.mant_bits * 0.30103      # log10(2) per bit, +1 implicit
        min_normal = 2.0 ** (1 - fmt.bias)
        if fmt.has_inf:
            max_val = (2 - 2.0 ** -fmt.mant_bits) * 2.0 ** fmt.max_exp
        else:
            # e4m3fn: largest is S.1111.110 (all-ones mantissa is the NaN slot)
            max_val = (2 - 2 * 2.0 ** -fmt.mant_bits) * 2.0 ** ((1 << fmt.exp_bits) - 1 - fmt.bias)
        rows.append(dict(name=fmt.name, bits=fmt.total_bits, exp=fmt.exp_bits,
                         mant=fmt.mant_bits, eps=eps, digits=decimal_digits + 1,
                         min_normal=min_normal, max_val=max_val,
                         dynamic_range_decades=__import__("math").log10(max_val / min_normal)))
    if verbose:
        print(f"{'format':<10} {'bits':>4} {'e':>2} {'m':>2} {'eps (2^-m)':>11} "
              f"{'~dec digits':>11} {'min normal':>11} {'max normal':>11} {'range':>11}")
        print("-" * 82)
        for r in rows:
            print(f"{r['name']:<10} {r['bits']:>4} {r['exp']:>2} {r['mant']:>2} "
                  f"{r['eps']:>11.3e} {r['digits']:>11.1f} {r['min_normal']:>11.2e} "
                  f"{r['max_val']:>11.2e} {r['dynamic_range_decades']:>8.1f} dec")
    return rows


def accumulation_demo(n=1000, verbose=True):
    """Why exponent bits matter more than mantissa bits for training.

    Add 0.1 to itself n times in each format. The error that shows up is not the
    one-off rounding of a single 0.1 - it is that error compounding, which is
    exactly what gradient accumulation and optimizer state do all day.
    """
    rows = []
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16),
                        ("fp8 E4M3", torch.float8_e4m3fn)):
        x = torch.tensor([0.1], dtype=dtype)
        if dtype == torch.float8_e4m3fn:
            # no fp8 add kernel on CPU: emulate by round-tripping through fp32
            # after every add, which is what real fp8 training does anyway.
            acc = torch.tensor([0.0], dtype=torch.float32)
            for _ in range(n):
                acc = (acc + x.float()).to(torch.float8_e4m3fn).float()
            got = float(acc)
        else:
            acc = torch.zeros(1, dtype=dtype)
            for _ in range(n):
                acc = acc + x
            got = float(acc.float())
        exact = 0.1 * n
        rows.append(dict(name=name, got=got, exact=exact,
                         rel_err=abs(got - exact) / exact))
    if verbose:
        print(f"adding 0.1 to itself {n} times (exact answer: {0.1*n}):")
        print(f"{'format':<10} {'result':>16} {'relative error':>16}")
        print("-" * 46)
        for r in rows:
            print(f"{r['name']:<10} {r['got']:>16.6f} {r['rel_err']:>15.3%}")
    return rows
