"""Corpus + batching.

Two batchers, on purpose:

  * `fixed_batch`   - every row is block_size long. This is the easy world where
                      "mean of the means" happens to be right, because every
                      micro-batch has an identical token count.
  * `varlen_batch`  - rows have different lengths, padded, with targets masked to
                      -100 on the padding. This is the real world, and the world
                      where averaging averages is wrong.

The whole gradient-accumulation experiment hinges on that second one.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
DATA_PATH = os.path.join(DATA_DIR, "input.txt")

IGNORE_INDEX = -100


def _synthetic_corpus(n_chars: int = 400_000) -> str:
    """Deterministic offline fallback: a tiny formal language with real structure.

    Not as interesting as Shakespeare, but it is learnable, reproducible without
    a network, and has the same char-level shape. Only used if the download fails.
    """
    g = torch.Generator().manual_seed(1234)
    subjects = ["the king", "a servant", "my lord", "the queen", "this night", "our army"]
    verbs = ["speaks", "waits", "falls", "rides", "answers", "remembers"]
    objects = ["in silence", "at the gate", "before dawn", "without mercy", "for the crown"]
    out = []
    total = 0
    while total < n_chars:
        i = int(torch.randint(len(subjects), (1,), generator=g))
        j = int(torch.randint(len(verbs), (1,), generator=g))
        k = int(torch.randint(len(objects), (1,), generator=g))
        line = f"{subjects[i]} {verbs[j]} {objects[k]}.\n"
        out.append(line)
        total += len(line)
    return "".join(out)


def load_text() -> tuple[str, str]:
    """Returns (text, provenance). Downloads once, then caches on disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return f.read(), "cached data/input.txt"
    try:
        with urllib.request.urlopen(DATA_URL, timeout=30) as r:
            text = r.read().decode("utf-8")
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        return text, f"downloaded {DATA_URL}"
    except Exception as e:  # offline: fall back to something reproducible
        text = _synthetic_corpus()
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        return text, f"synthetic fallback (download failed: {type(e).__name__})"


class CharDataset:
    def __init__(self, text: str, provenance: str = ""):
        self.text = text
        self.provenance = provenance
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(self.data))
        self.train, self.val = self.data[:n], self.data[n:]

    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    # ---------------- fixed-length ----------------

    def fixed_batch(self, batch_size: int, block_size: int, split="train", generator=None):
        """(B, T) inputs and (B, T) next-token targets. Every row is full length."""
        src = self.train if split == "train" else self.val
        ix = torch.randint(len(src) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([src[i:i + block_size] for i in ix])
        y = torch.stack([src[i + 1:i + 1 + block_size] for i in ix])
        return x, y

    # ---------------- variable-length ----------------

    def varlen_batch(self, lengths, split="train", generator=None):
        """Rows of differing true length, right-padded.

        Returns (x, y, pad_mask, n_real_tokens):
          x         (B, Tmax) int64, padded with token 0
          y         (B, Tmax) int64, padded with IGNORE_INDEX so padding never
                    contributes to the loss
          pad_mask  (B, Tmax) bool, True where the position is a real token
          n_real_tokens  int, sum(lengths) - the denominator the loss *should* use
        """
        src = self.train if split == "train" else self.val
        B, Tmax = len(lengths), max(lengths)
        x = torch.zeros(B, Tmax, dtype=torch.long)
        y = torch.full((B, Tmax), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(B, Tmax, dtype=torch.bool)
        for b, L in enumerate(lengths):
            i = int(torch.randint(len(src) - L - 1, (1,), generator=generator))
            x[b, :L] = src[i:i + L]
            y[b, :L] = src[i + 1:i + 1 + L]
            mask[b, :L] = True
        return x, y, mask, int(sum(lengths))


def skewed_lengths(n_micro: int, batch_size: int, lo: int, hi: int, generator=None):
    """One length-list per micro-batch, deliberately uneven across micro-batches.

    Real corpora do this to you for free: some documents are long, some are two
    lines. Here it is explicit so the effect is reproducible. Micro-batch m gets
    lengths concentrated near a different part of [lo, hi], so the per-micro-batch
    token counts differ by an order of magnitude.
    """
    out = []
    for m in range(n_micro):
        # sweep the centre of the length distribution across micro-batches
        frac = (m + 0.5) / n_micro
        centre = lo + frac * (hi - lo)
        spread = max(2.0, 0.25 * (hi - lo))
        L = torch.normal(
            mean=torch.full((batch_size,), float(centre)),
            std=torch.full((batch_size,), float(spread)),
            generator=generator,
        )
        L = L.clamp(lo, hi).round().to(torch.long).tolist()
        out.append([int(v) for v in L])
    return out
