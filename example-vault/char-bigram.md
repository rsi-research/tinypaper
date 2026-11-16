---
title: A character bigram predicts Shakespeare better than character frequencies
tinypaper: 0.3
date: 2026-11-15
---

This is the root of the language papers. It fixes the data, its layout (the `Text` dataclass),
the split, what an experiment returns, and the metric. Its model has no training: it counts.

An experiment returns two things. The bits per character on each split are small, so they are
reported in the paper. The cost in bits of every scored test character is large, so it goes
to the logs, where later papers can ask new questions of it.

Every paper that builds on this one scores the same characters: each split from position
`skip` onward. The first `skip` characters are context only. So any two models can be compared
character by character. The validation split is for choosing between settings; claims are
about the test split.

```env
name: cpu
hardware: 2 CPU, 8 GB
packages: [ "torch==2.11.0" ]
```

```data
name: text
load: load
files:
  text:
    url: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    sha256: 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
```

```python code=load
from dataclasses import dataclass
import torch


@dataclass
class Text:
    # One stream of ids, cut into three splits in order.
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    vocab: list           # the string each id stands for
    units: torch.Tensor   # how many characters each id covers: all ones here, more for a tokeniser
    skip: int             # ids at the start of a split that are context only and never scored


def load(files):
    text = open(files["text"]).read()
    vocab = sorted(set(text))
    index = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([index[c] for c in text])
    a, b = int(0.9 * len(ids)), int(0.95 * len(ids))
    return Text(train=ids[:a], val=ids[a:b], test=ids[b:], vocab=vocab,
                units=torch.ones(len(vocab), dtype=torch.long), skip=256)
```

```python code=count
def part(d, name):
    # the ids a split is scored on; "train" means a slice of it as long as the test split
    return d.train[:len(d.test)] if name == "train" else getattr(d, name)


def keys(ids, n, V):
    # one integer per position i >= n - 1: the n ids ending at i
    k = torch.zeros(len(ids) - n + 1, dtype=torch.long)
    for j in range(n):
        k = k * V + ids[j:len(ids) - n + 1 + j]
    return k


def ngram(d, n, alpha=1.0):
    # Count model over the n - 1 previous ids, with add-alpha smoothing. n = 1 is frequencies.
    V = len(d.vocab)
    train = keys(d.train, n, V)
    grams, gram_n = torch.unique(train, return_counts=True)
    ctxs, ctx_n = torch.unique(train // V, return_counts=True)

    def count(table, counts, q):
        i = torch.searchsorted(table, q).clamp(max=len(table) - 1)
        return torch.where(table[i] == q, counts[i], 0)

    def bits(ids):
        q = keys(ids, n, V)[d.skip - n + 1:]
        p = (count(grams, gram_n, q) + alpha) / (count(ctxs, ctx_n, q // V) + alpha * V)
        return -torch.log2(p)

    cost = {name: bits(part(d, name)) for name in ["test", "val", "train"]}
    return dict({name + "_bpc": per_char(b, d, name) for name, b in cost.items()}, test_bits=cost["test"])
```

```python code=metric
from statistics import mean


def per_char(bits, d, split):
    # bits per character: total bits over the scored ids, divided by the characters they cover
    return (bits.sum() / d.units[part(d, split)[d.skip:]].sum()).item()


def bpc(runs, split="test"):
    # the reported bits per character, averaged over runs
    return mean(x.out[split + "_bpc"] for x in runs)
```

```python experiments
run("unigram", ngram, d=text, n=1, env=cpu)
run("bigram", ngram, d=text, n=2, env=cpu)
```

```python claim=c1 stated=2026-11-15 differs=n
def c1(r):
    # knowing the previous character saves more than one bit per character
    return bpc(r.unigram) - bpc(r.bigram) > 1.0
```

```python claim=c2 stated=2026-11-15
def c2(r):
    # the bigram costs under 3.7 bits per character
    return bpc(r.bigram) < 3.7
```

```results
{"set": "a40-torch2.6", "ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"run":"cc113b7448b26487","experiment":"unigram","fn":"fn:ngram","args":{"d":"data:char-bigram/text","n":1},"seed":null,"code":"b665326ec5abf135","seconds":0.05,"out":{"test_bpc":4.850418567657471,"val_bpc":4.806154727935791,"train_bpc":4.73865270614624},"logged":["test_bits"]}
{"run":"7129f2ce2fdd90b8","experiment":"bigram","fn":"fn:ngram","args":{"d":"data:char-bigram/text","n":2},"seed":null,"code":"b665326ec5abf135","seconds":0.03,"out":{"test_bpc":3.5916237831115723,"val_bpc":3.5677802562713623,"train_bpc":3.551269769668579},"logged":["test_bits"]}
```
