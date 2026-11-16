---
title: Byte-pair tokens make a bigram much stronger, measured per character
tinypaper: 0.5
date: 2026-11-15
---

Cites: [char-bigram](char-bigram.md)

A tokeniser changes the unit a model predicts, so bits per token cannot be compared with bits
per character. This paper builds token data from the character data by code, keeps the number
of characters each token covers, and reports everything per character with the root paper's
metric.

The tokeniser is learned from the training split only. The test split is tokenised in two
pieces, cut at the root paper's `skip`, so token models are scored on exactly the characters
that character models are scored on. Validation and training are tokenised whole, so they
match only approximately.

The bridge: with zero merges the tokeniser changes nothing, and the bigram must score exactly
what the root paper recorded.

```cite
name: base
at: ./char-bigram.md
```

```data
name: tok0
load: tokenise
from: base.text
args: { merges: 0 }
```

```data
name: tok256
load: tokenise
from: base.text
args: { merges: 256 }
```

```python code=tokenise
import torch
from base import ngram, bpc, Text


def merge(x, a, b, new):
    # replace each adjacent pair (a, b) in x by the id `new`
    hit = (x[:-1] == a) & (x[1:] == b)
    if a == b:
        # in a run such as "aaa" only every other pair can merge, taken from the left
        keep, last = [], -2
        for i in hit.nonzero().flatten().tolist():
            if i > last + 1:
                keep.append(i)
                last = i
        hit = torch.zeros_like(hit)
        hit[keep] = True
    x = x.clone()
    x[:-1][hit] = new
    drop = torch.zeros(len(x), dtype=torch.bool)
    drop[1:] = hit
    return x[~drop]


def tokenise(d, merges):
    # Byte-pair encoding: `merges` times, give the most frequent adjacent pair in the
    # training split its own id. Same layout as the source data.
    vocab, V = list(d.vocab), len(d.vocab) + merges
    train, pairs = d.train, []
    for new in range(len(d.vocab), V):
        pair, n = torch.unique(train[:-1] * V + train[1:], return_counts=True)
        a, b = divmod(pair[n.argmax()].item(), V)
        pairs.append((a, b, new))
        vocab.append(vocab[a] + vocab[b])
        train = merge(train, a, b, new)

    def apply(ids):
        for a, b, new in pairs:
            ids = merge(ids, a, b, new)
        return ids

    head, rest = apply(d.test[:d.skip]), apply(d.test[d.skip:])
    return Text(train=train, val=apply(d.val), test=torch.cat([head, rest]), vocab=vocab,
                units=torch.tensor([len(v) for v in vocab]), skip=len(head))
```

```python experiments
call("bigram_tok0", ngram, d=tok0, n=2, env=base.cpu)
call("chars", ngram, d=tok0, n=2, alpha=0.03, env=base.cpu)
call("tokens", ngram, d=tok256, n=2, alpha=0.03, env=base.cpu)
```

```python claim=bridge stated=2026-11-15
def bridge(r):
    # zero merges: exactly the score the root paper reports for its bigram
    return r.bigram_tok0[0].out["test_bpc"] == r.base.bigram[0].out["test_bpc"]
```

```python claim=coverage stated=2026-11-15
def coverage(r):
    # a claim about data, with no run behind it: a test token covers over 1.8 characters on average
    ids = tok256.test[tok256.skip:]
    return tok256.units[ids].float().mean().item() > 1.8
```

```python claim=tokens_help stated=2026-11-15
def tokens_help(r):
    # a bigram over tokens beats a bigram over characters by over half a bit per character
    return bpc(r.chars) - bpc(r.tokens) > 0.5
```

```log
{"ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"call":"802d59e5a95ddb8a","fn":"fn:ngram","args":{"d":"data:bpe/tok0","n":2},"seconds":0.03,"out":{"test_bpc":3.5916237831115723,"val_bpc":3.5677802562713623,"train_bpc":3.551269769668579,"at_word_start":4.402455806732178,"elsewhere":3.404085636138916}}
{"call":"cb20930ba6c4642a","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:bpe/tok0","n":2},"seconds":0.03,"out":{"test_bpc":3.604349136352539,"val_bpc":3.5661234855651855,"train_bpc":3.5481133460998535,"at_word_start":4.402108192443848,"elsewhere":3.4198341369628906}}
{"call":"160c5a0ed4276080","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:bpe/tok256","n":2},"seconds":0.02,"out":{"test_bpc":2.896505355834961,"val_bpc":2.824864387512207,"train_bpc":2.6760613918304443,"at_word_start":5.905152797698975,"elsewhere":5.438847541809082}}
```
