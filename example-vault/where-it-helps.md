---
title: Neural models gain most over counts at the start of a word
tinypaper: 0.3
date: 2026-11-15
---

This paper runs nothing. Every cited paper logged the cost in bits of each test character,
on the same characters, so this paper can ask where the cost falls. Its claims need those
logs: with the papers alone they are open. It splits the scored
characters into those that start a word (they follow a space or a newline) and the rest.

```cite
name: base
at: ./char-bigram.md
```

```cite
name: ng
at: ./ngram.md
```

```cite
name: rnn
at: ./rnn.md
```

```cite
name: tr
at: ./transformer.md
```

```python code=analysis
import torch


def bits(runs):
    # per-character bits, averaged over seeds
    return torch.stack([x.out["test_bits"] for x in runs]).mean(0)


def starts(d):
    # for each scored test character: does it follow a space or a newline?
    gap = torch.tensor([c in " \n" for c in d.vocab])
    return gap[d.test[d.skip - 1:-1]]


def gain(counts, neural, where):
    return (bits(counts) - bits(neural))[where].mean().item()
```

```python claim=starts_are_hard stated=2026-11-15
def starts_are_hard(r):
    # for the 5-gram, a word's first character costs over 1.5 times a later character
    cost, first = bits(r.ng.sweep.where(n=5, alpha=0.03)), starts(base.text)
    return cost[first].mean() > 1.5 * cost[~first].mean()
```

```python claim=transformer_gain stated=2026-11-15 differs=fn
def transformer_gain(r):
    # the transformer's gain over the 5-gram is larger at word starts than elsewhere
    counts, first = r.ng.sweep.where(n=5, alpha=0.03), starts(base.text)
    return gain(counts, r.tr.transformer, first) > gain(counts, r.tr.transformer, ~first)
```

```python claim=gru_gain stated=2026-11-15 differs=fn
def gru_gain(r):
    # and the same holds for the GRU
    counts, first = r.ng.sweep.where(n=5, alpha=0.03), starts(base.text)
    return gain(counts, r.rnn.gru, first) > gain(counts, r.rnn.gru, ~first)
```
