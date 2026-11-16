---
title: Neural models gain most over counts at the start of a word
tinypaper: 0.5
date: 2026-11-15
---

Claims: ✅ 3 of 3 hold

Cites: [char-bigram](char-bigram.md), [ngram](ngram.md), [rnn](rnn.md), [transformer](transformer.md)

This paper runs nothing. Every cited paper logged its cost at the start of a word and
elsewhere, on the same characters, so this paper can ask where the cost falls. It splits the scored
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

**Code `analysis`**

```python code=analysis
from statistics import mean


def at(runs, where):
    # the logged cost at word starts ("at_word_start") or elsewhere ("elsewhere"), averaged over seeds
    return mean(x.out[where] for x in runs)


def gain(counts, neural, where):
    return at(counts, where) - at(neural, where)
```

**Claim `starts_are_hard`** (stated 2026-11-15): For the 5-gram, a word's first character costs over 1.5 times a later character.

```python claim=starts_are_hard stated=2026-11-15
def starts_are_hard(r):
    """For the 5-gram, a word's first character costs over 1.5 times a later character."""
    best = r.ng.sweep.where(n=5, alpha=0.03)
    return at(best, "at_word_start") > 1.5 * at(best, "elsewhere")
```

**Claim `transformer_gain`** (stated 2026-11-15): The transformer's gain over the 5-gram is larger at word starts than elsewhere.

```python claim=transformer_gain stated=2026-11-15
def transformer_gain(r):
    """The transformer's gain over the 5-gram is larger at word starts than elsewhere."""
    counts = r.ng.sweep.where(n=5, alpha=0.03)
    return gain(counts, r.tr.transformer, "at_word_start") > gain(counts, r.tr.transformer, "elsewhere")
```

**Claim `gru_gain`** (stated 2026-11-15): The GRU's gain over the 5-gram is also larger at word starts than elsewhere.

```python claim=gru_gain stated=2026-11-15
def gru_gain(r):
    """The GRU's gain over the 5-gram is also larger at word starts than elsewhere."""
    counts = r.ng.sweep.where(n=5, alpha=0.03)
    return gain(counts, r.rnn.gru, "at_word_start") > gain(counts, r.rnn.gru, "elsewhere")
```
