---
title: Longer count models help up to five characters, then overfit
tinypaper: 0.5
date: 2026-11-15
---

Cites: [char-bigram](char-bigram.md)

The bigram paper's count model takes any order `n`. This paper sweeps it, with two strengths of
smoothing. Nothing new is defined except the claims.

```cite
name: base
at: ./char-bigram.md
```

**Code `use`**

```python code=use
from base import ngram, bpc
```

**Experiments**

```python experiments
for n in [3, 4, 5, 6, 7]:
    for alpha in [1.0, 0.03]:
        call("sweep", ngram, d=base.text, n=n, alpha=alpha, env=base.cpu)
```

**Claim `longer_helps`** (stated 2026-11-15): The 5-gram with the smoothing that is best on validation beats the bigram by over 0.7 bits.

```python claim=longer_helps stated=2026-11-15
def longer_helps(r):
    """The 5-gram with the smoothing that is best on validation beats the bigram by over 0.7 bits."""
    alpha = min([1.0, 0.03], key=lambda a: bpc(r.sweep.where(n=5, alpha=a), "val"))
    return bpc(r.base.bigram) - bpc(r.sweep.where(n=5, alpha=alpha)) > 0.7
```

**Claim `then_overfits`** (stated 2026-11-15): With light smoothing, n = 7 is worse on test than n = 5 while still better on train.

```python claim=then_overfits stated=2026-11-15
def then_overfits(r):
    """With light smoothing, n = 7 is worse on test than n = 5 while still better on train."""
    def at(n, split):
        return bpc(r.sweep.where(n=n, alpha=0.03), split)
    return at(7, "test") > at(5, "test") and at(7, "train") < at(5, "train")
```

**Claim `smoothing_matters`** (stated 2026-11-15): At n = 5, alpha = 0.03 beats alpha = 1 by more than 0.2 bits.

```python claim=smoothing_matters stated=2026-11-15
def smoothing_matters(r):
    """At n = 5, alpha = 0.03 beats alpha = 1 by more than 0.2 bits."""
    return bpc(r.sweep.where(n=5, alpha=1.0)) - bpc(r.sweep.where(n=5, alpha=0.03)) > 0.2
```

**Log**

```log
{"ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"call":"8ec9b130e36168f9","fn":"fn:ngram","args":{"alpha":1.0,"d":"data:char-bigram/text","n":3},"seconds":0.04,"out":{"test_bpc":3.02884578704834,"val_bpc":2.939361095428467,"train_bpc":2.8658430576324463,"at_word_start":4.134876728057861,"elsewhere":2.773030996322632}}
{"call":"2e67aaa31cb11ee1","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:char-bigram/text","n":3},"seconds":0.04,"out":{"test_bpc":3.0444157123565674,"val_bpc":2.8840770721435547,"train_bpc":2.80180287361145,"at_word_start":4.130579471588135,"elsewhere":2.7931952476501465}}
{"call":"6b018d01f0030180","fn":"fn:ngram","args":{"alpha":1.0,"d":"data:char-bigram/text","n":4},"seconds":0.04,"out":{"test_bpc":2.8757052421569824,"val_bpc":2.7587738037109375,"train_bpc":2.5582144260406494,"at_word_start":4.169795036315918,"elsewhere":2.5763936042785645}}
{"call":"96c6bdfadfcd0e65","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:char-bigram/text","n":4},"seconds":0.04,"out":{"test_bpc":2.660233736038208,"val_bpc":2.4759681224823,"train_bpc":2.2095258235931396,"at_word_start":4.138306617736816,"elsewhere":2.3183679580688477}}
{"call":"21fef917f35b98ed","fn":"fn:ngram","args":{"alpha":1.0,"d":"data:char-bigram/text","n":5},"seconds":0.04,"out":{"test_bpc":3.2100939750671387,"val_bpc":3.0628271102905273,"train_bpc":2.7830042839050293,"at_word_start":4.414824962615967,"elsewhere":2.931450128555298}}
{"call":"3b965e6f2357439e","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:char-bigram/text","n":5},"seconds":0.04,"out":{"test_bpc":2.6302366256713867,"val_bpc":2.4227774143218994,"train_bpc":1.8935390710830688,"at_word_start":4.316840648651123,"elsewhere":2.2401397228240967}}
{"call":"7d4bb103ba10a779","fn":"fn:ngram","args":{"alpha":1.0,"d":"data:char-bigram/text","n":6},"seconds":0.04,"out":{"test_bpc":3.825535297393799,"val_bpc":3.6579229831695557,"train_bpc":3.2515029907226562,"at_word_start":4.784564971923828,"elsewhere":3.60371994972229}}
{"call":"6be5b35709373827","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:char-bigram/text","n":6},"seconds":0.04,"out":{"test_bpc":2.9795355796813965,"val_bpc":2.7330482006073,"train_bpc":1.7073183059692383,"at_word_start":4.674688816070557,"elsewhere":2.587461233139038}}
{"call":"8b6b41986189c2ad","fn":"fn:ngram","args":{"alpha":1.0,"d":"data:char-bigram/text","n":7},"seconds":0.04,"out":{"test_bpc":4.450703144073486,"val_bpc":4.283419609069824,"train_bpc":3.7376508712768555,"at_word_start":5.1291937828063965,"elsewhere":4.293774604797363}}
{"call":"c84ea811e108e632","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:char-bigram/text","n":7},"seconds":0.04,"out":{"test_bpc":3.514645576477051,"val_bpc":3.244791269302368,"train_bpc":1.5965131521224976,"at_word_start":5.044610023498535,"elsewhere":3.160778045654297}}
```
