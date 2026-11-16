---
title: Longer count models help up to five characters, then overfit
tinypaper: 0.3
date: 2026-11-15
---

The bigram paper's count model takes any order `n`. This paper sweeps it, with two strengths of
smoothing. Nothing new is defined except the claims.

```cite
name: base
at: ./char-bigram.md
```

```python code=use
from base import ngram, bpc
```

```python experiments
for n in [3, 4, 5, 6, 7]:
    for alpha in [1.0, 0.03]:
        run("sweep", ngram, d=base.text, n=n, alpha=alpha, env=base.cpu)
```

```python claim=longer_helps stated=2026-11-15 differs=n,alpha
def longer_helps(r):
    # the 5-gram with the smoothing that is best on validation beats the bigram by over 0.7 bits
    alpha = min([1.0, 0.03], key=lambda a: bpc(r.sweep.where(n=5, alpha=a), "val"))
    return bpc(r.base.bigram) - bpc(r.sweep.where(n=5, alpha=alpha)) > 0.7
```

```python claim=then_overfits stated=2026-11-15 differs=n
def then_overfits(r):
    # with light smoothing, n = 7 is worse on test than n = 5 while still better on train
    def at(n, split):
        return bpc(r.sweep.where(n=n, alpha=0.03), split)
    return at(7, "test") > at(5, "test") and at(7, "train") < at(5, "train")
```

```python claim=smoothing_matters stated=2026-11-15 differs=alpha
def smoothing_matters(r):
    # at n = 5, alpha = 0.03 beats alpha = 1 by more than 0.2 bits
    return bpc(r.sweep.where(n=5, alpha=1.0)) - bpc(r.sweep.where(n=5, alpha=0.03)) > 0.2
```

```results
{"set": "a40-torch2.6", "ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"run":"efcfafbd8a81deb6","experiment":"sweep","fn":"fn:ngram","args":{"alpha":1.0,"d":"data:char-bigram/text","n":3},"seed":null,"code":"b9820fe2e77c58ea","seconds":0.04,"out":{"test_bpc":3.02884578704834,"val_bpc":2.939361095428467,"train_bpc":2.8658430576324463},"logged":["test_bits"]}
{"run":"8df5f64b21de3646","experiment":"sweep","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:char-bigram/text","n":3},"seed":null,"code":"b9820fe2e77c58ea","seconds":0.04,"out":{"test_bpc":3.0444157123565674,"val_bpc":2.8840770721435547,"train_bpc":2.80180287361145},"logged":["test_bits"]}
{"run":"705b7d09c3749c09","experiment":"sweep","fn":"fn:ngram","args":{"alpha":1.0,"d":"data:char-bigram/text","n":4},"seed":null,"code":"b9820fe2e77c58ea","seconds":0.04,"out":{"test_bpc":2.8757052421569824,"val_bpc":2.7587738037109375,"train_bpc":2.5582144260406494},"logged":["test_bits"]}
{"run":"68bb2bc492685093","experiment":"sweep","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:char-bigram/text","n":4},"seed":null,"code":"b9820fe2e77c58ea","seconds":0.04,"out":{"test_bpc":2.660233736038208,"val_bpc":2.4759681224823,"train_bpc":2.2095258235931396},"logged":["test_bits"]}
{"run":"80c1fbaf09df9ed7","experiment":"sweep","fn":"fn:ngram","args":{"alpha":1.0,"d":"data:char-bigram/text","n":5},"seed":null,"code":"b9820fe2e77c58ea","seconds":0.04,"out":{"test_bpc":3.2100939750671387,"val_bpc":3.0628271102905273,"train_bpc":2.7830042839050293},"logged":["test_bits"]}
{"run":"d84ec5112b40bb45","experiment":"sweep","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:char-bigram/text","n":5},"seed":null,"code":"b9820fe2e77c58ea","seconds":0.04,"out":{"test_bpc":2.6302366256713867,"val_bpc":2.4227774143218994,"train_bpc":1.8935390710830688},"logged":["test_bits"]}
{"run":"bb908f9f6279d0c3","experiment":"sweep","fn":"fn:ngram","args":{"alpha":1.0,"d":"data:char-bigram/text","n":6},"seed":null,"code":"b9820fe2e77c58ea","seconds":0.04,"out":{"test_bpc":3.825535297393799,"val_bpc":3.6579229831695557,"train_bpc":3.2515029907226562},"logged":["test_bits"]}
{"run":"0dbe7fd6d8738007","experiment":"sweep","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:char-bigram/text","n":6},"seed":null,"code":"b9820fe2e77c58ea","seconds":0.04,"out":{"test_bpc":2.9795355796813965,"val_bpc":2.7330482006073,"train_bpc":1.7073183059692383},"logged":["test_bits"]}
{"run":"d115465a90152970","experiment":"sweep","fn":"fn:ngram","args":{"alpha":1.0,"d":"data:char-bigram/text","n":7},"seed":null,"code":"b9820fe2e77c58ea","seconds":0.04,"out":{"test_bpc":4.450703144073486,"val_bpc":4.283419609069824,"train_bpc":3.7376508712768555},"logged":["test_bits"]}
{"run":"a32bc1c3008bc026","experiment":"sweep","fn":"fn:ngram","args":{"alpha":0.03,"d":"data:char-bigram/text","n":7},"seed":null,"code":"b9820fe2e77c58ea","seconds":0.04,"out":{"test_bpc":3.514645576477051,"val_bpc":3.244791269302368,"train_bpc":1.5965131521224976},"logged":["test_bits"]}
```
