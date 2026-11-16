---
title: A recurrent network beats a fixed window of characters
tinypaper: 0.3
date: 2026-11-15
---

A GRU reads the whole window instead of the last eight characters. Everything else is cited.

```cite
name: base
at: ./char-bigram.md
```

```cite
name: nb
at: ./neural-bigram.md
```

```cite
name: mlp
at: ./mlp-lm.md
```

```python code=model
import torch
from base import bpc
from nb import train


class GRU(torch.nn.Module):
    def __init__(self, vocab, width=128, layers=1):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, width)
        self.rnn = torch.nn.GRU(width, width, layers, batch_first=True)
        self.out = torch.nn.Linear(width, vocab)

    def forward(self, x):
        return self.out(self.rnn(self.embed(x))[0])


def gru(vocab, **kw):
    return GRU(vocab, **kw)
```

```python experiments
run("gru", train, d=base.text, make=gru, ctx=128, steps=3000, seeds=2, env=base.cpu)
```

```python claim=beats_window stated=2026-11-15 differs=make,k,ctx
def beats_window(r):
    # the GRU beats the 8-character MLP by more than 0.1 bits per character
    return bpc(r.mlp.mlp.where(k=8)) - bpc(r.gru) > 0.1
```

```results
{"set": "a40-torch2.6", "ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"run":"e76dece631e9a16e","experiment":"gru","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","make":"fn:gru","steps":3000},"seed":1,"code":"ba7b4d0f3222dc32","seconds":11.99,"out":{"test_bpc":2.391141176223755,"val_bpc":2.2012479305267334,"train_bpc":2.0160043239593506,"params":115777},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.035894171583287,5.772923067385604,5.498965711450805,5.168386507372444,4.852627048318931,4.596697601778175,4.5469402691611505,4.45621770100451,4.402875905978203,4.2882484372822365]},"prefix_t":0.117}
{"run":"0d84cebc4dbd9e0f","experiment":"gru","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","make":"fn:gru","steps":3000},"seed":2,"code":"ba7b4d0f3222dc32","seconds":11.72,"out":{"test_bpc":2.3624839782714844,"val_bpc":2.1879334449768066,"train_bpc":2.01430082321167,"params":115777},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.06049181824029,5.809864940790936,5.575107246501995,5.280389520816245,4.970083631621591,4.724677459521828,4.495515048796059,4.45624556219395,4.476378883199856,4.358437308821762]},"prefix_t":0.054}
```
