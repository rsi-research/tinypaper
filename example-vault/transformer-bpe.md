---
title: On one megabyte of text, the transformer is worse on byte-pair tokens than on characters
tinypaper: 0.3
date: 2026-11-15
---

Cites: [char-bigram](char-bigram.md), [neural-bigram](neural-bigram.md), [transformer](transformer.md), [bpe](bpe.md)

The first version of this paper predicted that tokens would help. They did not. The training
split is half as many tokens as characters, the model passes over it twice as often in the
same number of steps, and it memorises it. The claim now states what was found.

This paper defines nothing. It runs the transformer paper's model on the tokeniser paper's
data and compares it, per character, with the transformer paper's recorded result. Its two
parents both build on the bigram paper, and the comparison is only valid if they used the same
version of it.

```cite
name: base
at: ./char-bigram.md
```

```cite
name: nb
at: ./neural-bigram.md
```

```cite
name: tr
at: ./transformer.md
```

```cite
name: bpe
at: ./bpe.md
```

```python code=use
from base import bpc
from nb import train
```

```python experiments
run("transformer_tok", train, d=bpe.tok256, make=tr.transformer, ctx=128, steps=3000, lr=1e-3, seeds=2, env=base.cpu)
```

```python claim=tokens_overfit stated=2026-11-15 differs=d
def tokens_overfit(r):
    # Same model, same steps, same window in ids. Per character, tokens are over 0.1 bits
    # worse on test and over 0.2 bits better on the training split: the token model overfits.
    def gap(split):
        return bpc(r.transformer_tok, split) - bpc(r.tr.transformer, split)
    return gap("test") > 0.1 and gap("train") < -0.2
```

```results
{"set": "a40-torch2.6", "ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"run":"ed66ffadea563fe6","experiment":"transformer_tok","fn":"fn:train","args":{"ctx":128,"d":"data:bpe/tok256","lr":0.001,"make":"fn:transformer","steps":3000},"seed":1,"code":"2ae0c91a64a96d42","seconds":47.72,"out":{"test_bpc":2.5327870845794678,"val_bpc":2.2769877910614014,"train_bpc":1.5601370334625244,"params":941377},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[8.583849476854159,8.446799254104,8.320369304009922,8.213620360549982,8.07445681498602,8.014498159450417,7.932055180071615,7.862021780848469,7.821940192168986,7.786533779870558]},"prefix_t":0.177}
{"run":"3cda0dde945f95ab","experiment":"transformer_tok","fn":"fn:train","args":{"ctx":128,"d":"data:bpe/tok256","lr":0.001,"make":"fn:transformer","steps":3000},"seed":2,"code":"2ae0c91a64a96d42","seconds":47.4,"out":{"test_bpc":2.4941468238830566,"val_bpc":2.2507729530334473,"train_bpc":1.5763384103775024,"params":941377},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[8.54763543402728,8.417935061844394,8.27394912275024,8.14291416519666,8.044423140700546,7.92354479057632,7.860938290148033,7.793071184395155,7.778823453667076,7.758801921384231]},"prefix_t":0.178}
```
