---
title: A small transformer beats the recurrent network at equal training steps
tinypaper: 0.3
date: 2026-11-15
---

A four-layer transformer with causal attention, trained by the shared loop for the same number
of steps as the GRU of the recurrent paper. The block takes its normalisation and its skip
connections as arguments, so later papers can vary them without copying it.

```cite
name: base
at: ./char-bigram.md
```

```cite
name: nb
at: ./neural-bigram.md
```

```cite
name: rnn
at: ./rnn.md
```

```python code=model
import torch
from base import bpc
from nb import train


class Block(torch.nn.Module):
    # attention then an MLP, each normalised first, each with a skip connection around it
    def __init__(self, width, heads=4, norm=torch.nn.LayerNorm, skip=True):
        super().__init__()
        self.heads, self.skip = heads, skip
        self.norm1, self.norm2 = norm(width), norm(width)
        self.qkv = torch.nn.Linear(width, 3 * width)
        self.proj = torch.nn.Linear(width, width)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width), torch.nn.GELU(), torch.nn.Linear(4 * width, width))

    def attend(self, x):
        B, T, W = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, W // self.heads).permute(2, 0, 3, 1, 4)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, W))

    def forward(self, x):
        a = self.attend(self.norm1(x))
        x = x + a if self.skip else a
        m = self.mlp(self.norm2(x))
        return x + m if self.skip else m


class Transformer(torch.nn.Module):
    def __init__(self, vocab, width=128, depth=4, positions=512, **block):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, width)
        self.where = torch.nn.Embedding(positions, width)
        self.blocks = torch.nn.Sequential(*[Block(width, **block) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(width)
        self.out = torch.nn.Linear(width, vocab)

    def forward(self, x):
        h = self.embed(x) + self.where(torch.arange(x.shape[1], device=x.device))
        return self.out(self.norm(self.blocks(h)))


def transformer(vocab, **kw):
    return Transformer(vocab, **kw)
```

```python experiments
run("transformer", train, d=base.text, make=transformer, ctx=128, steps=3000, lr=1e-3, seeds=2, env=base.cpu)
```

```python claim=beats_rnn stated=2026-11-15 differs=make,lr
def beats_rnn(r):
    # at 3000 steps the transformer is more than 0.05 bits per character better than the GRU
    return bpc(r.rnn.gru) - bpc(r.transformer) > 0.05
```

```results
{"set": "a40-torch2.6", "ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"run":"d040e7aa2baafe84","experiment":"transformer","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.001,"make":"fn:transformer","steps":3000},"seed":1,"code":"7d6c4eb3d5461258","seconds":43.22,"out":{"test_bpc":2.3091225624084473,"val_bpc":2.095316171646118,"train_bpc":1.852841854095459,"params":875585},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.312497996549563,5.848710318176017,5.476633420270968,5.150007409405339,4.926060544549634,4.8014333166019005,4.756172642374335,4.593830994953594,4.58846823193439,4.503627470437258]},"prefix_t":0.188}
{"run":"58044ec27e7560e4","experiment":"transformer","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.001,"make":"fn:transformer","steps":3000},"seed":2,"code":"7d6c4eb3d5461258","seconds":43.39,"out":{"test_bpc":2.3415350914001465,"val_bpc":2.1049256324768066,"train_bpc":1.888541340827942,"params":875585},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.271956182296371,5.731367931353529,5.299441758878826,5.006905805101922,4.8430799474042345,4.697117239721064,4.660617705693752,4.676487576784768,4.612495928086433,4.5370760322382635]},"prefix_t":0.173}
```
