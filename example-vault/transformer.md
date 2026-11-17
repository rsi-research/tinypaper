---
title: A small transformer beats the recurrent network at equal training steps
tinypaper: 0.5
date: 2026-11-15
---

Claims: ✅ 1 of 1 hold

Cites: [char-bigram](char-bigram.md), [neural-bigram](neural-bigram.md), [rnn](rnn.md)

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

**Code `model`**

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

**Experiments**

```python experiments
call("transformer", train, d=base.text, make=transformer, ctx=128, steps=3000, lr=1e-3, seeds=2, env=base.cpu)
```

**Claim `beats_rnn`** (stated 2026-11-15): At 3000 steps the transformer is more than 0.05 bits per character better than the GRU.

```python claim=beats_rnn stated=2026-11-15
def beats_rnn(r):
    """At 3000 steps the transformer is more than 0.05 bits per character better than the GRU."""
    return bpc(r.rnn.gru) - bpc(r.transformer) > 0.05
```

**Log**

```log
{"ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"call":"468d334fefe32f4a","fn":"fn:train","args":{"ctx":128,"d":"data:text@f08b4da5","lr":0.001,"make":"fn:transformer","steps":3000},"seed":1,"seconds":43.89,"out":{"test_bpc":2.3091225624084473,"val_bpc":2.095316171646118,"train_bpc":1.852841854095459,"at_word_start":3.9557554721832275,"elsewhere":1.9282708168029785,"params":875585},"emitted":[{"step":0,"loss":6.312497996549563,"t":0.059},{"step":1,"loss":5.848710318176017,"t":0.069},{"step":2,"loss":5.476633420270968,"t":0.088},{"step":3,"loss":5.150007409405339,"t":0.106},{"step":4,"loss":4.926060544549634,"t":0.126},{"step":5,"loss":4.8014333166019005,"t":0.139},{"step":6,"loss":4.756172642374335,"t":0.157},{"step":7,"loss":4.593830994953594,"t":0.176},{"step":8,"loss":4.58846823193439,"t":0.193},{"step":9,"loss":4.503627470437258,"t":0.202},{"step":100,"loss":3.5708410393690215,"t":1.512},{"step":200,"loss":3.358423547461858,"t":2.919},{"step":300,"loss":3.0493782910980287,"t":4.354},{"step":400,"loss":2.8362996978809485,"t":5.89},{"step":500,"loss":2.6855809819708174,"t":7.365},{"step":600,"loss":2.5055798119941812,"t":8.763},{"step":700,"loss":2.4588395710212434,"t":10.168},{"step":800,"loss":2.348069329218827,"t":11.692},{"step":900,"loss":2.2790748771895433,"t":13.187},{"step":1000,"loss":2.2823301648050744,"t":14.573},{"step":1100,"loss":2.1406186962685814,"t":16.023},{"step":1200,"loss":2.1211779493977003,"t":17.472},{"step":1300,"loss":2.0861383760935626,"t":18.966},{"step":1400,"loss":2.1405671014733225,"t":20.411},{"step":1500,"loss":2.033181994187839,"t":21.825},{"step":1600,"loss":2.0846006792121985,"t":23.26},{"step":1700,"loss":1.965553428371708,"t":24.799},{"step":1800,"loss":1.920153104231642,"t":26.273},{"step":1900,"loss":1.9195793701083637,"t":27.658},{"step":2000,"loss":1.9457755955051357,"t":29.061},{"step":2100,"loss":1.9459988289859558,"t":30.585},{"step":2200,"loss":1.932452099542794,"t":32.065},{"step":2300,"loss":1.9173004280017805,"t":33.463},{"step":2400,"loss":1.869853510316447,"t":34.923},{"step":2500,"loss":1.8006609342733695,"t":36.394},{"step":2600,"loss":1.8694858114089026,"t":37.879},{"step":2700,"loss":1.8506752089707827,"t":39.326},{"step":2800,"loss":1.849080413849332,"t":40.763},{"step":2900,"loss":1.867089749117082,"t":42.157}]}
{"call":"c2a59b33dface350","fn":"fn:train","args":{"ctx":128,"d":"data:text@f08b4da5","lr":0.001,"make":"fn:transformer","steps":3000},"seed":2,"seconds":44.0,"out":{"test_bpc":2.3415350914001465,"val_bpc":2.1049256324768066,"train_bpc":1.888541340827942,"at_word_start":3.9620819091796875,"elsewhere":1.9667171239852905,"params":875585},"emitted":[{"step":0,"loss":6.271956182296371,"t":0.052},{"step":1,"loss":5.731367931353529,"t":0.07},{"step":2,"loss":5.299441758878826,"t":0.089},{"step":3,"loss":5.006905805101922,"t":0.105},{"step":4,"loss":4.8430799474042345,"t":0.114},{"step":5,"loss":4.697117239721064,"t":0.131},{"step":6,"loss":4.660617705693752,"t":0.149},{"step":7,"loss":4.676487576784768,"t":0.168},{"step":8,"loss":4.612495928086433,"t":0.182},{"step":9,"loss":4.5370760322382635,"t":0.192},{"step":100,"loss":3.55130656195342,"t":1.486},{"step":200,"loss":3.3226091923155834,"t":2.89},{"step":300,"loss":3.018635016395095,"t":4.359},{"step":400,"loss":2.806941399465413,"t":6.014},{"step":500,"loss":2.62468450504006,"t":7.525},{"step":600,"loss":2.4694218355114828,"t":8.952},{"step":700,"loss":2.4320260998905243,"t":10.328},{"step":800,"loss":2.319602072917412,"t":11.858},{"step":900,"loss":2.287293756091626,"t":13.338},{"step":1000,"loss":2.2225400522673175,"t":14.757},{"step":1100,"loss":2.2630184049223407,"t":16.215},{"step":1200,"loss":2.1548672869099144,"t":17.584},{"step":1300,"loss":2.072656484109776,"t":19.109},{"step":1400,"loss":2.0805025046247887,"t":20.593},{"step":1500,"loss":2.0725883789800346,"t":21.983},{"step":1600,"loss":2.044595278847048,"t":23.366},{"step":1700,"loss":2.0471419979410244,"t":24.895},{"step":1800,"loss":2.04442226430028,"t":26.37},{"step":1900,"loss":2.036446568866517,"t":27.771},{"step":2000,"loss":1.9816351820885874,"t":29.215},{"step":2100,"loss":2.0034326072242417,"t":30.686},{"step":2200,"loss":1.8960777688853099,"t":32.155},{"step":2300,"loss":1.9469067253998604,"t":33.608},{"step":2400,"loss":1.936130120514146,"t":34.946},{"step":2500,"loss":1.9114752036344063,"t":36.435},{"step":2600,"loss":2.0123987507443233,"t":37.94},{"step":2700,"loss":1.9008420322795114,"t":39.418},{"step":2800,"loss":1.927479049210445,"t":40.824},{"step":2900,"loss":1.8844648123684498,"t":42.315}]}
```
