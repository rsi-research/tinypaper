---
title: An MLP over the last eight characters beats every count model
tinypaper: 0.3
date: 2026-11-15
---

The first version of this paper claimed that every doubling of the window helps. On the
records it did not: a window of 8 fits the training split better than a window of 4 and is no
better on test. The claim now says that.

The model embeds the last `k` characters, concatenates them, and passes them through one
hidden layer (Bengio et al., 2003). The loop, data and metric come from earlier papers. The
count models it is compared with are read from the n-gram paper's records, not rerun.

```cite
name: base
at: ./char-bigram.md
```

```cite
name: nb
at: ./neural-bigram.md
```

```cite
name: ng
at: ./ngram.md
```

```python code=model
import torch
from base import bpc
from nb import train


class Window(torch.nn.Module):
    def __init__(self, vocab, k, width=32, hidden=256):
        super().__init__()
        self.k = k
        self.embed = torch.nn.Embedding(vocab, width)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(k * width, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, vocab))

    def forward(self, x):
        e = self.embed(x)
        T = e.shape[1]
        # position t sees ids t, t - 1, ..., t - k + 1; earlier than the window start is zeros
        past = [torch.nn.functional.pad(e, (0, 0, j, 0))[:, :T] for j in range(self.k)]
        return self.net(torch.cat(past, -1))


def window(vocab, **kw):
    return Window(vocab, **kw)
```

```python experiments
for k in [1, 2, 4, 8]:
    run("mlp", train, d=base.text, make=window, k=k, steps=3000, seeds=2, env=base.cpu)
```

```python claim=context_helps stated=2026-11-15 differs=k
def context_helps(r):
    # each doubling of the window helps up to 4 characters; from 4 to 8 the test cost moves
    # by under 0.03 bits although the training cost keeps falling
    test = [bpc(r.mlp.where(k=k)) for k in [1, 2, 4, 8]]
    train = [bpc(r.mlp.where(k=k), "train") for k in [4, 8]]
    return test[0] > test[1] > test[2] and abs(test[3] - test[2]) < 0.03 and train[1] < train[0]
```

```python claim=beats_counts stated=2026-11-15 differs=fn
def beats_counts(r):
    # the 8-character MLP beats the best count model of any order in the n-gram paper
    best = min(bpc(r.ng.sweep.where(n=n, alpha=a)) for n in [3, 4, 5, 6, 7] for a in [1.0, 0.03])
    return bpc(r.mlp.where(k=8)) < best
```

```results
{"set": "a40-torch2.6", "ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"run":"4f712b964a84430e","experiment":"mlp","fn":"fn:train","args":{"d":"data:char-bigram/text","k":1,"make":"fn:window","steps":3000},"seed":1,"code":"823985e244f4b41d","seconds":9.31,"out":{"test_bpc":3.6072235107421875,"val_bpc":3.5860986709594727,"train_bpc":3.5595710277557373,"params":27233},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.135145359396214,5.874491205470955,5.638577787768217,5.376655433775495,5.201247576438103,4.987693623174038,4.804230442435533,4.666400450346362,4.488888901223617,4.4345227775287706]},"prefix_t":0.114}
{"run":"f11115ca316a1924","experiment":"mlp","fn":"fn:train","args":{"d":"data:char-bigram/text","k":1,"make":"fn:window","steps":3000},"seed":2,"code":"823985e244f4b41d","seconds":10.03,"out":{"test_bpc":3.6102118492126465,"val_bpc":3.5841290950775146,"train_bpc":3.575190544128418,"params":27233},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.074941112635179,5.823536873603926,5.593207044644099,5.410624759043314,5.1856993129042,5.052920107305568,4.913613816141392,4.7175921182715825,4.630370772921208,4.543692892747559]},"prefix_t":0.032}
{"run":"be7dfa7f98ec1170","experiment":"mlp","fn":"fn:train","args":{"d":"data:char-bigram/text","k":2,"make":"fn:window","steps":3000},"seed":1,"code":"823985e244f4b41d","seconds":9.84,"out":{"test_bpc":3.0329020023345947,"val_bpc":2.9086499214172363,"train_bpc":2.845823287963867,"params":35425},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.145696151061346,5.843262595727286,5.603619218292635,5.317802970650241,5.115327795789457,4.929313080442752,4.719571638583013,4.504975126489418,4.398175276164821,4.321203752840567]},"prefix_t":0.039}
{"run":"479cc3c6a8cdd8ca","experiment":"mlp","fn":"fn:train","args":{"d":"data:char-bigram/text","k":2,"make":"fn:window","steps":3000},"seed":2,"code":"823985e244f4b41d","seconds":8.92,"out":{"test_bpc":3.0358917713165283,"val_bpc":2.9212372303009033,"train_bpc":2.846074104309082,"params":35425},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.097906987900795,5.806284949930576,5.553246875716125,5.3084705040838225,5.088323423916281,4.853627299416349,4.671352174830004,4.531337315144273,4.434229031161097,4.280045208801382]},"prefix_t":0.04}
{"run":"c99f8a2538bef39e","experiment":"mlp","fn":"fn:train","args":{"d":"data:char-bigram/text","k":4,"make":"fn:window","steps":3000},"seed":1,"code":"823985e244f4b41d","seconds":11.11,"out":{"test_bpc":2.5458426475524902,"val_bpc":2.376831293106079,"train_bpc":2.2124571800231934,"params":51809},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.082019918544693,5.782917323192544,5.470441700874606,5.1879553813182175,4.8745837293892835,4.683599059397945,4.4651611428146785,4.37563901356106,4.225245688826361,4.206168685262056]},"prefix_t":0.021}
{"run":"92c48fcd35147cba","experiment":"mlp","fn":"fn:train","args":{"d":"data:char-bigram/text","k":4,"make":"fn:window","steps":3000},"seed":2,"code":"823985e244f4b41d","seconds":11.17,"out":{"test_bpc":2.5471928119659424,"val_bpc":2.3853490352630615,"train_bpc":2.2271203994750977,"params":51809},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.1100991819857615,5.788590686879206,5.4725137478522,5.157056634298904,4.877350930241667,4.72716123296559,4.4881817085632685,4.397974744393915,4.331587721334363,4.213502369460148]},"prefix_t":0.049}
{"run":"aa56f0d9e24b583a","experiment":"mlp","fn":"fn:train","args":{"d":"data:char-bigram/text","k":8,"make":"fn:window","steps":3000},"seed":1,"code":"823985e244f4b41d","seconds":12.5,"out":{"test_bpc":2.5442450046539307,"val_bpc":2.3651881217956543,"train_bpc":2.145476818084717,"params":84577},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.086040872921865,5.675188422587985,5.303166903096515,4.973453803647899,4.700749169341985,4.504546545723469,4.443001178251006,4.36330441784118,4.2365425412309365,4.191521266853372]},"prefix_t":0.039}
{"run":"3532a5a9c6b1e5ce","experiment":"mlp","fn":"fn:train","args":{"d":"data:char-bigram/text","k":8,"make":"fn:window","steps":3000},"seed":2,"code":"823985e244f4b41d","seconds":12.41,"out":{"test_bpc":2.524120569229126,"val_bpc":2.3495309352874756,"train_bpc":2.1513655185699463,"params":84577},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.061706015755381,5.657164296812262,5.274223942782813,4.939953647051634,4.6994142400059875,4.600704797543278,4.481647743691688,4.453134052074541,4.480165597206553,4.4053469527058]},"prefix_t":0.035}
```
