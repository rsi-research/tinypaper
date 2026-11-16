---
title: A bigram table trained by gradient descent matches the counted bigram
tinypaper: 0.5
date: 2026-11-15
---

Claims: ✅ 1 of 1 hold

Cites: [char-bigram](char-bigram.md)

This paper adds the training loop that the neural language papers share. The loop reports its
loss at each of the first ten steps and then every 100 with `emit`, so a rerun can be compared
with a record after a few seconds on any machine. Its only model is a
table of logits, one row per previous character, which is the counted bigram in another form.
So the paper's claim is a bridge: the loop and its scoring reproduce a number that is known
exactly.

```cite
name: base
at: ./char-bigram.md
```

**Code `train`**

```python code=train
import math
import torch
from tinypaper import emit
from base import bpc, part, summary


def score(model, ids, skip, ctx, device="cpu"):
    # Bits for every id from `skip` on. Windows of `ctx` ids advance by half a window and
    # only the second half of each is scored, so every scored id sees at least ctx / 2
    # earlier ids. One last window covers the tail.
    half, n = ctx // 2, len(ids)
    assert skip > half
    starts = torch.arange(skip - half - 1, n - ctx, half)
    done = starts[-1].item() + ctx

    def logp(rows):
        x, y = ids[rows].to(device), ids[rows + 1].to(device)
        return torch.log_softmax(model(x), -1).gather(-1, y[..., None])[..., 0].cpu()

    model.eval()
    with torch.no_grad():
        out = [logp(rows)[:, half:].flatten() for rows in (starts[:, None] + torch.arange(ctx)).split(256)]
        if done < n - 1:
            out.append(logp(torch.arange(n - 1 - ctx, n - 1)[None])[0, done - n + 1:])
    return -torch.cat(out) / math.log(2)


def train(d, make, steps, seed, ctx=64, lr=3e-3, size=64, device="cpu", **kw):
    # Next-id prediction on random windows. `make(vocab=..., **kw)` builds a model that maps
    # ids of shape (batch, time) to logits of shape (batch, time, vocab).
    torch.manual_seed(seed)
    model = make(vocab=len(d.vocab), **kw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ids = d.train.to(device)
    for step in range(steps):
        model.train()
        # drawn on the CPU, so a seed picks the same windows on every device
        i = (torch.randint(len(ids) - ctx - 1, (size, 1)) + torch.arange(ctx)).to(device)
        loss = torch.nn.functional.cross_entropy(model(ids[i]).flatten(0, 1), ids[i + 1].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step < 10 or step % 100 == 0:      # every early step, so a rerun can be compared after ten
            emit(step=step, loss=loss.item() / math.log(2))
    cost = {name: score(model, part(d, name), d.skip, ctx, device) for name in ["test", "val", "train"]}
    return dict(summary(cost, d), params=sum(p.numel() for p in model.parameters()))
```

**Code `model`**

```python code=model
def table(vocab):
    # logits for the next id, looked up by the current id
    t = torch.nn.Embedding(vocab, vocab)
    torch.nn.init.zeros_(t.weight)
    return t
```

**Experiments**

```python experiments
call("table", train, d=base.text, make=table, steps=3000, lr=0.03, seeds=2, env=base.cpu)
```

**Claim `bridge`** (stated 2026-11-15): The trained table is within 0.02 bits per character of the counted bigram.

```python claim=bridge stated=2026-11-15
def bridge(r):
    """The trained table is within 0.02 bits per character of the counted bigram."""
    return abs(bpc(r.table) - bpc(r.base.bigram)) < 0.02
```

**Log**

```log
{"ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"call":"e69548aef1e490d9","fn":"fn:train","args":{"d":"data:char-bigram/text","lr":0.03,"make":"fn:table","steps":3000},"seed":1,"seconds":10.37,"out":{"test_bpc":3.605323553085327,"val_bpc":3.5698604583740234,"train_bpc":3.5528621673583984,"at_word_start":4.404512882232666,"elsewhere":3.4204776287078857,"params":4225},"emitted":[{"step":0,"loss":6.022363264544005,"t":4.419},{"step":1,"loss":5.961982219617883,"t":4.422},{"step":2,"loss":5.903484728684011,"t":4.427},{"step":3,"loss":5.847699060188943,"t":4.431},{"step":4,"loss":5.787936464875324,"t":4.432},{"step":5,"loss":5.732456237568189,"t":4.434},{"step":6,"loss":5.677347148821616,"t":4.435},{"step":7,"loss":5.6178339283517476,"t":4.436},{"step":8,"loss":5.570467154581716,"t":4.438},{"step":9,"loss":5.513681922919834,"t":4.441},{"step":100,"loss":3.647590705073662,"t":4.649},{"step":200,"loss":3.5674660518284895,"t":4.891},{"step":300,"loss":3.5346964775330902,"t":5.103},{"step":400,"loss":3.5665868765172792,"t":5.313},{"step":500,"loss":3.5663880645728816,"t":5.529},{"step":600,"loss":3.597607043288102,"t":5.726},{"step":700,"loss":3.514931543365333,"t":5.91},{"step":800,"loss":3.5129410161642465,"t":6.087},{"step":900,"loss":3.591136024066738,"t":6.249},{"step":1000,"loss":3.5545883348971845,"t":6.415},{"step":1100,"loss":3.5396162132436717,"t":6.588},{"step":1200,"loss":3.547851774462888,"t":6.774},{"step":1300,"loss":3.569691851295956,"t":6.971},{"step":1400,"loss":3.5421006746180357,"t":7.176},{"step":1500,"loss":3.542739762148642,"t":7.373},{"step":1600,"loss":3.5600023487463455,"t":7.594},{"step":1700,"loss":3.558445905756037,"t":7.729},{"step":1800,"loss":3.5088557402756515,"t":7.93},{"step":1900,"loss":3.5375844102063785,"t":8.166},{"step":2000,"loss":3.5610208300047552,"t":8.408},{"step":2100,"loss":3.5328081080266167,"t":8.618},{"step":2200,"loss":3.5174012142317226,"t":8.835},{"step":2300,"loss":3.5059884455204666,"t":9.038},{"step":2400,"loss":3.555395621460334,"t":9.235},{"step":2500,"loss":3.542968155108988,"t":9.428},{"step":2600,"loss":3.5111637474502304,"t":9.614},{"step":2700,"loss":3.5480099985016817,"t":9.785},{"step":2800,"loss":3.5224671351955377,"t":9.945},{"step":2900,"loss":3.4991373446406957,"t":10.127}]}
{"call":"8f499f0fe5edc686","fn":"fn:train","args":{"d":"data:char-bigram/text","lr":0.03,"make":"fn:table","steps":3000},"seed":2,"seconds":5.91,"out":{"test_bpc":3.610396385192871,"val_bpc":3.5751848220825195,"train_bpc":3.550977945327759,"at_word_start":4.403777599334717,"elsewhere":3.4268946647644043,"params":4225},"emitted":[{"step":0,"loss":6.022363264544005,"t":0.012},{"step":1,"loss":5.961348635532104,"t":0.014},{"step":2,"loss":5.903586542413321,"t":0.015},{"step":3,"loss":5.844222258919101,"t":0.016},{"step":4,"loss":5.7861333987636785,"t":0.017},{"step":5,"loss":5.734755989575526,"t":0.018},{"step":6,"loss":5.674486389407164,"t":0.018},{"step":7,"loss":5.624069675306732,"t":0.019},{"step":8,"loss":5.569364401824384,"t":0.02},{"step":9,"loss":5.514252905320699,"t":0.022},{"step":100,"loss":3.6926560070098473,"t":0.198},{"step":200,"loss":3.574028221854811,"t":0.322},{"step":300,"loss":3.5639689566058452,"t":0.523},{"step":400,"loss":3.513451804637309,"t":0.738},{"step":500,"loss":3.5301688622664757,"t":0.944},{"step":600,"loss":3.571218713270316,"t":1.151},{"step":700,"loss":3.5551345517963244,"t":1.396},{"step":800,"loss":3.545911122230552,"t":1.584},{"step":900,"loss":3.5502378617609587,"t":1.765},{"step":1000,"loss":3.5018680851710955,"t":1.942},{"step":1100,"loss":3.5518661934993276,"t":2.129},{"step":1200,"loss":3.5823793554154073,"t":2.337},{"step":1300,"loss":3.524027361804165,"t":2.531},{"step":1400,"loss":3.5506530278801414,"t":2.72},{"step":1500,"loss":3.5335968204634733,"t":2.892},{"step":1600,"loss":3.544117343182053,"t":3.065},{"step":1700,"loss":3.5538233560661467,"t":3.237},{"step":1800,"loss":3.5347903800604614,"t":3.41},{"step":1900,"loss":3.5883333947882776,"t":3.605},{"step":2000,"loss":3.5478362960243106,"t":3.808},{"step":2100,"loss":3.5397018606038015,"t":4.021},{"step":2200,"loss":3.525605474608482,"t":4.244},{"step":2300,"loss":3.550135704066346,"t":4.442},{"step":2400,"loss":3.556045715880596,"t":4.56},{"step":2500,"loss":3.5586092892743566,"t":4.803},{"step":2600,"loss":3.5887382419484086,"t":5.047},{"step":2700,"loss":3.522904315094031,"t":5.267},{"step":2800,"loss":3.5826696621300638,"t":5.475},{"step":2900,"loss":3.4907985938309616,"t":5.681}]}
```
