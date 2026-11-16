---
title: A bigram table trained by gradient descent matches the counted bigram
tinypaper: 0.3
date: 2026-11-15
---

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

```python code=train
import math
import torch
from tinypaper import emit
from base import bpc, part, per_char


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
    return dict({name + "_bpc": per_char(b, d, name) for name, b in cost.items()}, test_bits=cost["test"],
                params=sum(p.numel() for p in model.parameters()))
```

```python code=model
def table(vocab):
    # logits for the next id, looked up by the current id
    t = torch.nn.Embedding(vocab, vocab)
    torch.nn.init.zeros_(t.weight)
    return t
```

```python experiments
run("table", train, d=base.text, make=table, steps=3000, lr=0.03, seeds=2, env=base.cpu)
```

```python claim=bridge stated=2026-11-15 differs=fn
def bridge(r):
    # the trained table is within 0.02 bits per character of the counted bigram
    return abs(bpc(r.table) - bpc(r.base.bigram)) < 0.02
```

```results
{"set": "a40-torch2.6", "ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"run":"36aa3806ec80c19e","experiment":"table","fn":"fn:train","args":{"d":"data:char-bigram/text","lr":0.03,"make":"fn:table","steps":3000},"seed":1,"code":"f2616b50d9d7c88e","seconds":10.68,"out":{"test_bpc":3.605323553085327,"val_bpc":3.5698604583740234,"train_bpc":3.5528621673583984,"params":4225},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.022363264544005,5.961982219617883,5.903484728684011,5.847699060188943,5.787936464875324,5.732456237568189,5.677347148821616,5.6178339283517476,5.570467154581716,5.513681922919834]},"prefix_t":4.133}
{"run":"1eaff3f4e6b9c324","experiment":"table","fn":"fn:train","args":{"d":"data:char-bigram/text","lr":0.03,"make":"fn:table","steps":3000},"seed":2,"code":"f2616b50d9d7c88e","seconds":5.38,"out":{"test_bpc":3.610396385192871,"val_bpc":3.5751848220825195,"train_bpc":3.550977945327759,"params":4225},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.022363264544005,5.961348635532104,5.903586542413321,5.844222258919101,5.7861333987636785,5.734755989575526,5.674486389407164,5.624069675306732,5.569364401824384,5.514252905320699]},"prefix_t":0.036}
```
