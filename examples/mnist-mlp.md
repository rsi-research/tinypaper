---
title: An MLP beats a linear classifier on MNIST
tinypaper: 0.1
date: 2026-11-14
---

One hidden layer is enough to do clearly better than a linear model on MNIST. Both models share
the data, the training loop and the metric, so the model is the only thing that differs.

Prose like this is for people. Nothing here is evidence; the claims at the bottom are.

```env
name: cpu
hardware: 4 CPU, 8 GB, no GPU
packages: [ "torch==2.11.0", "numpy==2.5.4" ]
pitfalls:
  - symptom: accuracy differs in the third decimal between machines
    fix: expected; CPU math libraries differ, and the claims leave room for it
```

```data
name: mnist
load: load
files:
  train_x:
    url: https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    sha256: 440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609
  train_y:
    url: https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    sha256: 3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c
  test_x:
    url: https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    sha256: 8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6
  test_y:
    url: https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    sha256: f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6
```

```python code=load
import gzip, time
from dataclasses import dataclass
from statistics import mean
import numpy as np
import torch
from tinypaper import Timing


def idx(path, header):
    # a raw IDX file: skip the header, read the rest as bytes
    return np.frombuffer(gzip.open(path).read(), np.uint8, offset=header)


def load(files):
    t = lambda a: torch.tensor(a.copy())
    return dict(
        train_x=t(idx(files["train_x"], 16)).reshape(-1, 784).float() / 255,
        train_y=t(idx(files["train_y"], 8)).long(),
        test_x=t(idx(files["test_x"], 16)).reshape(-1, 784).float() / 255,
        test_y=t(idx(files["test_y"], 8)).long())
```

```python code=model
def linear():
    return torch.nn.Linear(784, 10)


def mlp(hidden=256):
    return torch.nn.Sequential(
        torch.nn.Linear(784, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 10))


def mlp_wide():
    return mlp(1024)
```

```python code=evaluate
def evaluate(model, d):
    # the only place the test set is read
    with torch.no_grad():
        pred = model(d["test_x"]).argmax(1)
    return (pred == d["test_y"]).float().mean().item()
```

```python code=train
@dataclass
class Result:
    test_acc: float
    train_s: Timing    # depends on the machine; comparable only within one run of the pipeline


def train(d, make, epochs, seed):
    # `make` builds the model, so a follow-up can train any model with this same loop
    torch.manual_seed(seed)
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(d["train_x"])
    t0 = time.time()
    for _ in range(epochs):
        for i in torch.randperm(n).split(128):
            loss = torch.nn.functional.cross_entropy(model(d["train_x"][i]), d["train_y"][i])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return Result(test_acc=evaluate(model, d), train_s=time.time() - t0)
```

```experiment
name: linear
call: train(mnist, linear, epochs=5, seed={seed})
env: cpu
seeds: 3
```

```experiment
name: mlp
call: train(mnist, mlp, epochs=5, seed={seed})
env: cpu
seeds: 3
```

```experiment
name: mlp_wide
call: train(mnist, mlp_wide, epochs=5, seed={seed})
env: cpu
seeds: 3
```

```python claim=c1 stated=2026-11-14
def c1(r):
    # r.<experiment> is a list of Result objects, one per seed
    return mean(x.test_acc for x in r.mlp) >= 0.965
```

```python claim=c2 stated=2026-11-14
def c2(r):
    return mean(x.test_acc for x in r.mlp) - mean(x.test_acc for x in r.linear) > 0.04
```

```python claim=c3 stated=2026-11-14
def c3(r):
    return mean(x.test_acc for x in r.mlp_wide) - mean(x.test_acc for x in r.mlp) > 0.002
```
