---
title: A small CNN beats the MLP on MNIST
tinypaper: 0.1
date: 2026-11-14
---

Convolution helps even on MNIST. This paper adds one thing, a small CNN, and takes everything
else from the MLP paper by link: the data, the training loop, the metric, the result type and
the MLP's recorded accuracy. The model is the only thing that differs between the two.

The accuracy claim uses the MLP's recorded results, with no rerun. The cost claim compares
training times, which depend on the machine, so this paper reruns the MLP itself and compares
the two times from the same run.

```cite
name: base
at: ./mnist-mlp.md
```

```python code=model
from statistics import mean
import torch
from base import train, mlp


def cnn():
    return torch.nn.Sequential(
        torch.nn.Unflatten(1, (1, 28, 28)),
        torch.nn.Conv2d(1, 16, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(), torch.nn.Linear(16 * 12 * 12, 10))
```

```experiment
name: cnn
call: train(base.mnist, cnn, epochs=5, seed={seed})
env: base.cpu
seeds: 3
```

```experiment
name: mlp_here
call: train(base.mnist, mlp, epochs=5, seed={seed})
env: base.cpu
seeds: 3
```

```python claim=c1 stated=2026-11-14
def c1(r):
    # accuracy: against the MLP's recorded results in the cited paper
    return mean(x.test_acc for x in r.cnn) - mean(x.test_acc for x in r.base.mlp) > 0.005
```

```python claim=c2 stated=2026-11-14
def c2(r):
    # cost: against the MLP rerun here, because train_s is a Timing field
    return mean(x.train_s for x in r.cnn) < 5 * mean(x.train_s for x in r.mlp_here)
```
