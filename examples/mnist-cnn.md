---
title: A small CNN beats the MLP on MNIST
tinypaper: 0.1
date: 2026-11-14
---

Convolution helps even on MNIST. This paper adds one thing, a small CNN, and takes everything
else from the MLP paper by link: the data, the training loop, the metric, the result type and
the MLP's recorded accuracy. The model is the only thing that differs between the two.

```cite
name: base
at: ./mnist-mlp.md
```

```python code=model
import torch
from base import train


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

```claim
id: c1
stated: 2026-11-14
expr: mean(cnn.test_acc) - mean(base.mlp.test_acc) > 0.005
```

```claim
id: c2
stated: 2026-11-14
expr: mean(cnn.train_s) < 5 * mean(base.mlp.train_s)
```
