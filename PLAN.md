# Example set: plan

The examples are the test cases for the format. Each one exists to exercise a kind of
dependency or a rule. This file lists the planned set, what each paper tests, and how far it is.

Status of the set: the language papers are written in format 0.3 under `example-vault/` and run
with the checker and runner. The vision
papers and the joining papers are not written.

## The graph

```
   VISION                                        LANGUAGE
   mnist-mlp ── seeds-check                      char-bigram ───────────────┐
     |    \                                      /     |      \             |
   mnist-cnn  augment                        ngram  neural-bigram  bpe      |
     |    \   /                                |       |            |       |
     |  cnn-augment                            |     mlp-lm         |       |
   batchnorm                                   |     /              |       |
     |                                         |   rnn              |       |
   resnet                                      |    |  \            |       |
     |                                         |  transformer ── transformer-bpe
   calibration                                 |    |    \
                                               |    |   equal-params
                                             where-it-helps

   JOINS (each cites both sides)
   skip       resnet + transformer     skip connections removed and restored in both
   norms      batchnorm + transformer  batch, layer, group and no normalisation in both
   vit        resnet + transformer     a transformer on CIFAR-10 image patches
   pixel-lm   mnist-mlp + transformer  MNIST pixels as a sequence, in bits per pixel
```

## Language papers (written, format 0.3)

| Paper | Cites | What it tests in the format |
|---|---|---|
| `char-bigram` | none | the root: data, split, raw per-character outputs, metric; runs with no seed |
| `ngram` | char-bigram | a sweep written as a Python loop; a paper that defines no code |
| `neural-bigram` | char-bigram | a bridge with a known answer: the trained table must match the counts |
| `mlp-lm` | char-bigram, neural-bigram, ngram | three cites; claims against another paper's sweep |
| `rnn` | char-bigram, neural-bigram, mlp-lm | a sibling model |
| `transformer` | char-bigram, neural-bigram, rnn | a block written for reuse (`norm` and `skip` are arguments) |
| `bpe` | char-bigram | data derived from data by code; a metric whose unit changes; a claim about data alone |
| `transformer-bpe` | char-bigram, neural-bigram, transformer, bpe | a merge: defines nothing, two parents share an ancestor |
| `equal-params` | char-bigram, neural-bigram, rnn, transformer | a challenge to another paper's claim; choosing settings on validation |
| `where-it-helps` | char-bigram, ngram, rnn, transformer | analysis only: no runs, claims over other papers' logs, open without them |

## Vision papers (planned)

| Paper | Cites | What it tests | State |
|---|---|---|---|
| `mnist-mlp` | none | the vision root | planned |
| `mnist-cnn` | mnist-mlp | one cite | planned |
| `seeds-check` | mnist-mlp | a challenge by rerunning with more seeds | planned |
| `augment` | mnist-mlp | a method independent of the model; a shifted test set derived by code | planned |
| `cnn-augment` | mnist-cnn, augment | a merge | planned |
| `batchnorm` | mnist-mlp, mnist-cnn | new data, a replaced loop, a bridge | planned |
| `resnet` | batchnorm | extending a cited sweep | planned |
| `calibration` | mnist-cnn, batchnorm, resnet | analysis only | planned |

## Joining papers (planned)

| Paper | Cites | Claim | What it tests |
|---|---|---|---|
| `skip` | resnet, transformer | removing skip connections hurts both models, and hurts more with depth | one idea tested in two fields; importing blocks from both |
| `norms` | batchnorm, transformer | each model does best with its usual normalisation; none is worst at depth | a grid across two trees; adapters between data layouts |
| `vit` | resnet, transformer | at this scale the transformer is worse than the ResNet on CIFAR-10 | model from one tree, data and loop from the other |
| `pixel-lm` | mnist-mlp, transformer | a transformer over pixels beats a pixel bigram | vision data through the language loop and metric |

The format the papers use is defined in [SPEC.md](SPEC.md).

## What has been run

Every paper in the example vault has been run in full by its author run, on one A40 GPU with
PyTorch 2.6.0, and reports its results in its own results block. All 20 claims hold. `check`
flags each paper as run differently from its declared environment (PyTorch 2.11.0 on CPU).

The logs are in the store named in `tinypaper.toml`. They hold that run as record set
`a40-torch2.6`, and a second record set, `cpu-torch2.11`: a two-minute `reproduce` of each paper
on the declared environment. There the four count-model and small papers reached `full`, and
the five larger ones reached `prefix`, with all 28 neural runs matching the reported values on
their first ten steps.

| Model | Test bits per character |
|---|---|
| unigram / bigram (counts) | 4.85 / 3.59 |
| best 5-gram (counts) | 2.63 |
| bigram over byte-pair tokens | 2.90 |
| trained bigram table | 3.61 |
| MLP, window 1 / 2 / 4 / 8 | 3.61 / 3.03 / 2.55 / 2.53 |
| GRU, width 128 | 2.38 |
| transformer | 2.33 |
| transformer on byte-pair tokens | 2.51 |
| GRU at the transformer's size, tuned | 2.24 |
| transformer, tuned | 2.33 |

Two claims failed on the first full run and were rewritten to what was found, and each paper
says so in its prose:

- `mlp-lm` / `context_helps` first said every doubling of the window helps. A window of 8 is no
  better than 4 on test.
- `transformer-bpe` first said tokens beat characters. On this much text the token model
  overfits and is worse.

The challenge holds: at equal size and tuned learning rates the GRU beats the transformer, so
`transformer` / `beats_rnn` is true as stated and misleading as a conclusion. `check` on
`transformer` reports the challenge.
