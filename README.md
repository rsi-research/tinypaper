# TinyPaper

**Papers small enough to rerun.** TinyPaper is a markdown format for research results that can be
checked by running them. One file holds a result's data sources, code, experiments and claims.
A tool compiles the file into a program, runs it, and reports which claims hold.

> **Status: design draft.** This repository holds the format as two worked examples. The checker
> and runner described below are not written yet.

## Why

Research agents can produce results faster than anyone can read them, and prose is a poor thing
to trust: it can be wrong, stale, or written to mislead. A tiny paper is meant to be:

- **Compact.** It holds only what is needed to regenerate a result, plus the claims still open.
- **Linked.** It cites blocks of other papers (code, data, recorded results) instead of restating them.
- **Executable.** Its blocks assemble into a pipeline: set up, fetch data, run experiments, evaluate claims.
- **Checkable.** Each claim is a function over typed results that should return true, so anyone can rerun it.
- **Readable by both.** Agents read the typed blocks. People read the prose around them.

## What a tiny paper looks like

A paper is one markdown file. Fenced blocks carry the formal content; everything else is prose
for people and is never evidence.

| Block | Declares |
|---|---|
| header | title, format version, date |
| `cite` | another paper, under a short name |
| `env` | hardware, pinned packages, known pitfalls |
| `data` | files by URL and sha256, and the function that reads them |
| `python code=<name>` | source code, in the file, joined into one module |
| `experiment` | a function call with its environment and seeds; returns a typed result |
| `python claim=<id>` | a Python function over experiment results that should return true |

A claim, from [examples/mnist-mlp.md](examples/mnist-mlp.md):

````markdown
```experiment
name: mlp
call: train(mnist, mlp, epochs=5, seed={seed})
env: cpu
seeds: 3
```

```python claim=c2 stated=2026-11-14
def c2(r):
    return mean(x.test_acc for x in r.mlp) - mean(x.test_acc for x in r.linear) > 0.04
```
````

A claim is an ordinary Python function. It receives `r`, where `r.<experiment>` is the list of
that experiment's results, one per seed, and it can compute any statistic Python can.

Three rules keep it honest:

1. **All code is in the file.** There are no pointers to scripts whose output must be trusted. A
   result is the value an experiment's function returns, not text a script printed.
2. **A claim's status is computed, never written.** A claim whose experiments have recorded
   results is reported as holding or failing. A claim without them is reported as open.
3. **Prose is not evidence.** A number in the prose must link to a recorded result.

## Linking

One `cite` line gives another paper a short name. That name then works in code, in experiments
and in claims. From [examples/mnist-cnn.md](examples/mnist-cnn.md):

````markdown
```cite
name: base
at: ./mnist-mlp.md
```

```python code=model
from base import train          # the same training loop, metric and result type
```

```experiment
name: cnn
call: train(base.mnist, cnn, epochs=5, seed={seed})
env: base.cpu
```

```python claim=c1 stated=2026-11-14
def c1(r):
    return mean(x.test_acc for x in r.cnn) - mean(x.test_acc for x in r.base.mlp) > 0.005
```
````

The CNN paper is about ten lines of new code. Its data, training loop, metric and baseline
numbers all come from the MLP paper, so the model is the only thing that differs between the
two. A cite to a paper in another repository is pinned to a commit and a content hash.

The checker records which results a claim reads. Results that a claim compares must share the
same data, the same metric code and the same result type, and every block is hashed, so this is
checked mechanically.

Some result fields depend on the machine, such as training time. They are declared with the
`Timing` type, and a claim may compare them only when one run of the pipeline produced both.
The CNN paper therefore reruns the MLP itself for its cost claim, and uses the MLP paper's
recorded results only for its accuracy claim.

## Checking and reproducing

Two operations are planned:

- **check** confirms that every link resolves, every hash matches and every recorded result was
  produced by the current blocks, then evaluates each claim on the results recorded beside the
  file. A paper that depends on something unreachable fails the check.
- **reproduce** ignores the recorded results and runs the whole pipeline again, with seeds the
  reproducer picks. It reports a score: the fraction of claims that still hold.

Setting up an environment on a new machine is the step most likely to fail. A reproduction that
cannot run an experiment reports its claims as "not attempted", not as refuted.

## Examples

- [examples/mnist-mlp.md](examples/mnist-mlp.md): an MLP against a linear classifier on MNIST.
  It has two claims and one more about a wider MLP.
- [examples/mnist-cnn.md](examples/mnist-cnn.md): a follow-up that cites the MLP paper and adds a CNN.
  It has an accuracy claim against the MLP's recorded results and a cost claim against a rerun.

The code and claims in both examples have been run by hand, with two seeds, to confirm they work;
every claim returned true. Neither example has recorded results yet, because the runner that writes them does not exist, so every
claim in them is currently open.

## Roadmap

1. Write the format down as a specification.
2. Build the checker: parse the blocks, resolve links, evaluate claims on recorded results.
3. Build the runner: assemble a paper into a program, run it, and record the results.
4. Expose check and reproduce to agents as tools.
