# TinyPaper

**Papers small enough to rerun.** TinyPaper is a markdown format for research results that can be
checked by running them. One file holds a result's data sources, code, experiments, claims and
reported results. A tool compiles the file into a program, runs it, and reports which claims hold.

> **Status: early.** The format is at version 0.3 ([SPEC.md](SPEC.md)). The tools work on the
> ten papers of the [example vault](example-vault). They are not yet exposed to agents as tools.

## Why

Research agents can produce results faster than anyone can read them, and prose is a poor thing
to trust: it can be wrong, stale, or written to mislead. A tiny paper is meant to be:

- **Compact.** It holds the code the paper contributes, its claims and its results. Everything else is cited or installed.
- **Linked.** It cites other papers' code, data and results instead of restating them.
- **Executable.** Its blocks assemble into a program: fetch data, run experiments, evaluate claims.
- **Checkable.** Each claim is a function of the results that should return true, so anyone can rerun it.
- **Readable by both.** Agents read the blocks. People read the prose around them.

## What a tiny paper looks like

A paper is one markdown file. Fenced blocks carry the formal content; everything else is prose
for people and is never evidence.

| Block | Declares |
|---|---|
| header | title, format version, date |
| `cite` | another paper, under a short name |
| `env` | hardware and pinned packages |
| `data` | files by URL and sha256, or data derived from other data, and the function that reads it |
| `python code=<name>` | source code |
| `python experiments` | the runs, declared with `run(...)` |
| `python claim=<id>` | a function of the results that should return true |
| `results` | the small values each run returned, written by the runner |

From [example-vault/mlp-lm.md](example-vault/mlp-lm.md):

````markdown
```cite
name: nb
at: ./neural-bigram.md
```

```python code=model
from nb import train        # the training loop of an earlier paper

class Window(torch.nn.Module):
    ...
```

```python experiments
for k in [1, 2, 4, 8]:
    run("mlp", train, d=base.text, make=window, k=k, steps=3000, seeds=2, env=base.cpu)
```

```python claim=beats_counts stated=2026-11-15 differs=fn
def beats_counts(r):
    best = min(bpc(r.ng.sweep.where(n=n, alpha=a)) for n in [3, 4, 5, 6, 7] for a in [1.0, 0.03])
    return bpc(r.mlp.where(k=8)) < best
```
````

## Papers, vaults, logs and the store

- A **vault** is a flat folder of papers in a git repository. Anyone can host one. This
  repository holds the tools and one [example vault](example-vault).
- A paper reports its own results: the small values each run returned are written into the
  paper by the runner. With the paper alone you can read them, cite them, and rerun the paper.
- **Logs** are everything too large for the paper: full run outputs and the run log. They are
  optional. With them you can recompute anything from the raw outputs.
- A **store** is where logs are shared. This repository's store is the public dataset
  [rsi-research-org/tinypaper-store](https://huggingface.co/datasets/rsi-research-org/tinypaper-store),
  set in [tinypaper.toml](tinypaper.toml). A token, when a store needs one, goes in a `.env`
  file (see [.env.example](.env.example)).

The rules that keep it honest:

1. **The paper's own code is in the file.** A result is the value a function returned, not text a script printed.
2. **A claim's status is computed, never written.** It holds, fails, or is open.
3. **Prose is not evidence.** Results are written by the runner, and `check` says whether they are only reported or also backed by logs.
4. **Everything is logged.** The run log records every run made for a paper and every edit to a claim, so a reader can see how much was tried.

## Using it

```python
import tinypaper as tp

vault = tp.Vault("example-vault")          # a folder of papers
paper = vault["transformer"]               # one paper, as typed parts

paper.claims                               # its claims; also .cites, .envs, .data, .code, .results
print(paper.check().text())                # reachable? consistent? which claims hold on the reported results?

paper.load_logs()                          # optional: fetch the full outputs from the store
print(paper.check().text())                # now also checks that the logs agree with the paper

result = paper.reproduce(time="10m")       # rerun on this machine, as far as ten minutes gets
print(result.text())                       # the tier reached, each claim, and the time still needed

paper.run(name="mine")                     # as the author: run what has no fresh result, and write the results block
paper.push()                               # upload the logs to the store
vault.to_jsonl("vault.jsonl")              # every paper as one line of JSON
```

- **`check`** runs nothing. It fails if the paper breaks the format, if something needed to
  rerun it is unreachable (a cited paper, a data file), or if the logs contradict what the
  paper reports. It reports each claim's status, stale results, how the machine differed from
  the declared environment, and claims in other papers that challenge this one.
- **`reproduce`** reports a tier, not a score: valid, checked, starts, prefix, partial, full,
  robust. More time reaches a higher tier, and a second call continues where the first
  stopped. A paper written for a GPU still reaches a tier on a CPU.

## The example vault

Ten papers on character-level language modelling, from a counted bigram to a transformer.
[PLAN.md](PLAN.md) lists what each one tests and what is planned next. They include a paper
that defines no code, a paper with no runs, a tokeniser whose data is derived by code, and a
paper that challenges another's claim.

## Tests

```bash
python tests/test_tinypaper.py
```

## Roadmap

1. Check every submitted paper automatically in CI.
2. Write the vision examples and the papers that join the two sets.
3. Expose `check`, `run` and `reproduce` to agents as tools.
