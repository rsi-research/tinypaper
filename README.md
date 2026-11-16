# TinyPaper

**Papers small enough to rerun.** TinyPaper is a markdown format for research results that can be
checked by running them. One file holds a result's data sources, code, experiments, claims and
a log of what its runs gave. A tool compiles the file into a program, runs it, and reports which
claims hold.

> **Status: early.** The format is at version 0.5 ([SPEC.md](SPEC.md)). The tools work on the
> ten papers of the [example vault](example-vault). They are not yet exposed to agents as tools.

## Why

Research agents can produce results faster than anyone can read them, and prose is a poor thing
to trust: it can be wrong, stale, or written to mislead. A tiny paper is meant to be:

- **Compact.** It holds the code the paper contributes, its claims and a short log. Everything else is cited or installed.
- **Linked.** It cites other papers' code, data and logs instead of restating them.
- **Executable.** Its blocks assemble into a program: fetch data, run experiments, evaluate claims.
- **Checkable.** Each claim is a function of the log that should return true, so anyone can rerun it.
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
| `python experiments` | the calls, declared with `call(...)` |
| `python claim=<id>` | a function of the log that should return true |
| `log` | what each call returned and emitted, written by the runner |

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
    call("mlp", train, d=base.text, make=window, k=k, steps=3000, seeds=2, env=base.cpu)
```

```python claim=beats_counts stated=2026-11-15
def beats_counts(r):
    best = min(bpc(r.ng.sweep.where(n=n, alpha=a)) for n in [3, 4, 5, 6, 7] for a in [1.0, 0.03])
    return bpc(r.mlp.where(k=8)) < best
```
````

## Papers, vaults and logs

- A **vault** is a flat folder of papers in a git repository. Anyone can host one. This
  repository holds the tools and one [example vault](example-vault).
- A paper carries its **log**: what each of its calls returned and emitted, written into the
  file by the runner. There is no other store. With the paper alone you can read its results,
  check its claims, cite them, and rerun it.
- The log is short because the author chooses what goes in it: the numbers the claims read,
  the metric later papers will compare against, a curve at a few points. It is all a later
  paper has unless it reruns yours.
- A paper can be written first and run, or written afterwards as the summary of research done
  elsewhere. Either way its log comes from running the paper itself.
- A vault's cites are ordinary markdown links, so a vault opens as a linked set of notes in a
  tool such as Obsidian.

The rules that keep it honest:

1. **The paper's own code is in the file.** A result is the value a function returned, not text a script printed.
2. **A claim's status is computed, never written.** It holds, fails, or is open.
3. **Prose is not evidence.** The log is written by the runner, and a rerun is compared with it point by point.
4. **What was tried is counted.** The log records how many runs were made for the paper and which claims were edited after results existed.

## Using it

```python
import tinypaper as tp

vault = tp.Vault("example-vault")          # a folder of papers
paper = vault["transformer"]               # one paper, as typed parts

paper.claims                               # its claims; also .cites, .envs, .data, .code, .log
print(paper.check().text())                # reachable? consistent? which claims hold on its log?

result = paper.reproduce(time="10m")       # rerun on this machine, as far as ten minutes gets
print(result.text())                       # calls started and matching, claims rerun, and the time still needed

paper.run()                                # as the author: run what is not logged, and write the log into the paper
                                           # (safe to call from several processes at once, one per GPU)
vault.run()                                # the same for every paper, cited papers first
```

- **`check`** runs nothing. It fails if the paper breaks the format, or if something needed to
  rerun it is unreachable (a cited paper, a data file). It reports each claim's status, which
  calls are not logged, how the machine differed from the declared environment, and claims in
  other papers that challenge this one.
- **`reproduce`** reports counts: how many calls started, how many match the paper on their
  first points, how many were rerun to the end, and which claims held on the rerun. More time
  gets further, and a second call continues where the first stopped. A paper written for a GPU
  can still be started and compared on a CPU.

## The example vault

Ten papers on character-level language modelling, from a counted bigram to a transformer.
[PLAN.md](PLAN.md) lists what each one tests and what is planned next. They include a paper
that defines no code, a paper with no runs, a tokeniser whose data is derived by code, and a
paper that challenges another's claim.

## CI

[.github/workflows/check.yml](.github/workflows/check.yml) runs on every push and pull request. It
tests the tools, then checks the papers the change touches and the papers that cite them, reruns
each for up to two minutes, and writes a table to the job summary. It executes the code in the
papers, so it is given no secrets.

## Tests

```bash
python tests/test_tinypaper.py
```

## Roadmap

1. Write the vision examples and the papers that join the two sets.
2. Expose `check`, `run` and `reproduce` to agents as tools.
