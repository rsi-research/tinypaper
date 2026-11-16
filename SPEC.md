# TinyPaper format, version 0.4

A tiny paper is one markdown file that holds a research result in a form a program can check:
its data sources, its code, the calls it makes, its claims, and a log of what those calls gave.
This document defines the file and the operations on it.

Scope: computational experiments. A paper may also hold a proof or a description of how data
was collected (see [Things that cannot be rerun](#things-that-cannot-be-rerun)).

## The four objects

| Object | What it is |
|---|---|
| **Paper** | cites, environments, functions, calls, named sets of calls (experiments), and claims |
| **Call** | a function, its arguments and a seed |
| **Log** | for one machine, what happened when a paper's calls ran |
| **Claim** | a function of a log that returns true or false |

A **vault** is a flat folder of papers in a git repository. Anyone can host one.

A paper carries its author's log inside the file. There is no other store: the log is all a
later paper has unless it reruns yours, so what goes into it is the author's most important
choice (see [What to log](#what-to-log)).

### Two ways to arrive at a paper

- **Write it, then run it.** The paper is the program from the start.
- **Do the research anywhere, then summarise it.** Explore in scripts, notebooks or a framework,
  and write the paper afterwards as the compact form of what was found.

Either way, the log is only ever written by running the paper. Rerunning a summary is what
shows that the compact form regenerates the result. A summary that has not been run is still a
valid paper: its claims are open, and numbers in its prose are not evidence.

## 1. The file

A paper starts with a header and is otherwise prose with fenced blocks. Prose is for people
and is never evidence.

```
---
title: A character bigram predicts Shakespeare better than character frequencies
tinypaper: 0.4
date: 2026-11-15
---
```

All three header fields are required. Seven kinds of block carry the formal content. Any other
fenced block is part of the prose. A block with an unknown field is an error.

| Block | Declares | Written by |
|---|---|---|
| `cite` | another paper, under a short name | author |
| `env` | hardware and pinned packages | author |
| `data` | files by URL and hash, or data derived from another data block | author |
| `python code=<name>` | source code | author |
| `python experiments` | the calls, under experiment names | author |
| `python claim=<id>` | a function of a log that should return true | author |
| `log` | what each call returned and emitted | the runner |

### Where code goes

- **Inline** what this paper contributes, in `code` blocks.
- **Cite** what another paper contributed.
- **Install** what is established, by name and version in an `env` block.

A reader should be able to understand the contribution from the paper alone, without reading
its installed dependencies.

### `cite`

````
```cite
name: base
at: ./char-bigram.md
```
````

- `name` is the short name. It then works everywhere: `from base import ngram` in code,
  `base.text` for a data block, `base.cpu` for an env, `r.base.bigram` for its logged calls.
- `at` is a path relative to the citing paper. A cited paper is always a local file, so a
  vault is complete in itself and ordinary markdown tools can follow its links.
- `source` is optional: the address the file came from, for a paper from another vault. If the
  file is missing it is copied from there.
- `code` is optional: the cited paper's code hash. If present it must match.
- A paper cites directly every paper it takes something from. Names do not reach through a
  cite: there is no `a.b.name`.
- The tools keep one line under the header, `Cites: [char-bigram](char-bigram.md), ...`, derived
  from the cite blocks. It makes the cites ordinary links, on GitHub and in a notes tool such
  as Obsidian, where they become the graph. `paper.tidy()` rewrites it, and so does a run.

### `env`

````
```env
name: cpu
hardware: 2 CPU, 8 GB
packages: [ "torch==2.11.0" ]
```
````

`packages` are pip requirements. A package installed from git (`git+https://...@<commit>`) is
allowed and is listed by `check`, because its code is not visible in any paper.

### `data`

Data is either files or derived.

````
```data
name: text
load: load
files:
  text:
    url: https://example.org/input.txt
    sha256: 86c4e6aa...
```

```data
name: tok256
load: tokenise
from: base.text
args: { merges: 256 }
```
````

- With `files`, the loader named by `load` is called with a dict from each key to a local
  path. Every file is fetched from its URL and its hash verified first.
- With `from`, the loader is called with the source data and the `args`.
- The loader is a function in the paper's code, so how the data is read is visible.
- A data block is a way of writing a call whose output feeds other calls. It is not logged.
- Guideline: a loader returns a dataclass defined in the paper, so the layout is stated once
  and Python enforces it.

### `python code=<name>`

The code blocks of a paper are executed in order in one namespace. They may import from
cited papers by their short names and from installed packages. `name` is a label.

`tinypaper.source(obj)` gives the source text of a function or class defined in a paper, for a
call that needs it, for example to charge an algorithm for the length of its code.

### `python experiments`

````
```python experiments
for k in [1, 2, 4, 8]:
    run("mlp", train, d=base.text, make=window, k=k, steps=3000, seeds=2, env=base.cpu)
```
````

The block only declares calls. `run(name, fn, env=None, seeds=None, **arguments)`:

- `fn` is the function to call, with `arguments` as keywords. An argument is a literal, a
  function, a data block, or another call. A call passed as an argument is replaced by its
  output, and is run first if it is not logged.
- `seeds=k` makes k calls with `seed=1..k`, and `seeds=[0, 5]` with exactly those seeds. Without
  it there is one call, with no seed.
- **The runner sets `seed` and `device`.** Neither may appear in the call. If the function
  has a `device` parameter the runner passes one, so a paper written for a GPU can still be
  started on a CPU.
- `name` is the experiment the call belongs to. An experiment is a named set of calls, and
  claims select calls by it.
- `run` returns the call (a list of them with `seeds`), so it can be passed on.

**A call's identity** is its function, its arguments, its seed, and the code it depends on.
The experiment name is not part of it, so two experiments may share a call, and it runs once.
Changing the code makes every call a different call: an entry logged for the old code belongs
to a call the paper no longer makes. The tools take "the code it depends on" to be all the code
and data blocks of the paper and of the papers it cites.

How finely to split work into calls is the author's choice. One function may pretrain and
fine-tune ten times. Splitting it lets other papers reuse the stages and lets claims select
among them.

### What a call logs

A call logs two things:

- **what its function returns**, which can be numbers, strings, lists, dicts and dataclasses of
  them; and
- **the points it emits as it runs**:

```python
from tinypaper import emit
emit(step=step, loss=loss)
```

Both go into the log as they are. A call may add at most 20,000 characters to the log. A call
that goes over is refused with a message, and nothing is logged for it.

#### What to log

The log is all a later paper will have unless it reruns yours. Log what someone building on
your work would need, and nothing raw.

- **Every number your own claims read.**
- **The standard metric of the papers you cite**, computed by their function on their split, so
  your result can be compared with theirs directly.
- **A curve at a few points**: the first ten steps, so a rerun can be checked cheaply; then
  sparsely; then the end.
- **What the result cost**: model size, and anything else a comparison should be fair about.
  The runner logs the time each call took.
- **A summary where a distribution matters**, such as the metric by group or a few quantiles.
  Never the raw array.
- **If you introduce a better metric**, publish it as a function others can import, and log the
  old metric beside it so the two lines of work stay comparable.

The root paper of a line of work sets these conventions, and later papers inherit them by
importing its functions.

Guideline: draw random numbers on the CPU. A GPU and a CPU give different random numbers for
the same seed. If a call draws its batches and initial weights on the CPU and then moves them,
a seed means the same thing on every device and early points agree closely.

### `python claim=<id>`

````
```python claim=c1 stated=2026-11-15 differs=n
def c1(r):
    return bpc(r.unigram) - bpc(r.bigram) > 1.0
```
````

A claim is a function named `<id>` that takes `r` and returns true or false. It can use the
paper's code and data.

- `r.<experiment>` is the list of that experiment's logged calls. `.where(k=8)` keeps the calls
  with those argument values.
- `r.<cite>.<experiment>` is a cited paper's calls. **A cited paper is always read from its own
  log**, whichever log of this paper the claim is being evaluated on.
- Each call has `.args`, `.seed`, `.emitted`, `.seconds` and `.out`.
- Reading an experiment needs every one of its calls to be in the log: a claim does not see a
  sweep with holes in it.

Header attributes:

- `stated` (required): the date the claim was made.
- `differs` (optional): what the author says differs among the calls whose output the claim
  reads. It lists argument names, or `fn` if the function itself differs. The seed is ignored.
  It is a statement about the claim, and `check` fails if it is wrong.
- `challenges=<cite>.<claim>`: this claim disputes that claim of a cited paper.

A claim is true or false. There is no third verdict. To say "better on average but not on
every stream", make two claims.

A claim's status is computed, never written:

| Status | Meaning |
|---|---|
| holds | the function returned true |
| fails | it returned false |
| open | a call it reads is not in the log |
| error | the function raised |

### `log`

````
```log
{"log": "a40-torch2.6", "ran_on": {"python": "3.11.4", "device": "cuda", "gpu": "NVIDIA A40", "packages": {"torch": "2.6.0+cu124"}}, "history": {"runs": 43, "claims_edited": []}}
{"call": "58044ec27e7560e4", "fn": "fn:train", "args": {...}, "seed": 1, "seconds": 41.7, "out": {"test_bpc": 2.33, "params": 875585}, "emitted": [{"step": 0, "loss": 6.02, "t": 0.31}, ...]}
```
````

The runner writes this block, at the end of the paper, when the author runs it. Nobody writes
it by hand. The first line names the log, the machine, and two counts from the history. Each
further line is one call: its identity, function, arguments and seed, how long it took, what it
returned, and what it emitted.

The log holds only calls the paper still makes, and only calls that finished.

`paper.run()` runs every call that is not in the log; `only=[...]` limits it to some
experiments. Several processes may run the same paper at once, for example one per GPU: each
takes the next call nobody has started, and they do not overwrite each other.

### The history

Beside a vault, in a `logs/` folder that is not committed, the tools keep for each paper a
history: every run ever made for it and every change to a claim's text, in order. It holds
runs of calls the paper no longer makes, which is how much was tried. Two counts from it are
written into the paper's log: how many runs were made, and which claims were edited after
results existed. The same folder holds reruns made on this machine.

## 2. `check`

`paper.check()` reads a paper and reports. It runs no experiments.

**Validity** is a property of the file alone: header fields, block syntax and fields, `stated`
on every claim, no `seed` or `device` in a call, `challenges=` naming a real claim of a cited
paper.

`check` passes if the paper is valid and:

1. Every cited paper, and every paper those cite, is reachable, and matches its pinned code
   hash where a cite gives one.
2. Every data file of the paper and of the papers it cites is reachable with its stated hash.
3. No claim raises.
4. Where a claim declares `differs=` and evaluates, the declaration is right.

It does not fail on a claim that is open or that fails, or on a call that is not in the log.

It also reports:

- each claim's status on the paper's log, and on every rerun kept on this machine;
- how many of the paper's calls are logged and how many are not;
- how the machine differed from the env (package versions, device);
- claims of other papers in the vault that challenge this paper's claims;
- from the history: runs made, how many were for calls the paper no longer makes, and claims
  edited after results existed;
- lines of code in the paper, its cites, its installs, and installs from git.

## 3. `reproduce`

`paper.reproduce(time="10m")` reruns a paper on the machine it is called on, within a time
limit. It makes a new log, kept on this machine, and reports the highest tier that log
reaches. It never changes the paper, and cited papers are read from their own logs.

| Tier | Reached when |
|---|---|
| valid | the paper is valid |
| checked | `check` passes |
| starts | every call starts on this machine: data loads, the function is called, a first point is emitted |
| prefix | the first ten points of every compared call match the paper's log |
| partial | at least one claim was rerun in full |
| full | every call was rerun and every claim evaluated |
| robust | the same with seeds the paper never used (`seeds="fresh"`) |

How the time is spent:

1. Every call is taken ten emitted points. This gives `starts` and `prefix`, and measures this
   machine's speed against the author's.
2. The time left goes to whole claims, cheapest first. A claim's cost is the logged time of the
   calls it reads, scaled by the measured speed. A claim is started only if it is expected to
   fit.
3. A call that emits is stopped when the time is up. A call that never emits cannot be stopped,
   so it is started only if it is expected to fit.

The result carries the tier, what was done inside each tier, each claim's status, warnings,
the measured speed, and the seconds still needed here to rerun everything. A claim whose calls
were not rerun is "not attempted", never "fails". Calling `reproduce` again continues: finished
calls are kept. After a full rerun the result also says how many of all the logged points
agree with the paper.

- A tier counts only if the tiers before it were reached, with one exception: a full rerun is
  stronger evidence than the tiers below it, so `full` and `robust` stand even if `checked`,
  `starts` or `prefix` did not pass. The result then carries a warning naming what did not pass.
- **Matching points.** If the rerun's machine and package versions equal the author's, the
  emitted values must be equal. Otherwise arithmetic differs in the last digits and the
  difference grows with every training step, so a value must be within 0.1% of the paper's.
  Failing that, it must lie in the range of the paper's seeds at that point, widened on each
  side by that range's width (1% of the value if there is one seed).
- A rerun covers the paper's own calls. A full rerun of a paper says nothing new about the
  papers it cites.

## Things that cannot be rerun

- **Collected data.** A paper may include the code that collected its data. It is there to be
  read. `reproduce` never reruns it: the data is taken as given, pinned by its hash.
- **Proofs.** A machine-checked proof is code: the call runs the proof checker and the claim
  is that it passes. An informal proof is prose and is not evidence.

## Not in this version

- A store for full outputs. A log holds what its author chose to log, and nothing else is kept.
- Sharing reruns. A rerun stays on the machine that made it; a disagreement is published as a
  paper that challenges the claim.
- Experiments that call external services.
- Figures. A log holds values; a reader can plot them.
