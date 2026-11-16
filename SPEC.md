# TinyPaper format, version 0.3

A tiny paper is one markdown file that holds a research result in a form a program can check:
its data sources, its code, the runs it declares, its claims, and the results it reports. This
document defines the file, the logs a run leaves behind, and the operations on a paper.

Scope: computational experiments. A paper may also hold a proof or a description of how data
was collected (see [Things that cannot be rerun](#things-that-cannot-be-rerun)).

## Where things live

| Thing | What it is | Where |
|---|---|---|
| **Paper** | one markdown file: code, runs, claims, reported results | a vault |
| **Vault** | a flat folder of papers | a git repository; anyone can host one |
| **Logs** | a paper's full run outputs and its run log | beside the vault in `logs/`, and optionally in a store |
| **Store** | where logs are shared | a directory, or a Hugging Face repository |

A paper is complete without logs. With the paper alone you can read its reported results, cite
them, and rerun it. With its logs you can also recompute anything from the raw outputs.

Rule of thumb: what an author writes is markdown, what the runner measures and is small goes
into the paper, and what the runner measures and is large goes to the logs.

### Two ways to arrive at a paper

- **Write it, then run it.** The paper is the program from the start.
- **Do the research anywhere, then summarise it.** Explore in scripts, notebooks or a framework,
  and write the paper afterwards as the compact form of what was found.

Either way, the results block is only ever written by running the paper. Rerunning a summary
is what shows that the compact form regenerates the result. A summary that has not been run is
still a valid paper: its claims are open, and numbers in its prose are not evidence.

## 1. The file

A paper starts with a header and is otherwise prose with fenced blocks. Prose is for people
and is never evidence.

```
---
title: A character bigram predicts Shakespeare better than character frequencies
tinypaper: 0.3
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
| `python experiments` | the runs | author |
| `python claim=<id>` | a function of the results that should return true | author |
| `results` | the small values each run returned | the runner |

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
  `base.text` for a data block, `base.cpu` for an env, `r.base.bigram` for its results.
- `at` is a path relative to the citing paper. A cited paper is always a local file, so a
  vault is complete in itself and ordinary markdown tools can follow its links.
- `source` is optional: the address the file came from, for a paper from another vault. If the
  file is missing it is copied from there.
- `code` is optional: the cited paper's code hash. If present it must match. It pins the code
  and data blocks, not the whole file, because the results block changes when a paper is run.
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
- Guideline: a loader returns a dataclass defined in the paper, so the layout is stated once
  and Python enforces it.
- Data loads the first time its name is used.

### `python code=<name>`

The code blocks of a paper are executed in order in one namespace. They may import from
cited papers by their short names and from installed packages. `name` is a label.

### `python experiments`

````
```python experiments
for k in [1, 2, 4, 8]:
    run("mlp", train, d=base.text, make=window, k=k, steps=3000, seeds=2, env=base.cpu)
```
````

The block only declares runs. `run(name, fn, env=None, seeds=None, **arguments)`:

- `name` is the experiment. Calls with the same name form one sweep.
- `fn` is the function to call, with `arguments` as keywords. An argument is a literal, a
  function, a data block, or another run. A run passed as an argument is replaced by its
  output, and is run first if it has no result.
- `seeds=k` calls the function k times with `seed=1..k`, and `seeds=[0, 5]` with exactly those
  seeds. Without it the function is called once, with no seed.
- **The runner sets `seed` and `device`.** Neither may appear in the call. If the function
  has a `device` parameter the runner passes one, so a paper written for a GPU can still be
  started on a CPU.
- `run` returns the declared run (a list of them with `seeds`), so it can be passed on.

How finely to split work into runs is the author's choice. One function may pretrain and
fine-tune ten times. Splitting it lets other papers reuse the stages and lets claims select
among them.

### What a run returns

Anything that can be written down: numbers, strings, lists, dicts, dataclasses, and numpy or
torch arrays. The runner splits it by size:

- **Small values go into the paper**: numbers, strings, and short lists and dicts of them.
- **Arrays go to the logs**, whatever their size.

Guideline: return both. Return the summary numbers the claims need, so the claims can be
evaluated from the paper alone, and return the raw observations behind them (the loss on every
test example), so later papers can ask new questions of the logs. Returning model weights is
possible but rarely worth it: do it only when the trained model is itself part of the
contribution or another run takes it as an argument.

A function may report values as it runs:

```python
from tinypaper import emit
emit(step=step, loss=loss)
```

`tinypaper.source(obj)` gives the source text of a function or class defined in a paper, for a
run that needs it, for example to charge an algorithm for the length of its code.

The first ten emitted values of each run are reported in the paper, and the whole stream goes
to the logs. They let a rerun be compared with the paper before it finishes, and they let the
runner stop a run when its time is up.

Guideline: draw random numbers on the CPU. A GPU and a CPU give different random numbers for
the same seed. If a run draws its batches and initial weights on the CPU and then moves them,
a seed means the same thing on every device and early values agree closely.

### `python claim=<id>`

````
```python claim=c1 stated=2026-11-15 differs=n
def c1(r):
    return bpc(r.unigram) - bpc(r.bigram) > 1.0
```
````

A claim is a function named `<id>` that takes `r` and returns true or false. It can use the
paper's code and data.

- `r.<experiment>` is the list of that experiment's runs. `.where(k=8)` keeps the runs with
  those argument values.
- `r.<cite>.<experiment>` is a cited paper's runs.
- Each run has `.args`, `.seed`, `.emitted` and `.out`.

Header attributes:

- `stated` (required): the date the claim was made.
- `differs`: what differs between the runs whose output the claim reads. It lists argument
  names, or `fn` if the function itself differs (arguments are then not compared). The seed is
  ignored. `check` computes the real difference and fails if it is not what was declared.
- `challenges=<cite>.<claim>`: this claim disputes that claim of a cited paper.

A claim is true or false. There is no third verdict. To say "better on average but not on
every stream", make two claims: one for the average, one for every stream.

A claim's status is computed, never written:

| Status | Meaning |
|---|---|
| holds | the function returned true |
| fails | it returned false |
| open | a run it reads has no fresh result, or it needs an output that is only in the logs and the logs are not loaded |
| error | the function raised |

### `results`

````
```results
{"set": "a40-torch2.6", "ran_on": {"python": "3.11.4", "device": "cuda", "gpu": "NVIDIA A40", "packages": {"torch": "2.6.0+cu124"}}}
{"run": "58044ec27e7560e4", "experiment": "transformer", "fn": "fn:train", "args": {...}, "seed": 1, "code": "9f2c...", "seconds": 41.7, "out": {"test_bpc": 2.33, "params": 875585}, "logged": ["test_bits"], "prefix": {"step": [0, 1, ...], "loss": [6.02, ...]}, "prefix_t": 0.41}
```
````

The runner writes this block, at the end of the paper, when the author runs it. Nobody writes
it by hand. The first line names the author's record set and the machine. Each further line is
one run:

- `run`: the run's identity, a hash of its experiment name, function name, arguments and seed.
- `code`: the paper's code hash when it ran. **A result is stale** when this no longer equals
  the paper's code hash, which covers its code and data blocks and the code hashes of the
  papers it cites. A stale result is not used: claims that read it are open.
- `seconds`: how long it took.
- `out`: the small values it returned. `logged`: the names of the parts that went to the logs.
- `prefix`, `prefix_t`: the first ten emitted values, and when the last of them was emitted.

Only the author's runs are in the paper, and only runs the paper still declares. Another
person's rerun never edits the paper: it goes to the logs.

`paper.run()` does every declared run that has no fresh result; `only=[...]` limits it to some
experiments. Several processes may run the same paper at once, for example one per GPU: each
takes the next run nobody has started, and they do not overwrite each other.

## 2. Logs

A paper's logs are one JSON file, `logs/<paper>.json` beside the vault's papers, and
`<vault name>/<paper>.json` in the store. It holds:

- **record sets**: one per machine that ran the paper, by name. A record set holds each run in
  full: identity, arguments, seed, status (`done`, `stopped`, `error`), seconds, what it ran
  on, every emitted value, and the whole output with arrays written inline as lists.
- **the run log**: one entry for every run ever made and every claim whose text changed, in
  order. Whether a logged run is still in the paper is computed, not stored: it is if its
  identity matches a declared run. So the log shows how many runs were tried and dropped, and
  which claims were edited after results existed.

An array above 10 MB is written beside the logs file, in `<paper>.large/<sha256>.npy`, and the
logs refer to it by hash.

Sharing logs is optional. `paper.push()` uploads a paper's logs to the store, merged with what
is there: other people's record sets are kept. `paper.load_logs()` fetches them.

The store is set in a `tinypaper.toml` above the vault:

```toml
name = "example-vault"
store = "https://huggingface.co/datasets/org/store/resolve/main"
```

Tokens go in a `.env` file above the vault, as `KEY=VALUE` lines, or in the environment:
`HF_TOKEN` for Hugging Face, `TINYPAPER_TOKEN` for another private http store. A token is never
written to a paper or to the logs, and `.env` is not committed. There is no central server.

## 3. Evidence

How far a paper's results can be trusted depends on what stands behind them. `check` reports
which of these applies to the paper's own results:

| Evidence | Meaning |
|---|---|
| none | the paper has not been run |
| reported | the results block says so. This is the author's word, as in any paper |
| logged | the logs are loaded and agree with every reported result |
| cited | the paper has no runs: its claims rest on the results of the papers it cites |

Beyond that is a rerun: see [`reproduce`](#5-reproduce).

## 4. `check`

`paper.check()` reads a paper and reports. It runs no experiments. It fails if any of these is
wrong:

1. **Format.** Header fields, block syntax and fields, `stated` on every claim, no `seed` or
   `device` in a call, `challenges=` naming a real claim of a cited paper.
2. **Availability.** Everything needed to rerun the paper must be reachable with a matching
   hash: every cited paper (recursively), and every data file of the paper and of the papers
   it cites.
3. **`differs=`.** For each claim that evaluates, the declared difference equals the computed
   one.
4. **Errors.** No claim raises.
5. **Agreement.** Where the logs hold a run the paper reports, they must say the same.

It also reports, without failing:

- each claim's status on the paper's results, and on every other record set in the logs;
- how many declared runs have fresh, stale or missing results;
- how the machine differed from the env (package versions, device);
- claims of other papers in the vault that challenge this paper's claims;
- from the run log, when logs are loaded: runs made, versions of the code, runs no longer
  declared, claims edited after results existed;
- lines of code in the paper, its cites, its installs, and installs from git.

When a claim reads a cited paper, it reads that paper's record set of the same name if it has
one, and otherwise what that paper reports.

## 5. `reproduce`

`paper.reproduce(time="10m")` reruns a paper on the machine it is called on, within a time
limit, and reports the highest tier reached with coverage inside each tier.

| Tier | Reached when |
|---|---|
| valid | the paper parses and follows the format |
| checked | `check` passes |
| starts | every run starts on this machine: data loads, the function is called, a first value is emitted |
| prefix | the first emitted values of every compared run match the paper |
| partial | at least one claim was rerun in full |
| full | every run was rerun and every claim evaluated |
| robust | the same with seeds the paper never used (`seeds="fresh"`) |

How the time is spent:

1. Every run is taken ten emitted values. This gives `starts` and `prefix`, and measures this
   machine's speed against the recorded time.
2. The time left goes to whole claims, cheapest first. A claim's cost is the recorded seconds
   of its runs, scaled by the measured speed. A claim is started only if it is expected to fit.
3. A run that emits is stopped when the time is up. A run that never emits cannot be stopped,
   so it is started only if it is expected to fit.

The result carries the tier, the coverage, each claim's status, warnings, the measured speed,
and the seconds still needed here to rerun everything. A claim whose runs were not rerun is
"not attempted", never "fails". Calling `reproduce` again continues: finished runs are kept.

- A tier counts only if the tiers before it were reached, with one exception: a full rerun is
  stronger evidence than the tiers below it, so `full` and `robust` stand even if `checked`,
  `starts` or `prefix` did not pass. The result then carries a warning naming what did not pass.
- **Prefix matching.** If the rerun's machine and package versions equal the paper's, the
  emitted values must be equal. Otherwise arithmetic differs in the last digits and the
  difference grows with every training step, so a value must be within 0.1% of the reported
  one. Failing that, it must lie in the range of the reported seeds at that step, widened on
  each side by that range's width (1% of the value if there is one seed).
- A rerun goes to the logs as its own record set. It never changes the paper.

## Things that cannot be rerun

- **Collected data.** A paper may include the code that collected its data. It is there to be
  read. `reproduce` never reruns it: the data is taken as given, pinned by its hash.
- **Proofs.** A machine-checked proof is code: the run calls the proof checker and the claim
  is that it passes. An informal proof is prose and is not evidence.

## Not in this version

- Experiments that call external services.
- Figures. Results hold values; a reader can plot them.
- A mirror for data files. A paper's data comes from its own URLs.
