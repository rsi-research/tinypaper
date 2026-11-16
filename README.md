# TinyPaper

**The paper format for research agents.**

- ✅ **Every paper is code.** Data, algorithms, experiments, and even the claims.
- ✅ **Every claim is verifiable by design.** No hallucinated results, so research agents can trust each other's work.
- ✅ **Papers cite each other.** A cite imports another paper's algorithms, experiments and claims into your namespace, so agents build a shared library as they go.
- ✅ **Every paper is tiny.** One markdown file is the compaction of a research result, small enough for an agent to read in one go.

## Can you trust a paper a machine wrote?

Research agents produce results faster than anyone can read them. A prose report can be wrong, and an agent that builds on it inherits the error.

TinyPaper forces every claim to be written in code, so it can be checked. `verify()` checks the format and the claims; `reproduce()` reruns the experiments.

![One paper file holds code, experiments, claims and a log. verify() answers which claims hold; reproduce() answers whether the log is real.](docs/paper.svg)

## Install

```bash
git clone https://github.com/rsi-research/tinypaper && pip install -e tinypaper
```

Working with an agent? Point it at [tinypaper/format.py](tinypaper/format.py): that one file defines the format. The [example vault](example-vault) has ten papers to learn from.

## Use

```python
import tinypaper as tp

paper = tp.Paper.load("example-vault/transformer.md")

report = paper.verify()               # seconds, runs nothing: checks the format and every claim against the reported log
report.ok                             # True if the paper is well formed and every claim holds
report.claims                         # {"beats_rnn": "holds", ...}

result = paper.reproduce(time="10m", device="cuda")  # reruns the paper here and compares with its log, within 0.1%
result.matched                        # calls whose numbers agree with the log
result.claims                         # claims rerun to the end, and whether each held
```

TinyPaper does not run your experiments. Run them any way you like, on any machines, and write what each call gave into the paper's log.

## Vaults

A vault is an append-only folder of papers in a git repository. It's like a tiny arXiv. It lets many agents work on one problem without piling unchecked work on unchecked work: a paper builds only on claims it can verify, and a disagreement is itself a paper.

![The example vault: ten papers and their citations, with one challenge.](docs/vault.svg)

This is the [example vault](example-vault): character-level language models, from a counted bigram to a transformer. `equal-params` reruns the comparison at equal model size and challenges the transformer paper's `beats_rnn` claim.

```python
vault = tp.Vault("example-vault")    # every paper in the folder

vault.verify()                       # verify all of them, cited papers first
vault.papers                         # {"char-bigram": ..., "ngram": ..., ...}, cited papers first

paper = vault["transformer"]
paper.cites                          # what it builds on: {"base": char-bigram, "nb": neural-bigram, "rnn": rnn}
paper.cited_by                       # what builds on it: [equal-params, transformer-bpe, where-it-helps]

vault.claims("holds")                # every claim in the vault with that status
```

## Status

Early. The format is at version 0.5, and [tinypaper/format.py](tinypaper/format.py) is its whole definition. Next: `verify` and `reproduce` as agent tools, and a set of vision papers.
