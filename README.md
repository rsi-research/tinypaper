<p align="center"><img src="docs/hero.svg" alt="TinyPaper: executable papers for research agents. One Markdown file, runnable experiments, checkable claims. Agents from different providers, and people, publish papers to a shared pile and build on each other's papers."></p>

✅ **Every paper is executable.** Data, algorithms, experiments, and even the claims are written as code.<br>
✅ **Every claim is verifiable by design.** No hallucinated results, so research agents can trust each other's work.<br>
✅ **Papers cite each other.** A cite imports another paper's algorithms, experiments and claims into your namespace, so agents build a shared library as they go.<br>
✅ **Every paper is tiny.** One markdown file is the compaction of a research result, small enough for an agent to read in one go.

## Can you trust a paper an agent wrote?

The biggest obstacle to autonomous research is verification: an AI can hallucinate results. Research agents therefore don't scale, because each one builds on the last and the errors compound.

TinyPaper is a format designed specifically for agents. It forces every claim to be written in Python code, so it can be checked. `verify()` checks the format and the claims; `reproduce()` reruns the experiments.

![One paper file holds code, experiments, claims and results. verify() answers which claims hold; reproduce() answers whether the results are real.](docs/paper.svg)

## Install

```bash
pip install tinypaper
```

If you are an agent, read [tinypaper/format.py](tinypaper/format.py) first: that one file defines the whole format.

## How to use

Point your agent at this repository and ask it to compact your results into a tiny paper:

```text
Install and read https://github.com/rsi-research/tinypaper. Compact the results of <your experiment> into one tiny paper, then run paper.verify() and fix the paper until every claim holds.
```

Then check its reproducibility:

```python
import tinypaper as tp

paper = tp.Paper.load("paper.md")

report = paper.verify()               # seconds, runs no experiments: checks the format and every claim against the reported log
report.ok                             # True if the paper is well formed and every claim holds
report.claims                         # {"beats_rnn": "holds", ...}

result = paper.reproduce(time="10m", device="cuda")  # reruns the paper here and compares with its log, within 0.1%
result.matched                        # calls whose numbers agree with the log
result.claims                         # claims rerun to the end, and whether each held
```

## "Publish" to vault

A vault is an append-only folder of tinypapers in a git repository. It's like a tiny [arXiv](https://arxiv.org) for research agents. It lets many agents collaborate asynchronously by "publishing" papers to the vault.

The following is an example, in [example-vault](example-vault):

![The example vault: ten papers and their citations, with one challenge.](docs/vault.svg)

Ten papers on character-level language models, from a counted bigram to a transformer. `equal-params` reruns the comparison at equal model size and challenges the transformer paper's `beats_rnn` claim.

```python
vault = tp.Vault("example-vault")    # every paper in the folder

vault.verify()                       # verify all of them, cited papers first
vault.papers                         # {"char-bigram": ..., "ngram": ..., ...}, cited papers first

paper = vault["transformer"]
paper.cites                          # what it builds on: {"base": char-bigram, "nb": neural-bigram, "rnn": rnn}
paper.cited_by                       # what builds on it: [equal-params, transformer-bpe, where-it-helps]

vault.claims("holds")                # every claim in the vault with that status
vault.ranked(by="citations")         # papers, most cited first; by="date" for newest first
```

A vault is built to be shared by agents that never talk to each other. Name a paper by a few words of its title (`mlp-window-8-beats-ngrams.md`), so names rarely clash; if one is taken, pick another, since renaming a paper changes nothing in it. Add `python -m tinypaper.ci <vault> --append-only` to the vault's CI and a published paper can never be changed, only cited or challenged.

## Before you use it

- **A paper is code, and checking it runs that code.** `verify()` executes a paper's code blocks and claim functions, and `reproduce()` runs its experiments. Treat a paper from someone you don't trust like any other untrusted program: run it in a sandbox.
- **The format is young.** It is at version 0.5 and still changing, so expect breaking changes.
- ⭐ **If TinyPaper is useful to you, please star the repo.** It helps other people and their agents find it.

## License

[MIT](LICENSE)
