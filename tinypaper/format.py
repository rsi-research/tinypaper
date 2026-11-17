'''The TinyPaper format, version 0.5. This file is the whole definition: read it and you can write a paper.

A tiny paper is ONE markdown file: the compaction of a research result, small enough to read in
one go. It starts with a header, and holds fenced blocks. Everything outside the blocks is prose
for people and is never evidence. `EXAMPLE` below is a complete paper.

    block (the fence's first line)                      class         what it declares
    ---  ...  ---  (top of the file)                    Header        title, format version, date
    ```cite                                             Cite          another paper, under a short name
    ```env                                              Env           hardware and pinned packages
    ```data                                             Data          files by URL and sha256, or data derived by code
    ```python code=<name>                               Code          the code this paper contributes
    ```python experiments                               Experiments   the calls, declared with call(...)
    ```python claim=<id> stated=<date>                  Claim         a function of the log that returns true
    ```log                                              Log, Entry    what each call returned and emitted

cite, env and data blocks are YAML with exactly the fields of their class; an unknown field is
an error. Any other fence (```python with no attribute, ```bash, ...) is ordinary prose.
A markdown viewer hides the words after ```python, so paper.tidy() writes a label line above
each code, experiments, claim and log block (for example **Claim `larger_is_faster`**, with its
statement). Under the header it writes `Claims: ✅ 2 of 2 hold` and a `Cites:` line of links.
Do not write those lines yourself; run paper.tidy() after any change, the log included. Blocks may come in any order, except that the log is last. A paper with no log yet is valid:
its claims are open. A line starting with ``` always ends a block, so never write one inside.

Cite names, env names, data names and everything the code defines share one namespace, and
experiment names share r with cite names: keep them all distinct.

Name the file by a few words of its title, lowercase and joined by hyphens, such as
`mlp-window-8-beats-ngrams.md`. In a vault shared by many authors a short name like `mlp.md`
will already be taken; if yours is, pick another. Renaming a paper changes nothing in it.
A vault is append-only: publish by adding a file, never by changing someone else's.

Keep it tiny. Put in the paper only the code it contributes. Cite other papers for what they
already define, and install established packages for the rest.

Who does what
    The author runs the experiments, any way they like, and writes the log (see Log).
    paper.verify()     checks the format and evaluates every claim on the log. Runs no experiments,
                       but it does execute the paper's code blocks and claims: only verify papers you trust.
    paper.reproduce()  reruns the calls on this machine and compares them with the log.
'''
import hashlib, json, re
from typing import Annotated, Any, Literal, Optional, Union
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

EXAMPLE = '''---
title: A larger step size converges faster on a quadratic
tinypaper: 0.5
date: 2026-11-16
---

Gradient descent on f(x) = x^2 from x = 1, with two step sizes.

```env
name: cpu
hardware: 1 CPU core
packages: [ "numpy==1.26.4" ]
```

```python code=descent
import numpy as np
from tinypaper import emit


def descend(lr, seed, steps=50, device="cpu"):
    x = 1.0 + 0.01 * np.random.default_rng(seed).standard_normal()
    for step in range(steps):
        x = x - lr * 2 * x
        if step < 10 or step % 10 == 0:
            emit(step=step, loss=x * x)
    return dict(final_loss=float(x * x), lr=lr)
```

```python experiments
for lr in [0.01, 0.1]:
    call("sweep", descend, lr=lr, seeds=2, env=cpu)
```

```python claim=larger_is_faster stated=2026-11-16
def larger_is_faster(r):
    """After 50 steps, a step size of 0.1 reaches a lower loss than 0.01."""
    mean = lambda calls: sum(c.out["final_loss"] for c in calls) / len(calls)
    return mean(r.sweep.where(lr=0.1)) < mean(r.sweep.where(lr=0.01))
```

```log
{"ran_on": {"python": "3.11.4", "device": "cpu", "packages": {"numpy": "1.26.4"}}}
{"fn": "fn:descend", "args": {"lr": 0.01}, "seed": 1, "seconds": 0.01, "out": {"final_loss": 0.135, "lr": 0.01}, "emitted": [{"step": 0, "loss": 0.967, "t": 0.0}]}
```
'''
# (The example's log is cut short: a real one has one line for each of the four calls, with every emitted point.)

FORMAT = "0.5"
LIMIT = 20000               # characters one call may add to a log: its `out` and `emitted` as JSON


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Header(Strict):
    """The first lines of the file, between two `---` lines. All three fields are required.

        ---
        title: A larger step size converges faster on a quadratic
        tinypaper: 0.5
        date: 2026-11-16
        ---
    """
    title: str
    tinypaper: str                 # the format version
    date: str

    @field_validator("tinypaper", "date", mode="before")
    @classmethod
    def text(cls, v):
        return str(v)


class Prose(Strict):
    """The text between blocks. For people; never evidence."""
    kind: Literal["prose"] = "prose"
    text: str


class Block(Strict):
    raw: Optional[str] = None      # the block exactly as written; the paper renders back from these


class Cite(Block):
    """Another paper, under a short name. Papers in one vault (a flat folder) cite each other by file.

        ```cite
        name: nb
        at: ./neural-bigram.md
        ```

    The name then works three ways:
        in code          from nb import train        (or nb.train)
        in experiments   nb.text, nb.cpu             a data block or an env of the cited paper
        in claims        r.nb.<experiment>           the cited paper's logged calls, from its own log
    """
    kind: Literal["cite"] = "cite"
    name: str
    at: str                        # a path relative to this paper: a cited paper is always a local file
    source: Optional[str] = None   # an address to copy the file from if it is missing, when it comes from another vault
    code: Optional[str] = None     # if given, the cited paper's code hash must equal it

    @field_validator("code", mode="before")
    @classmethod
    def text(cls, v):
        return None if v is None else str(v)


class Env(Block):
    """What the calls are meant to run on. Its name is passed to call(..., env=<name>).

        ```env
        name: gpu
        hardware: 1 GPU, 24 GB
        packages: [ "torch==2.6.0", "numpy==1.26.4" ]
        ```

    Pin exact versions with `==`. verify reports a logged version that differs from the pin.
    """
    kind: Literal["env"] = "env"
    name: str
    hardware: str = ""
    packages: list[str] = []


class DataFile(Strict):
    url: str
    sha256: str


class Data(Block):
    """Data, under a name the experiments can pass to a call. Loaded the first time it is used.

    Either files to download, each pinned by its hash:

        ```data
        name: text
        load: load_text
        files:
          train:
            url: https://example.org/train.txt
            sha256: 159b8535...
        ```

      `load` names a function in this paper's code. It is called with {"train": <local path>}
      (a string) and returns the data object: anything, usually a dataclass. The url may be
      https:// or file:// with an absolute path. The file is downloaded once and its hash checked.

    Or data derived from other data by code:

        ```data
        name: tokens
        load: tokenise
        from: text
        args: { merges: 500 }
        ```

      `from` is an expression in the paper's namespace (here the data block `text`; `nb.text`
      for a cited paper's). `load` is called as tokenise(<its value>, merges=500).
    """
    kind: Literal["data"] = "data"
    name: str
    load: str
    files: Optional[dict[str, DataFile]] = None
    source: Optional[str] = Field(None, alias="from")
    args: dict[str, Any] = {}

    @model_validator(mode="after")
    def one_origin(self):
        if (self.files is None) == (self.source is None):
            raise ValueError("a data block has either `files` or `from`")
        return self


class Code(Block):
    """The code this paper contributes: ```python code=<name>, with any name.

    All code blocks of a paper run in order in one namespace, which also holds each cited
    paper (by its cite name) and each env (by its name). A result is what a function returns,
    never text it prints.

    Inside a function called by an experiment:
        from tinypaper import emit
        emit(step=step, loss=loss)      # log a point as the call runs; numbers, strings and booleans

    Draw random numbers on the CPU and then move them to the device, so a seed gives the same
    numbers on every machine and a rerun's first points agree with the log.
    """
    kind: Literal["code"] = "code"
    name: str
    source: str


class Experiments(Block):
    """The calls the paper makes: ```python experiments. Ordinary Python (loops are fine) that calls

        call(name, fn, env=None, seeds=None, **arguments)

    name        the experiment this call belongs to. An experiment is a named set of calls;
                claims select calls by it. Two experiments may contain the same call.
    fn          a function from the paper's code, or from a cited paper.
    env         an env block, by name.
    seeds       seeds=3 makes three calls with seed 1, 2, 3; seeds=[0, 5] those exact seeds.
                Each is passed to fn as `seed`. Without seeds there is one call, and fn is
                called without a `seed` argument.
    arguments   passed to fn by keyword. Each is a literal (number, string, bool, list, dict),
                a function, a data block, or another call.

    `seed` and `device` are never written in a call. If fn has a `device` parameter, whoever
    runs it chooses the device.

    call(...) returns the call (a list of calls when seeds is given). One call passed as an
    argument to another gives that function the first one's return value. Pass a single call,
    never a list:

        pre = call("pretrain", train, d=text)          # or call(..., seeds=2)[0]
        call("finetune", tune, init=pre, seeds=3)

    A call's identity is its function, arguments, seed and the paper's code (its code and data
    blocks, and those of the papers it cites). Change the code and every call is a new call.
    """
    kind: Literal["experiments"] = "experiments"
    source: str


class Claim(Block):
    """A function of the log that returns true: ```python claim=<id> stated=<YYYY-MM-DD>

        ```python claim=larger_is_faster stated=2026-11-16
        def larger_is_faster(r):
            'A step size of 0.1 reaches a lower loss than 0.01.'
            return r.sweep.where(lr=0.1)[0].out["final_loss"] < r.sweep.where(lr=0.01)[0].out["final_loss"]
        ```

    The function has the claim's id as its name and takes one argument, r. Its docstring is
    the claim in words: one sentence a reader can check against the code. verify prints it
    beside the claim's status.

        r.<experiment>              that experiment's logged calls, in the order they were declared
        r.<experiment>.where(lr=0.1)    those whose arguments, as written in the call, have these values
        r.<cite>.<experiment>       a cited paper's calls, read from that paper's own log
    Each call has .out (what the function returned), .emitted (its points, a list of dicts),
    .args, .seed and .seconds.

    A claim holds if the function returns something true, fails if not, and is open if a call
    it reads is not in the log. Its status is computed, never written. The claim can use
    anything in the paper's namespace. `stated` is the date the claim was written.

    To dispute a claim of a paper you cite, add challenges=<cite>.<claim id> to the fence line:
        ```python claim=gap_closes stated=2026-11-16 challenges=tr.beats_rnn
    It is a label: your claim is evaluated like any other, and verify on the cited paper lists
    it as a challenge. Disagreement is published as a paper, never as an edit to the original.
    """
    kind: Literal["claim"] = "claim"
    id: str
    stated: str
    challenges: Optional[str] = None
    source: str

    @property
    def hash(self):
        return sha(self.source)[:16]

    @property
    def statement(self):
        """The claim in words: the docstring of its function."""
        import ast
        try:
            return next((" ".join(ast.get_docstring(n).split()) for n in ast.parse(self.source).body
                         if isinstance(n, ast.FunctionDef) and ast.get_docstring(n)), "")
        except SyntaxError:
            return ""


class Entry(Strict):
    """One line of the log: what one call gave. See Log for how to write it."""
    call: Optional[str] = None     # the call's identity. If left out, the entry belongs to the declared call
    #                                with the same fn, args and seed, and paper.tidy() writes the id in
    fn: str                        # "fn:" + the function's name
    args: dict[str, Any]           # the call's arguments, without seed: a literal as it is, a function as "fn:<name>",
    #                                a data block as "data:<block name>@<first 8 characters of its paper's code hash>",
    #                                a call as "call:<its id>". Copy these from `d.args` (see Log); do not write them by hand
    seed: Optional[int] = None
    seconds: float = 0             # how long the call took
    out: Any = None                # what the function returned, as JSON: numbers, strings, lists, dicts
    emitted: list[dict[str, Any]] = []     # the points it emitted, in order. "t" (seconds since the call started)
    #                                        is added to each point for you and is never compared


class Log(Block):
    """The last block of the paper, written by the author: one JSON object per line.

        ```log
        {"ran_on": {"python": "3.11.4", "device": "cuda", "gpu": "NVIDIA A40", "packages": {"torch": "2.6.0+cu124"}}}
        {"fn": "fn:descend", "args": {"lr": 0.1}, "seed": 1, "seconds": 0.01, "out": {...}, "emitted": [{"step": 0, "loss": 0.64, "t": 0.0}, ...]}
        ```

    The first line describes the machine. Each further line is an Entry, one per call.

    TinyPaper does not run your experiments. Run the functions any way you like, on any
    machines, then write the entries. To get the calls and collect what they emit:

        import json, time, tinypaper as tp
        outs = {}
        for d in tp.Paper.load("my-paper.md").calls():     # each declared call, in the order declared: d.fn, d.kw, d.seed, d.args, d.id
            kw = {k: outs[v.id] if hasattr(v, "id") else v for k, v in d.kw.items()}    # an argument that is a call: its output
            if d.seed is not None:
                kw["seed"] = d.seed                        # (add device=... too if fn takes it)
            start = time.time()
            with tp.capture() as points:                   # collects what the function emits
                outs[d.id] = out = d.fn(**kw)
            print(json.dumps(dict(call=d.id, fn="fn:" + d.fn.__name__, args=d.args, seed=d.seed,
                                  seconds=round(time.time() - start, 2), out=out, emitted=points)))

    `out` and emitted values must be plain JSON: turn arrays and numpy scalars into lists and floats.

    Rules
      - An entry is exactly what the paper's function gives for those arguments and that seed.
        reproduce calls the function itself and compares the emitted points; a value agrees if
        it is within 0.1% of the logged one. Do not round what you log.
      - An entry that belongs to no declared call is an error.
      - One call may add at most 20,000 characters. Log a summary, never raw arrays.
      - A code change makes every call a new call: old entries stop counting and claims become open.

    What to log (it is all a later paper has unless it reruns yours)
      - every number your own claims read;
      - the standard metric of the papers you cite, computed by their function, so results compare;
      - a curve at a few points: the first ten steps (reproduce compares those), then sparsely, then the end;
      - what the result cost: model size, and anything a fair comparison needs;
      - a summary where a distribution matters, such as a few quantiles.
    """
    kind: Literal["log"] = "log"
    ran_on: dict[str, Any] = {}
    entries: list[Entry] = []


Part = Annotated[Union[Prose, Cite, Env, Data, Code, Experiments, Claim, Log], Field(discriminator="kind")]

# ---- markdown to parts, and back

FENCE = re.compile(r"^```([^\n`]+)\n(.*?)^```[ \t]*$", re.S | re.M)
KINDS = {"cite", "env", "data", "python", "log"}       # the first word of a fence line that makes it a block


def parse_block(kind, attrs, body, raw):
    """One block, from its fence line (`kind` and the words after it) and its body."""
    if kind in ("cite", "env", "data"):
        return {"cite": Cite, "env": Env, "data": Data}[kind].model_validate({**yaml.safe_load(body), "raw": raw})
    if kind == "log":
        lines = [json.loads(l) for l in body.splitlines() if l.strip()]
        return Log(ran_on=lines[0].get("ran_on", {}), entries=[Entry.model_validate(l) for l in lines[1:]])
    if attrs == ["experiments"]:
        return Experiments(source=body, raw=raw)
    a = dict(x.split("=", 1) if "=" in x else (x, "") for x in attrs)
    if set(a) == {"code"}:
        return Code(name=a["code"], source=body, raw=raw)
    if "claim" in a:
        if set(a) - {"claim", "stated", "challenges"}:
            raise ValueError(f"claim {a['claim']} has an unknown attribute: {', '.join(sorted(set(a) - {'claim', 'stated', 'challenges'}))}")
        if "stated" not in a:
            raise ValueError(f"claim {a['claim']} has no `stated` date")
        return Claim(id=a["claim"], stated=a["stated"], source=body, raw=raw, challenges=a.get("challenges"))
    if not attrs:
        return None                                # a plain ```python fence is prose
    raise ValueError("a python block is `code=<name>`, `experiments` or `claim=<id>`")


def parse(text):
    """A paper's text as (header, parts, problems). Parts are prose and blocks in file order."""
    parts, problems, pos = [], [], 0
    for m in FENCE.finditer(text):
        head = m.group(1).split()
        if head[0] not in KINDS:
            continue                               # an ordinary code fence in the prose
        line = text[:m.start()].count("\n") + 1
        try:
            block = parse_block(head[0], head[1:], m.group(2), m.group(0))
        except ValidationError as e:
            problems.append(f"line {line}: {head[0]} block: " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()))
            continue
        except Exception as e:
            problems.append(f"line {line}: {head[0]} block: {e}")
            continue
        if block is None:
            continue
        if text[pos:m.start()]:
            parts.append(Prose(text=text[pos:m.start()]))
        parts.append(block)
        pos = m.end()
    if text[pos:]:
        parts.append(Prose(text=text[pos:]))
    header = None
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    try:
        header = Header.model_validate(yaml.safe_load(m.group(1)) if m else {})
    except ValidationError as e:
        problems += [f"header has no `{x['loc'][0]}`" for x in e.errors()]
    return header, parts, problems


def log_text(log):
    return "```log\n" + "\n".join([json.dumps(dict(ran_on=log.ran_on))] + [e.model_dump_json(exclude_defaults=True) for e in log.entries]) + "\n```"


def render(parts):
    """Parts back to markdown: prose and blocks as written, and the log one JSON line per call."""
    return "".join(p.text if p.kind == "prose" else log_text(p) if p.kind == "log" else p.raw for p in parts)


LABEL = re.compile(r"(?:^|(?<=\n))\*\*(?:Code|Experiments|Claim|Log)\b[^\n]*\n\n?\Z")


def label(part):
    """The line `tidy` writes above a block, so a markdown viewer shows what kind of block it is."""
    if part.kind == "code":
        return f"**Code `{part.name}`**"
    if part.kind == "experiments":
        return "**Experiments**"
    if part.kind == "claim":
        return f"**Claim `{part.id}`** (stated {part.stated}" + (f", challenges `{part.challenges}`" if part.challenges else "") + ")" \
            + (": " + part.statement if part.statement else "")
    if part.kind == "log":
        return "**Log**"


def labelled(parts):
    """The parts with every label line brought up to date."""
    out = []
    for p in parts:
        want = None if p.kind == "prose" else label(p)
        if want:
            before = LABEL.sub("", out.pop().text if out and out[-1].kind == "prose" else "").rstrip("\n")
            before = before + "\n\n" if before or out else ""
            out.append(Prose(text=before + want + "\n\n"))
        out.append(p)
    return out
