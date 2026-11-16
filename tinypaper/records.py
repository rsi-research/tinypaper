"""Runs as claims see them, and how a run's output is written down."""
import io, json, os, time
import numpy as np
from .model import Record, Row, sha
from .store import Unreachable, fetch

LARGE = 10 * 1024 * 1024    # arrays above this many bytes are kept beside the logs, not inside them
PREFIX = 10                 # how many emitted values a paper reports for each run

_current = None             # the run being executed, or None
_touched = None             # runs whose output a claim has read, while a claim is evaluated


class Stop(Exception):
    """Raised by `emit` when the runner's limit for this run is reached."""


class Open(Exception):
    """A claim read something that is not available: no fresh result, or an output that is only in the logs."""


def emit(**values):
    """Report values from inside a run as it goes, for example emit(step=100, loss=2.3)."""
    c = _current
    if c is None:
        return
    entry = {k: (v if isinstance(v, (int, str, bool)) else float(v)) for k, v in values.items()}
    entry["t"] = round(time.time() - c["start"], 3)
    c["stream"].append(entry)
    if (c["max"] is not None and len(c["stream"]) >= c["max"]) or (c["deadline"] is not None and time.time() > c["deadline"]):
        raise Stop()


def is_array(obj):
    return isinstance(obj, np.ndarray) or (type(obj).__module__.split(".")[0] == "torch" and hasattr(obj, "detach"))


def encode(obj, large_dir):
    """A run's return value as JSON. Arrays are written inline as lists; very large ones go to `large_dir`."""
    if is_array(obj):
        torch = not isinstance(obj, np.ndarray)
        a = obj.detach().cpu().numpy() if torch else obj
        head = dict(dtype=str(a.dtype), shape=list(a.shape), torch=torch)
        if a.nbytes > LARGE:
            buf = io.BytesIO()
            np.save(buf, a, allow_pickle=False)
            digest = sha(buf.getvalue())
            os.makedirs(large_dir, exist_ok=True)
            path = os.path.join(large_dir, digest + ".npy")
            if not os.path.exists(path):
                open(path, "wb").write(buf.getvalue())
            return {"$file": digest, **head}
        flat = a.reshape(-1)
        # the shortest text that reads back as the same 32-bit number
        values = [float(s) for s in map(str, flat)] if a.dtype == np.float32 else flat.tolist()
        return {"$array": values, **head}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): encode(v, large_dir) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v, large_dir) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        return {k: encode(getattr(obj, k), large_dir) for k in obj.__dataclass_fields__}
    raise TypeError(f"a run returned {type(obj).__name__}, which cannot be recorded")


def decode(obj, paper):
    if isinstance(obj, dict) and ("$array" in obj or "$file" in obj):
        if "$array" in obj:
            a = np.array(obj["$array"], dtype=obj["dtype"]).reshape(obj["shape"])
        else:
            a = np.load(large_path(obj["$file"], paper), allow_pickle=False)
        if obj.get("torch"):
            import torch
            return torch.from_numpy(a)
        return a
    if isinstance(obj, dict):
        return {k: decode(v, paper) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v, paper) for v in obj]
    return obj


def large_files(obj):
    if isinstance(obj, dict):
        if "$file" in obj:
            yield obj["$file"]
        elif "$array" not in obj:
            for v in obj.values():
                yield from large_files(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from large_files(v)


def large_path(digest, paper):
    """A very large array: beside the local logs, then in the store."""
    path = os.path.join(paper.large_dir, digest + ".npy")
    if os.path.exists(path):
        return path
    store = paper.vault.store
    data = fetch(store, f"{paper.vault.name}/{paper.stem}.large/{digest}.npy") if store else None
    if data is None or sha(data) != digest:
        raise Unreachable(f"array {digest[:12]} of {paper.stem} is not in the logs or the store")
    os.makedirs(paper.large_dir, exist_ok=True)
    open(path, "wb").write(data)
    return path


def is_small(v, budget=1000):
    """Does a value belong in the paper? Numbers, strings, and short lists and dicts of them do."""
    if is_array(v):
        return False
    try:
        text = json.dumps(v)
    except TypeError:
        return False
    return len(text) <= budget and '"$array"' not in text and '"$file"' not in text    # a recorded array is never small


def split(out):
    """(the small part of an output, the names of the parts left to the logs)."""
    if isinstance(out, dict):
        small = {k: v for k, v in out.items() if is_small(v)}
        return small, [k for k in out if k not in small]
    return (out, []) if is_small(out) else (None, ["*"])


def to_row(rec):
    """What the paper reports of a finished run."""
    small, logged = split(rec.out)
    first = rec.emitted[:PREFIX]
    keys = [k for k in (first[0] if first else {}) if k != "t"]
    return Row(run=rec.id, experiment=rec.experiment, fn=rec.fn, args=rec.args, seed=rec.seed, code=rec.code,
               seconds=rec.seconds, out=small, logged=logged, prefix={k: [e.get(k) for e in first] for k in keys},
               prefix_t=first[-1]["t"] if first else None)


class Partial(dict):
    """The reported part of an output. Asking for a part that is only in the logs leaves the claim open."""

    def __init__(self, small, logged, where):
        super().__init__(small or {})
        self.logged, self.where = logged, where

    def __missing__(self, key):
        if key in self.logged:
            raise Open(f"{self.where}: `{key}` is in the logs, which are not loaded")
        raise KeyError(key)


class Run:
    """One run as a claim sees it."""

    def __init__(self, paper, experiment, fn, args, seed, emitted, ran_on, seconds, out=None, row=None, record=None):
        self.paper, self.experiment, self.fn, self.args, self.seed = paper, experiment, fn, args, seed
        self.emitted, self.ran_on, self.seconds = emitted, ran_on, seconds
        self._out, self._row, self._record = out, row, record

    @property
    def id(self):
        return self._record.id if self._record else self._row.run

    @property
    def out(self):
        if _touched is not None and self not in _touched:
            _touched.append(self)
        if self._out is None:
            if self._record is not None:
                self._out = decode(self._record.out, self.paper)
            elif self._row.logged == ["*"]:
                raise Open(f"{self.paper.stem}.{self.experiment}: the output is in the logs, which are not loaded")
            elif isinstance(self._row.out, dict):
                self._out = Partial(self._row.out, self._row.logged, f"{self.paper.stem}.{self.experiment}")
            else:
                self._out = self._row.out
        return self._out


class Runs(list):
    def where(self, **kw):
        """The runs whose arguments have these values."""
        from .paper import canonical
        return Runs(x for x in self if all(x.args.get(k) == canonical(v) for k, v in kw.items()))


def prefix_of(emitted):
    first = emitted[:PREFIX]
    keys = [k for k in (first[0] if first else {}) if k != "t"]
    return {k: [e.get(k) for e in first] for k in keys}


def source_runs(paper, source):
    """{run id: Run} for one source of a paper's results, and that source's seed offset.

    The source "paper" is the results block, completed from the logs where they hold the same
    run. Any other source is a record set in the logs."""
    out = {}
    if source == "paper":
        res = paper.results
        logged = paper.logs.sets.get(res.set) if res else None
        for row in (res.rows if res else []):
            rec = logged.runs.get(row.run) if logged else None
            if rec is not None and rec.code == row.code and rec.status == "done":
                out[row.run] = Run(paper, row.experiment, row.fn, row.args, row.seed, rec.emitted, rec.ran_on, rec.seconds, record=rec)
                out[row.run]._row = row
            else:
                emitted = [dict(zip(row.prefix, v)) for v in zip(*row.prefix.values())] if row.prefix else []
                out[row.run] = Run(paper, row.experiment, row.fn, row.args, row.seed, emitted, res.ran_on, row.seconds, row=row)
        return out, 0
    rs = paper.logs.sets.get(source)
    for rec in (rs.runs.values() if rs else []):
        if rec.status == "done":
            out[rec.id] = Run(paper, rec.experiment, rec.fn, rec.args, rec.seed, rec.emitted, rec.ran_on, rec.seconds, record=rec)
    return out, (rs.seed_offset if rs else 0)


def code_of(run):
    return run._record.code if run._record else run._row.code


class Records:
    """`r` in a claim: r.<experiment> is that experiment's runs, r.<cite> is a cited paper's results."""

    def __init__(self, paper, source):
        self._paper, self._source = paper, source
        self._runs, self._offset = source_runs(paper, source)

    def __getattr__(self, key):
        paper = self._paper
        if key in paper.cites:
            cited = paper.cites[key].paper
            # a cited paper is read from the record set of the same name, else from what it reports
            return Records(cited, self._source if self._source in cited.logs.sets else "paper")
        declared = [d for d in paper.declared() if d.name == key]
        if not declared:
            raise AttributeError(f"paper {paper.stem} declares no experiment `{key}`")
        runs = Runs()
        for d in declared:
            run = self._runs.get(d.id(self._offset))
            if run is None:
                raise Open(f"{paper.stem}.{key} has no result")
            if code_of(run) != paper.code_hash:
                raise Open(f"{paper.stem}.{key} has a stale result")
            runs.append(run)
        return runs


def evaluate(paper, claim, source):
    """One claim on one source: (status, runs it read). Status is holds, fails, open or error."""
    from . import records
    from .paper import Namespace
    records._touched = touched = []
    try:
        scope = Namespace(parent=paper.build())
        exec(compile(claim.source, f"{paper.path}:claim={claim.id}", "exec"), scope)
        result = scope[claim.id](Records(paper, source))
        return ("holds" if bool(result) else "fails"), touched
    except Open as e:
        return "open: " + str(e), touched
    except Unreachable as e:
        return "open: " + str(e), touched
    except Exception as e:
        return f"error: {type(e).__name__}: {e}", touched
    finally:
        records._touched = None


def differs(runs):
    """What differs between runs: `fn` if the function does, else the argument names that do."""
    if len({x.fn for x in runs}) > 1:
        return ["fn"]
    keys = sorted({k for x in runs for k in x.args})
    return [k for k in keys if len({json.dumps(x.args.get(k), sort_keys=True) for x in runs}) > 1]
