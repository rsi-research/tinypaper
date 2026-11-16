"""Calls as claims see them, and how a call's output is written into a log."""
import json, time
from .store import Unreachable

LIMIT = 20000               # characters one call may add to a log; beyond it the call must log a summary
EARLY = 10                  # how many emitted points count as a call's first steps

_current = None             # the call being run, or None
_read = None                # calls whose output a claim has read, while a claim is evaluated


class Stop(Exception):
    """Raised by `emit` when the runner's limit for this call is reached."""


class Open(Exception):
    """A claim read a call that has no entry in the log."""


class TooLarge(Exception):
    pass


def emit(**values):
    """Log a point from inside a call as it runs, for example emit(step=100, loss=2.3)."""
    c = _current
    if c is None:
        return
    entry = {k: (v if isinstance(v, (int, str, bool)) else float(v)) for k, v in values.items()}
    entry["t"] = round(time.time() - c["start"], 3)
    c["stream"].append(entry)
    if (c["max"] is not None and len(c["stream"]) >= c["max"]) or (c["deadline"] is not None and time.time() > c["deadline"]):
        raise Stop()


def source(obj):
    """The source text of a function or class defined in a paper, for example to measure its length."""
    import ast, inspect, linecache
    if hasattr(obj, "__code__"):
        return inspect.getsource(obj)
    inside = next(v for v in vars(obj).values() if hasattr(v, "__code__"))     # a class: find it through one of its methods
    text = "".join(linecache.getlines(inside.__code__.co_filename))
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == obj.__name__:
            return ast.get_source_segment(text, node)
    raise OSError(f"the source of {obj.__name__} was not found")


def encode(obj):
    """What a call returned, as plain JSON values. Arrays become lists."""
    if type(obj).__module__.split(".")[0] in ("torch", "numpy") and hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        return {k: encode(getattr(obj, k)) for k in obj.__dataclass_fields__}
    raise TypeError(f"a call returned {type(obj).__name__}, which cannot be logged")


def fits(out, emitted):
    """Refuse a call whose log is too large: a log holds what later work needs, not raw data."""
    size = len(json.dumps(out)) + len(json.dumps(emitted))
    if size > LIMIT:
        raise TooLarge(f"this call would add {size:,} characters to the log; the limit is {LIMIT:,}. "
                       "Log a summary of what it measured, and emit fewer points.")


class Call:
    """One logged call as a claim sees it."""

    def __init__(self, paper, entry, experiment):
        self.paper, self.entry, self.experiment = paper, entry, experiment
        self.id, self.fn, self.args, self.seed = entry.call, entry.fn, entry.args, entry.seed
        self.emitted, self.seconds = entry.emitted, entry.seconds

    @property
    def out(self):
        if _read is not None and self not in _read:
            _read.append(self)
        return self.entry.out


class Calls(list):
    def where(self, **kw):
        """The calls whose arguments have these values."""
        from .paper import canonical
        return Calls(x for x in self if all(x.args.get(k) == canonical(v) for k, v in kw.items()))


class Records:
    """`r` in a claim: r.<experiment> is that experiment's calls, r.<cite> is a cited paper's.

    `entries` is the log to read this paper's calls from: the paper's own by default, or a
    rerun's. A cited paper is always read from its own log."""

    def __init__(self, paper, entries=None):
        self._paper = paper
        self._entries = paper.logged if entries is None else entries

    def __getattr__(self, key):
        paper = self._paper
        if key in paper.cites:
            return Records(paper.cites[key])
        declared = [d for d in paper.declared() if d.name == key]
        if not declared:
            raise AttributeError(f"paper {paper.stem} declares no experiment `{key}`")
        found = Calls()
        for d in declared:
            entry = self._entries.get(d.id)
            if entry is None:
                raise Open(f"{paper.stem}.{key} is not in the log")
            found.append(Call(paper, entry, key))
        return found


def evaluate(paper, claim, entries=None):
    """One claim on one log: (status, calls it read). Status is holds, fails, open or error."""
    from . import records
    from .paper import Namespace
    records._read = read = []
    try:
        scope = Namespace(parent=paper.build())
        exec(compile(claim.source, f"{paper.path}:claim={claim.id}", "exec"), scope)
        result = scope[claim.id](Records(paper, entries))
        return ("holds" if bool(result) else "fails"), read
    except Open as e:
        return "open: " + str(e), read
    except Unreachable as e:
        return "open: " + str(e), read
    except Exception as e:
        return f"error: {type(e).__name__}: {e}", read
    finally:
        records._read = None


def differ(calls):
    """What differs between calls: `fn` if the function does, else the argument names that do."""
    if len({x.fn for x in calls}) > 1:
        return ["fn"]
    keys = sorted({k for x in calls for k in x.args})
    return [k for k in keys if len({json.dumps(x.args.get(k), sort_keys=True) for x in calls}) > 1]
