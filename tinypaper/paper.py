"""Paper: one markdown file as typed parts (see format.py), its namespace, its calls and its log."""
import importlib.metadata, inspect, json, linecache, os, platform, re, sys, time, types
from dataclasses import dataclass
from typing import Any, Optional
from pydantic import PrivateAttr
from . import format as fmt, records, store
from .verify import PaperVerify
from .format import Entry, Header, Log, Part, Prose, Strict, sha
from .records import Stop, TooLarge, encode, fits
from .reproduce import PaperReproduce
from .store import Unreachable

NAMES = {}      # id(object) -> canonical name, for data objects passed as call arguments
PAPERS = {}     # absolute path -> Paper


class Local(Strict):
    """What `reproduce` keeps beside the vault for one paper. None of it is part of the paper."""
    rerun: dict[str, Entry] = {}               # the calls a reproduction has finished on this machine


@dataclass
class Declared:
    """One call an experiments block declares, under one experiment name. It holds live functions."""
    paper: Any
    name: str                      # the experiment
    fn: Any
    kw: dict                       # the arguments as written, live: pass them to fn
    env: Any
    seed: Optional[int]
    args: dict                     # the arguments as a log entry names them

    @property
    def id(self):
        """The call's identity: its function, arguments, seed and code. The experiment name is not part of it."""
        return sha(json.dumps(["fn:" + self.fn.__name__, self.args, self.seed, self.paper.code_hash], sort_keys=True))[:16]


@dataclass
class Outcome:
    """One execution of a call: its entry if it finished, and how it ended."""
    entry: Optional[Entry]
    status: str                    # done, stopped or error
    error: Optional[str]
    emitted: list
    seconds: float
    machine: dict


class Namespace(dict):
    """A paper's module namespace. Data blocks load the first time their name is used."""

    def __init__(self, paper=None, parent=None):
        super().__init__()
        self.paper, self.parent = paper, parent

    def __missing__(self, key):
        if self.parent is not None:
            return self.parent[key]
        if self.paper is not None and key in self.paper.data:
            self[key] = self.paper.load_data(key)
            return self[key]
        raise KeyError(key)


def canonical(v):
    """A JSON value that names a call argument: literals as they are, objects by name."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [canonical(x) for x in v]
    if isinstance(v, dict):
        return {str(k): canonical(x) for k, x in v.items()}
    if id(v) in NAMES:
        return NAMES[id(v)]
    if isinstance(v, Declared):
        return "call:" + v.id
    if callable(v):
        return "fn:" + getattr(v, "__name__", type(v).__name__)
    raise TypeError(f"call argument {v!r} is not a literal, a function, a data block or a call")


def default_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def ran_on(env, device):
    """What a call actually ran on: installed versions of the env's pinned packages, and the device."""
    packages = {}
    for pin in (env.packages if env else []):
        name = pin if pin.startswith("git+") else re.split(r"[=<>@ ]", pin)[0]
        try:
            packages[name] = importlib.metadata.version(name)
        except Exception:
            packages[name] = "unknown"
    out = dict(python=platform.python_version(), device=device, packages=packages)
    if device.startswith("cuda"):
        import torch
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


class Paper(Strict, PaperVerify, PaperReproduce):
    path: str
    header: Optional[Header] = None
    parts: list[Part] = []
    problems: list[str] = []
    _ns: Any = PrivateAttr(None)
    _declared: Any = PrivateAttr(None)
    _local: Any = PrivateAttr(None)
    _text: Any = PrivateAttr(None)
    _cited: Any = PrivateAttr(default_factory=dict)
    _unreachable: Any = PrivateAttr(default_factory=list)

    # ---- the file

    @classmethod
    def load(cls, path):
        """The paper at `path`. Parsed again if the file, or a paper it cites, has changed."""
        path = os.path.abspath(path)
        old = PAPERS.get(path)
        try:
            same = old is not None and old._text == open(path).read() and all(cls.load(p.path) is p for p in old._cited.values())
        except (OSError, Unreachable):
            same = False
        if not same:
            PAPERS.pop(path, None)
            PAPERS[path] = cls.parse(open(path).read(), path)
        return PAPERS[path]

    @classmethod
    def parse(cls, text, path):
        header, parts, problems = fmt.parse(text)
        paper = cls(path=os.path.abspath(path), header=header, parts=parts, problems=problems)
        paper._text = text
        for c in paper._of("cite"):
            try:
                paper._cited[c.name] = paper._resolve(c)
            except Unreachable as e:               # reported by verify; the paper still loads
                paper._unreachable.append(str(e))
        return paper

    def render(self):
        """The paper as markdown."""
        return fmt.render(self.parts)

    def _of(self, kind):
        return [p for p in self.parts if p.kind == kind]

    stem = property(lambda self: os.path.basename(self.path)[:-3])
    dir = property(lambda self: os.path.dirname(self.path))
    cites = property(lambda self: dict(self._cited))        # {short name: the cited Paper}
    envs = property(lambda self: {e.name: e for e in self._of("env")})
    data = property(lambda self: {d.name: d for d in self._of("data")})
    code = property(lambda self: self._of("code"))
    claims = property(lambda self: self._of("claim"))
    log = property(lambda self: (self._of("log") or [None])[0])
    lines = property(lambda self: sum(len([l for l in c.source.splitlines() if l.strip()]) for c in self.code))

    def _entry_ids(self):
        """The call id of each log entry, in order. An entry written without one belongs to the
        declared call with its function, arguments and seed, or to none."""
        key = lambda fn, args, seed: (fn, json.dumps(args, sort_keys=True), seed)
        ids = {key("fn:" + d.fn.__name__, d.args, d.seed): d.id for d in self.calls()}
        return [e.call or ids.get(key(e.fn, e.args, e.seed)) for e in self.log.entries]

    @property
    def logged(self):
        """{call id: entry} for the paper's log; entries that belong to no declared call are under None."""
        if not self.log:
            return {}
        return {i: e if e.call else e.model_copy(update={"call": i}) for i, e in zip(self._entry_ids(), self.log.entries)}

    @property
    def cited_by(self):
        """The papers in this paper's vault that cite it."""
        return [p for p in self.vault.papers.values() if any(c.path == self.path for c in p.cites.values())]

    @property
    def vault(self):
        from .vault import Vault
        return Vault.of(self.dir)

    def _resolve(self, cite):
        """A cited paper: a local file. If it is missing and the cite names a source, it is copied from there.
        A `code` pin must match."""
        if re.match(r"\w+://", cite.at):
            raise Unreachable(f"cite `{cite.name}`: `at` is a local path; put the address in `source`")
        path = os.path.normpath(os.path.join(self.dir, cite.at))
        if not os.path.isfile(path) and cite.source:
            try:
                data = store.get(cite.source)
            except Exception:
                raise Unreachable(f"cited paper `{cite.name}` is not at {cite.at} and could not be fetched from {cite.source}") from None
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(data)
        if not os.path.isfile(path):
            raise Unreachable(f"cited paper `{cite.name}` at {cite.at} not found")
        cited = Paper.load(path)
        if cite.code and cited.code_hash != cite.code:
            raise Unreachable(f"cited paper `{cite.name}` at {cite.at} has code {cited.code_hash}, not the pinned {cite.code}")
        return cited

    @property
    def code_hash(self):
        """Hash of the code a call can depend on: this paper's code and data blocks, and its cites'."""
        parts = [c.source for c in self.code] + [d.model_dump_json(by_alias=True, exclude={"raw"}) for d in self.data.values()]
        parts += [a + ":" + p.code_hash for a, p in sorted(self._cited.items())]
        return sha("\n".join(parts))[:16]

    # ---- its namespace

    def view(self, alias):
        """This paper as a module, for `from <alias> import name` and `<alias>.name`."""
        ns = self.build()
        m = types.ModuleType(alias)

        def missing(name):
            try:
                return ns[name]
            except KeyError:
                raise AttributeError(f"paper {self.stem} defines no `{name}`") from None
        m.__dict__["__getattr__"] = missing
        return m

    def build(self):
        if self._ns is not None:
            return self._ns
        self._ns = ns = Namespace(self)
        views = {a: p.view(a) for a, p in self._cited.items()}
        ns.update(views)
        ns.update(self.envs)
        saved = {a: sys.modules.get(a) for a in views}
        sys.modules.update(views)
        try:
            for c in self.code:
                where = f"{self.path}:code={c.name}"
                linecache.cache[where] = (len(c.source), None, c.source.splitlines(True), where)    # so tracebacks show the line
                exec(compile(c.source, where, "exec"), ns)
        finally:
            for a, old in saved.items():
                if old is None:
                    sys.modules.pop(a, None)
                else:
                    sys.modules[a] = old
        return ns

    def load_data(self, name, verify=False):
        spec, ns = self.data[name], self.build()
        if spec.files is not None:
            obj = ns[spec.load]({k: store.data_file(f.url, f.sha256, verify) for k, f in spec.files.items()})
        else:
            obj = ns[spec.load](eval(spec.source, ns), **spec.args)
        NAMES[id(obj)] = f"data:{self.stem}/{name}"
        return obj

    def declared(self):
        """Every call the experiments blocks declare, in order, each under its experiment name.
        The same call may appear under two names; it is one call."""
        if self._declared is None:
            out, ns = [], self.build()

            def call(name, fn, env=None, seeds=None, **kw):
                if "device" in kw or "seed" in kw:
                    raise ValueError(f"call in {name}: `device` and `seed` are set when the call runs, not in the call")
                args = {k: canonical(v) for k, v in sorted(kw.items())}
                which = list(seeds) if isinstance(seeds, (list, tuple)) else range(1, seeds + 1) if seeds else [None]
                made = [Declared(self, name, fn, kw, env, s, args) for s in which]
                out.extend(made)
                return made if seeds else made[0]
            for block in self._of("experiments"):
                scope = Namespace(parent=ns)
                scope["call"] = call
                exec(compile(block.source, f"{self.path}:experiments", "exec"), scope)
            self._declared = out
        return self._declared

    def calls(self, only=None):
        """The distinct calls behind the declared experiments, in order."""
        seen = {}
        for d in self.declared():
            if not only or d.name in only:
                seen.setdefault(d.id, d)
        return list(seen.values())

    # ---- beside the vault, in logs/ and not committed: the calls a reproduction has finished here

    local_path = property(lambda self: os.path.join(self.dir, "logs", self.stem + ".json"))

    @property
    def local(self):
        if self._local is None:
            self._local = Local.model_validate_json(open(self.local_path).read()) if os.path.exists(self.local_path) else Local()
        return self._local

    def save_local(self):
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        with open(self.local_path + ".tmp", "w") as f:
            f.write(self.local.model_dump_json())
        os.replace(self.local_path + ".tmp", self.local_path)

    # ---- one call, as `reproduce` runs it

    def _execute(self, d, device, entries=None, max_emits=None, deadline=None):
        """Run one call. Nothing is stored here. `entries` is the log to take the outputs of calls
        it uses as arguments from."""
        entries, kw = self.logged if entries is None else entries, {}
        for k, v in d.kw.items():
            if isinstance(v, Declared):        # another call as an argument: its output, run first if it is not logged
                have = entries.get(v.id) or v.paper._execute(v, device, entries).entry
                v = have.out if have else None
            kw[k] = v
        if d.seed is not None:
            kw["seed"] = d.seed
        if "device" in inspect.signature(d.fn).parameters:
            kw["device"] = device
        start = time.time()
        records._current = stream = dict(stream=[], max=max_emits, deadline=deadline, start=start)
        out, status, error = None, "done", None
        try:
            out = encode(d.fn(**kw))
            fits(out, stream["stream"])
        except Stop:
            status = "stopped"
        except TooLarge as e:
            status, error = "error", str(e)
        except Exception as e:
            status, error = "error", f"{type(e).__name__}: {e}"
        finally:
            records._current = None
        seconds = round(time.time() - start, 2)
        entry = Entry(call=d.id, fn="fn:" + d.fn.__name__, args=d.args, seed=d.seed, seconds=seconds, out=out,
                      emitted=stream["stream"]) if status == "done" else None
        return Outcome(entry, status, error, stream["stream"], seconds, ran_on(d.env, device))

    def _write_log(self, log):
        parts = []
        for p in self.parts:
            if p.kind == "log":
                continue
            if p.kind == "prose" and parts and parts[-1].kind == "prose":
                parts[-1] = Prose(text=parts[-1].text + p.text)
            else:
                parts.append(p)
        if parts and parts[-1].kind == "prose":
            parts[-1] = Prose(text=parts[-1].text.rstrip("\n") + "\n\n")
        elif parts:
            parts.append(Prose(text="\n\n"))
        self.parts = parts + [log, Prose(text="\n")]
        self._text = self.render()
        with open(self.path, "w") as f:
            f.write(self._text)

    def tidy(self):
        """Rewrite the parts of the file that are derived from the rest: the line of links to the
        papers it cites, the label above each block, and the id of any log entry written without one.
        Returns whether the file changed."""
        cites = self._of("cite")
        line = "Cites: " + ", ".join(f"[{os.path.basename(c.at)[:-3]}]({c.at[2:] if c.at.startswith('./') else c.at})" for c in cites) + "\n"
        first = self.parts[0] if self.parts and self.parts[0].kind == "prose" else None
        if first is None:
            return False
        text = re.sub(r"^Cites: .*\n\n?", "", first.text, flags=re.M)
        if cites:
            m = re.match(r"---\n.*?\n---\n\n?", text, re.S)
            at = m.end() if m else 0
            text = text[:at] + line + "\n" + text[at:]
        self.parts[0] = Prose(text=text)
        self.parts = fmt.labelled(self.parts)
        if self.log and any(e.call is None for e in self.log.entries):
            try:
                ids = {id(e): i for i, e in zip(self._entry_ids(), self.log.entries)}
            except Exception:              # the paper does not build: verify reports why
                ids = {}
            self.log.entries = [e if e.call or not ids.get(id(e)) else e.model_copy(update={"call": ids[id(e)]}) for e in self.log.entries]
        if self.render() == self._text:
            return False
        self._text = self.render()
        with open(self.path, "w") as f:
            f.write(self._text)
        return True
