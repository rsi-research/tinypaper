"""Paper: one markdown file as a tree of typed parts. Parsing, rendering, its namespace and its calls."""
import fcntl, importlib.metadata, inspect, json, linecache, os, platform, re, socket, sys, time, types
from contextlib import contextmanager
from typing import Annotated, Any, Optional, Union
import yaml
from pydantic import Field, PrivateAttr, ValidationError
from . import records, store
from .check import PaperChecks
from .model import (Cite, Claim, Code, Data, Declared, Entry, Env, Experiments, Header, Local, LocalLog, Log, Prose, Strict, sha)
from .records import Stop, TooLarge, encode, fits
from .reproduce import PaperReproduce
from .store import Unreachable

FENCE = re.compile(r"^```([^\n`]+)\n(.*?)^```[ \t]*$", re.S | re.M)
KINDS = {"cite", "env", "data", "python", "log"}
NAMES = {}      # id(object) -> canonical name, for data objects passed as call arguments
PAPERS = {}     # absolute path -> Paper
Part = Annotated[Union[Prose, Cite, Env, Data, Code, Experiments, Claim, Log], Field(discriminator="kind")]


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
        return "call:" + v.id()
    if callable(v):
        return "fn:" + getattr(v, "__name__", type(v).__name__)
    raise TypeError(f"call argument {v!r} is not a literal, a function, a data block or a call")


def block_text(part):
    """A block as markdown, written from its fields."""
    if part.kind == "log":
        meta = dict(log=part.name, ran_on=part.ran_on, history=part.history)
        return "```log\n" + "\n".join([json.dumps(meta)] + [e.model_dump_json(exclude_defaults=True) for e in part.entries]) + "\n```"
    if part.kind == "code":
        return f"```python code={part.name}\n{part.source}```"
    if part.kind == "experiments":
        return f"```python experiments\n{part.source}```"
    if part.kind == "claim":
        head = f"claim={part.id} stated={part.stated}" + (f" differs={','.join(part.differs)}" if part.differs is not None else "")
        return f"```python {head}" + (f" challenges={part.challenges}" if part.challenges else "") + f"\n{part.source}```"
    fields = part.model_dump(by_alias=True, exclude={"raw", "kind"}, exclude_defaults=True)
    return f"```{part.kind}\n" + yaml.safe_dump(fields, sort_keys=False) + "```"


def parse_block(kind, attrs, body, raw):
    if kind in ("cite", "env", "data"):
        return {"cite": Cite, "env": Env, "data": Data}[kind].model_validate({**yaml.safe_load(body), "raw": raw})
    if kind == "log":
        lines = [json.loads(l) for l in body.splitlines() if l.strip()]
        return Log(name=lines[0]["log"], ran_on=lines[0].get("ran_on", {}), history=lines[0].get("history", {}),
                   entries=[Entry.model_validate(l) for l in lines[1:]])
    if attrs and attrs[0] == "experiments":
        return Experiments(source=body, raw=raw)
    a = dict(x.split("=", 1) if "=" in x else (x, "") for x in attrs)
    if "code" in a:
        return Code(name=a["code"], source=body, raw=raw)
    if "claim" in a:
        if "stated" not in a:
            raise ValueError(f"claim {a['claim']} has no `stated` date")
        return Claim(id=a["claim"], stated=a["stated"], source=body, raw=raw, challenges=a.get("challenges"),
                     differs=sorted(x for x in a["differs"].split(",") if x) if "differs" in a else None)
    raise ValueError("a python block is `code=<name>`, `experiments` or `claim=<id>`")


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


def merged(machine, more):
    """One description of a machine from the descriptions its calls gave."""
    return {**machine, **more, "packages": {**machine.get("packages", {}), **more.get("packages", {})}}


class Paper(Strict, PaperChecks, PaperReproduce):
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
        parts, problems, pos = [], [], 0
        for m in FENCE.finditer(text):
            head = m.group(1).split()
            if head[0] not in KINDS:
                continue                              # an ordinary code fence in the prose
            line = text[:m.start()].count("\n") + 1
            try:
                block = parse_block(head[0], head[1:], m.group(2), m.group(0))
            except ValidationError as e:
                problems.append(f"line {line}: {head[0]} block: " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()))
                continue
            except Exception as e:
                problems.append(f"line {line}: {head[0]} block: {e}")
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
        paper = cls(path=os.path.abspath(path), header=header, parts=parts, problems=problems)
        paper._text = text
        for c in paper._of("cite"):
            try:
                paper._cited[c.name] = paper._resolve(c)
            except Unreachable as e:               # reported by check; the paper still loads
                paper._unreachable.append(str(e))
        return paper

    def render(self):
        """The paper as markdown. For a paper read from a file this is the file, with the log block rewritten."""
        return "".join(p.text if p.kind == "prose" else (p.raw if p.raw is not None else block_text(p)) for p in self.parts)

    def dump(self):
        """The paper as plain data. `Paper.from_dump` reads it back."""
        out = self.model_dump(mode="json", by_alias=True)
        out["path"] = os.path.basename(self.path)
        return out

    @classmethod
    def from_dump(cls, data, folder="."):
        data = dict(data)
        data["path"] = os.path.join(os.path.abspath(folder), data["path"])
        paper = cls.model_validate(data)
        paper._text = paper.render()
        return paper

    def _of(self, kind):
        return [p for p in self.parts if p.kind == kind]

    stem = property(lambda self: os.path.basename(self.path)[:-3])
    dir = property(lambda self: os.path.dirname(self.path))
    cites = property(lambda self: {c.name: _Cited(c, self) for c in self._of("cite")})
    envs = property(lambda self: {e.name: e for e in self._of("env")})
    data = property(lambda self: {d.name: d for d in self._of("data")})
    code = property(lambda self: self._of("code"))
    claims = property(lambda self: self._of("claim"))
    log = property(lambda self: (self._of("log") or [None])[0])
    lines = property(lambda self: sum(len([l for l in c.source.splitlines() if l.strip()]) for c in self.code))

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

            def run(name, fn, env=None, seeds=None, **kw):
                if "device" in kw or "seed" in kw:
                    raise ValueError(f"run {name}: `device` and `seed` are set by the runner, not in the call")
                args = {k: canonical(v) for k, v in sorted(kw.items())}
                which = list(seeds) if isinstance(seeds, (list, tuple)) else range(1, seeds + 1) if seeds else [None]
                made = [Declared(self, name, fn, kw, env, s, args) for s in which]
                out.extend(made)
                return made if seeds else made[0]
            for block in self._of("experiments"):
                scope = Namespace(parent=ns)
                scope["run"] = run
                exec(compile(block.source, f"{self.path}:experiments", "exec"), scope)
            self._declared = out
        return self._declared

    def calls(self, only=None):
        """The distinct calls behind the declared experiments, in order."""
        seen = {}
        for d in self.declared():
            if not only or d.name in only:
                seen.setdefault(d.id(), d)
        return list(seen.values())

    # ---- logs: the author's is in the file; reruns and the history are kept beside the vault, in logs/

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

    @contextmanager
    def locked(self):
        """Hold this paper's lock and see what other processes have written. Several processes may
        run one paper at once; each reads and writes the log and the local file only inside this."""
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        with open(self.local_path + ".lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                self._local = None
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def entries(self, log=None):
        """({call identity: entry}, seed offset) for one log: the author's (None) or a local rerun by name."""
        if log is None:
            return ({e.call: e for e in self.log.entries} if self.log else {}), 0
        kept = self.local.logs.get(log)
        return (dict(kept.entries), kept.seed_offset) if kept else ({}, 0)

    def _disk_log(self):
        """The log block as it is in the file now, which another process may have just written."""
        m = re.search(r"^```log\n(.*?)^```[ \t]*$", open(self.path).read(), re.S | re.M)
        return parse_block("log", [], m.group(1), m.group(0)) if m else None

    def _note(self, **event):
        self.local.history.append(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), **event))

    def history_counts(self):
        h = self.local.history
        return dict(runs=sum(e["event"] == "run" for e in h),
                    claims_edited=sorted({e["claim"] for e in h if e["event"] == "claim_edit" and e["after_results"] and not e["first"]}))

    def note_claim_edits(self):
        """Note in the history any claim whose text changed since it was last noted. Call inside the lock."""
        last = {}
        for e in self.local.history:
            if e["event"] == "claim_edit":
                last[e["claim"]] = e["hash"]
        after = any(e["event"] == "run" and e["status"] == "done" for e in self.local.history)
        for c in self.claims:
            if last.get(c.id) != c.hash:
                self._note(event="claim_edit", claim=c.id, hash=c.hash, first=c.id not in last, after_results=after)

    # ---- running it

    def execute(self, d, device, max_emits=None, deadline=None, offset=0, log=None):
        """Run one call and return (its entry, the machine it ran on). Nothing is stored here."""
        call = {}
        for k, v in d.kw.items():
            if isinstance(v, Declared):        # another call as an argument: its output, run first if it is not logged
                have = v.paper.entries(log)[0].get(v.id(offset))
                v = (have or v.paper.execute(v, device, offset=offset, log=log)[0]).out
            call[k] = v
        seed = None if d.seed is None else d.seed + offset
        if seed is not None:
            call["seed"] = seed
        if "device" in inspect.signature(d.fn).parameters:
            call["device"] = device
        start = time.time()
        records._current = stream = dict(stream=[], max=max_emits, deadline=deadline, start=start)
        out, status, error = None, "done", None
        try:
            out = encode(d.fn(**call))
            fits(out, stream["stream"])
        except Stop:
            status = "stopped"
        except TooLarge as e:
            out, status, error = None, "error", str(e)
        except Exception as e:
            status, error = "error", f"{type(e).__name__}: {e}"
        finally:
            records._current = None
        entry = Entry(call=d.id(offset), fn="fn:" + d.fn.__name__, args=d.args, seed=seed, seconds=round(time.time() - start, 2),
                      out=out, emitted=stream["stream"], status=status, error=error)
        return entry, ran_on(d.env, device)

    def _taken(self, d):
        """Is another live process already running this call?"""
        who = self.local.running.get(d.id())
        if not who:
            return False
        host, _, pid = who["owner"].rpartition(":")
        if host == socket.gethostname():
            try:
                os.kill(int(pid), 0)
            except (OSError, ValueError):
                return False                       # that process is gone
            return int(pid) != os.getpid()
        return time.time() - who["started"] < 24 * 3600

    def run(self, name="author", device=None, deep=False, rerun=False, only=None, say=print):
        """Run every declared call that is not in the author's log, and write the log into the paper.

        The paper file is rewritten: its log block holds what each call returned and emitted. `name`
        names the machine or occasion. With deep=True the cited papers run first. `only` is a list
        of experiment names to run. Several processes may call this at once, for example one per
        GPU: each takes the next call nobody has started."""
        device, done = device or default_device(), []
        self.tidy()
        if deep:
            for p in self._cited.values():
                done += p.run(name, device, True, rerun, None, say)
        with self.locked():
            self.note_claim_edits()
            self.save_local()
        for d in self.calls(only):
            with self.locked():                    # take the next call nobody has done or is doing
                old = self._disk_log()
                have = {e.call for e in old.entries} if old and old.name == name else set()
                if (d.id() in have and not rerun) or self._taken(d):
                    continue
                self.local.running[d.id()] = dict(owner=f"{socket.gethostname()}:{os.getpid()}", started=time.time())
                self.save_local()
            entry, machine = self.execute(d, device)
            done.append(entry)
            say(f"{self.stem}: {d.name} {json.dumps({k: v for k, v in d.args.items() if not str(v).startswith(('fn:', 'data:'))})}"
                f" seed={d.seed} {entry.status} in {entry.seconds}s" + (f" ({entry.error})" if entry.error else ""))
            with self.locked():                    # add it to whatever the log holds now
                self.local.running.pop(d.id(), None)
                self._note(event="run", log=name, experiment=d.name, call=entry.call, args=d.args, seed=d.seed,
                           status=entry.status, seconds=entry.seconds)
                self.save_local()
                if entry.status == "done":
                    old = self._disk_log()
                    same = old is not None and old.name == name
                    kept = {e.call: e for e in old.entries} if same else {}
                    kept[entry.call] = entry
                    order = [x.id() for x in self.calls()]                         # only calls the paper still makes
                    self._write_log(Log(name=name, ran_on=merged(old.ran_on if same else {}, machine), history=self.history_counts(),
                                        entries=[kept[i] for i in order if i in kept]))
        return done

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

    def keep(self, name, entry, machine, offset=0):
        """Store a rerun's entry in the local log `name`, and note it in the history."""
        with self.locked():
            kept = self.local.logs.setdefault(name, LocalLog(seed_offset=offset))
            kept.entries[entry.call] = entry
            kept.ran_on = merged(kept.ran_on, machine)
            self._note(event="run", log=name, call=entry.call, args=entry.args, seed=entry.seed, status=entry.status, seconds=entry.seconds)
            self.save_local()

    def tidy(self):
        """Rewrite the parts of the file that are derived from the rest: the line of links to the
        papers it cites. Returns whether the file changed."""
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
        if text == first.text:
            return False
        self.parts[0] = Prose(text=text)
        self._text = self.render()
        with open(self.path, "w") as f:
            f.write(self._text)
        return True


class _Cited:
    """A cite together with the paper it resolved to."""

    def __init__(self, cite, owner):
        self.cite, self.name, self.at = cite, cite.name, cite.at
        self.paper = owner._cited.get(cite.name)
