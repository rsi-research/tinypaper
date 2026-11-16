"""Paper: one markdown file as a tree of typed parts. Parsing, rendering, its namespace and its runs."""
import importlib.metadata, inspect, json, os, platform, re, sys, time, types
from typing import Annotated, Any, Optional, Union
import yaml
from pydantic import Field, PrivateAttr, ValidationError
from . import records, store
from .check import PaperChecks
from .model import (Cite, Claim, Code, Data, Declared, Env, Experiments, Header, Logs, Prose, Record, RecordSet, Results,
                    Row, Strict, sha)
from .records import Stop, decode, encode, to_row
from .reproduce import PaperReproduce
from .store import Unreachable

FENCE = re.compile(r"^```([^\n`]+)\n(.*?)^```[ \t]*$", re.S | re.M)
KINDS = {"cite", "env", "data", "python", "results"}
NAMES = {}      # id(object) -> canonical name, for data objects passed as run arguments
PAPERS = {}     # absolute path -> Paper
Part = Annotated[Union[Prose, Cite, Env, Data, Code, Experiments, Claim, Results], Field(discriminator="kind")]


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
    """A JSON value that names a run argument: literals as they are, objects by name."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [canonical(x) for x in v]
    if isinstance(v, dict):
        return {str(k): canonical(x) for k, x in v.items()}
    if id(v) in NAMES:
        return NAMES[id(v)]
    if isinstance(v, Declared):
        return "run:" + v.id()
    if callable(v):
        return "fn:" + getattr(v, "__name__", type(v).__name__)
    raise TypeError(f"run argument {v!r} is not a literal, a function, a data block or a run")


def block_text(part):
    """A block as markdown, written from its fields."""
    if part.kind == "results":
        lines = [json.dumps(dict(set=part.set, ran_on=part.ran_on))] + [r.model_dump_json(exclude_defaults=True) for r in part.rows]
        return "```results\n" + "\n".join(lines) + "\n```"
    if part.kind == "code":
        return f"```python code={part.name}\n{part.source}```"
    if part.kind == "experiments":
        return f"```python experiments\n{part.source}```"
    if part.kind == "claim":
        head = f"claim={part.id} stated={part.stated}" + (f" differs={','.join(part.differs)}" if part.differs else "")
        return f"```python {head}" + (f" challenges={part.challenges}" if part.challenges else "") + f"\n{part.source}```"
    fields = part.model_dump(by_alias=True, exclude={"raw", "kind"}, exclude_defaults=True)
    return f"```{part.kind}\n" + yaml.safe_dump(fields, sort_keys=False) + "```"


def parse_block(kind, attrs, body, raw):
    if kind in ("cite", "env", "data"):
        return {"cite": Cite, "env": Env, "data": Data}[kind].model_validate({**yaml.safe_load(body), "raw": raw})
    if kind == "results":
        lines = [json.loads(l) for l in body.splitlines() if l.strip()]
        return Results(set=lines[0]["set"], ran_on=lines[0]["ran_on"], rows=[Row.model_validate(l) for l in lines[1:]])
    if attrs and attrs[0] == "experiments":
        return Experiments(source=body, raw=raw)
    a = dict(x.split("=", 1) for x in attrs)
    if "code" in a:
        return Code(name=a["code"], source=body, raw=raw)
    if "claim" in a:
        if "stated" not in a:
            raise ValueError(f"claim {a['claim']} has no `stated` date")
        return Claim(id=a["claim"], stated=a["stated"], source=body, raw=raw, challenges=a.get("challenges"),
                     differs=sorted(x for x in a.get("differs", "").split(",") if x))
    raise ValueError("a python block is `code=<name>`, `experiments` or `claim=<id>`")


def default_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def ran_on(env, device):
    """What a run actually ran on: installed versions of the env's pinned packages, and the device."""
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


class Paper(Strict, PaperChecks, PaperReproduce):
    path: str
    header: Optional[Header] = None
    parts: list[Part] = []
    problems: list[str] = []
    _ns: Any = PrivateAttr(None)
    _declared: Any = PrivateAttr(None)
    _logs: Any = PrivateAttr(None)
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
        """The paper as markdown. For a paper read from a file this is the file, with the results block rewritten."""
        return "".join(p.text if p.kind == "prose" else (p.raw if p.raw is not None else block_text(p)) for p in self.parts)

    def dump(self, logs=False):
        """The paper as plain data, optionally with its logs. `Paper.from_dump` reads it back."""
        out = self.model_dump(mode="json", by_alias=True)
        out["path"] = os.path.basename(self.path)
        if logs:
            out["logs"] = self.logs.model_dump(mode="json")
        return out

    @classmethod
    def from_dump(cls, data, folder="."):
        data = dict(data)
        logs = data.pop("logs", None)
        data["path"] = os.path.join(os.path.abspath(folder), data["path"])
        paper = cls.model_validate(data)
        paper._text = paper.render()
        if logs is not None:
            paper._logs = Logs.model_validate(logs)
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
    results = property(lambda self: (self._of("results") or [None])[0])
    lines = property(lambda self: sum(len([l for l in c.source.splitlines() if l.strip()]) for c in self.code))

    @property
    def vault(self):
        from .vault import Vault
        return Vault.of(self.dir)

    def _resolve(self, cite):
        """A cited paper: a path relative to this paper, or an https URL. A `code` pin must match."""
        if re.match(r"https?://", cite.at):
            path = os.path.join(store.CACHE, "papers", sha(cite.at)[:12], os.path.basename(cite.at))
            try:
                data = store.get(cite.at)
            except Exception:
                raise Unreachable(f"cited paper `{cite.name}` at {cite.at} not found") from None
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(data)
        else:
            path = os.path.normpath(os.path.join(self.dir, cite.at))
        if not os.path.isfile(path):
            raise Unreachable(f"cited paper `{cite.name}` at {cite.at} not found")
        cited = Paper.load(path)
        if cite.code and cited.code_hash != cite.code:
            raise Unreachable(f"cited paper `{cite.name}` at {cite.at} has code {cited.code_hash}, not the pinned {cite.code}")
        return cited

    @property
    def code_hash(self):
        """Hash of what a run's result can depend on: this paper's code and data blocks, and its cites'."""
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
                exec(compile(c.source, f"{self.path}:code={c.name}", "exec"), ns)
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
        """Every run the experiments blocks declare, in order."""
        if self._declared is None:
            out, ns = [], self.build()

            def run(name, fn, env=None, seeds=None, **kw):
                if "device" in kw or "seed" in kw:
                    raise ValueError(f"run {name}: `device` and `seed` are set by the runner, not in the call")
                args = {k: canonical(v) for k, v in sorted(kw.items())}
                made = [Declared(self, name, fn, kw, env, s, args) for s in (range(1, seeds + 1) if seeds else [None])]
                out.extend(made)
                return made if seeds else made[0]
            for block in self._of("experiments"):
                scope = Namespace(parent=ns)
                scope["run"] = run
                exec(compile(block.source, f"{self.path}:experiments", "exec"), scope)
            self._declared = out
        return self._declared

    # ---- its logs: kept beside the vault's papers, in logs/, and optionally in the store

    logs_path = property(lambda self: os.path.join(self.dir, "logs", self.stem + ".json"))
    large_dir = property(lambda self: os.path.join(self.dir, "logs", self.stem + ".large"))

    @property
    def logs(self):
        if self._logs is None:
            self._logs = Logs.model_validate_json(open(self.logs_path).read()) if os.path.exists(self.logs_path) else Logs()
        return self._logs

    def save_logs(self):
        os.makedirs(os.path.dirname(self.logs_path), exist_ok=True)
        with open(self.logs_path, "w") as f:
            f.write(self.logs.model_dump_json())

    def load_logs(self, deep=True):
        """Fetch this paper's logs from the store and add them to the local ones. Returns whether any were found.

        With deep=True the logs of the papers it cites are fetched too."""
        found = False
        if deep:
            for p in self._cited.values():
                found |= p.load_logs(True)
        where = self.vault.store
        data = store.fetch(where, f"{self.vault.name}/{self.stem}.json") if where else None
        if data is not None:
            self._logs = store.merge(Logs.model_validate_json(data), self.logs)
            self.save_logs()
            found = True
        return found

    def push(self, deep=False, tries=5):
        """Upload this paper's logs to the store, merged with what is there. Returns the store path written."""
        where = self.vault.store
        if not where:
            raise ValueError("no store is configured: set `store` in tinypaper.toml")
        if deep:
            for p in self._cited.values():
                p.push(True, tries)
        rel = f"{self.vault.name}/{self.stem}.json"
        for _ in range(tries):
            commit = store.head(where)
            there = store.fetch(store.at_commit(where, commit), rel)
            merged = store.merge(Logs.model_validate_json(there), self.logs) if there else self.logs
            files = {rel: merged.model_dump_json().encode()}
            for rs in merged.sets.values():
                for rec in rs.runs.values():
                    for digest in records.large_files(rec.out):
                        local = os.path.join(self.large_dir, digest + ".npy")
                        if os.path.exists(local):
                            files[f"{self.vault.name}/{self.stem}.large/{digest}.npy"] = open(local, "rb").read()
            if store.upload(where, files, f"{self.vault.name}/{self.stem}: logs", expect=commit):
                self._logs = merged
                self.save_logs()
                return rel
        raise RuntimeError(f"the store kept changing; gave up after {tries} tries")

    # ---- running it

    def execute(self, d, name, device, max_emits=None, deadline=None, offset=0):
        """Run one declared run into record set `name` of the logs and return its record."""
        call = {}
        for k, v in d.kw.items():
            if isinstance(v, Declared):        # another run as an argument: its output, run first if needed
                rec = v.paper.fresh(v, name, offset) or v.paper.execute(v, name, device, offset=offset)
                v = decode(rec.out, v.paper)
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
            out = d.fn(**call)
        except Stop:
            status = "stopped"
        except Exception as e:
            status, error = "error", f"{type(e).__name__}: {e}"
        finally:
            records._current = None
        rec = Record(id=d.id(offset), experiment=d.name, fn="fn:" + d.fn.__name__, args=d.args, seed=seed,
                     env=d.env.name if d.env else None, code=self.code_hash, status=status, error=error,
                     seconds=round(time.time() - start, 2), ran_on=ran_on(d.env, device), emitted=stream["stream"],
                     out=encode(out, self.large_dir) if status == "done" else None)
        self.logs.sets.setdefault(name, RecordSet(seed_offset=offset)).runs[rec.id] = rec
        self.logs.events.append(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), event="run", set=name, experiment=d.name,
                                     id=rec.id, args=d.args, seed=seed, code=self.code_hash, status=status, seconds=rec.seconds))
        self.save_logs()
        return rec

    def fresh(self, d, name, offset=0):
        """The record of a declared run in set `name`, if it finished and the code has not changed since."""
        rs = self.logs.sets.get(name)
        rec = rs.runs.get(d.id(offset)) if rs else None
        return rec if rec is not None and rec.status == "done" and rec.code == self.code_hash else None

    def log_claim_edits(self):
        """Note in the run log any claim whose text changed since it was last logged."""
        events, last = self.logs.events, {}
        for e in events:
            if e["event"] == "claim_edit":
                last[e["claim"]] = e["hash"]
        changed = [c for c in self.claims if last.get(c.id) != c.hash]
        existed = any(e["event"] == "run" and e["status"] == "done" for e in events)
        for c in changed:
            events.append(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), event="claim_edit", claim=c.id, hash=c.hash,
                               first=c.id not in last, records_existed=existed))
        if changed:
            self.save_logs()

    def run(self, name="author", device=None, deep=False, rerun=False, say=print):
        """Run every declared run that has no fresh result, as the author.

        The small values go into the paper's results block, and the paper file is rewritten. The
        full outputs go to the logs, as record set `name`. With deep=True the cited papers run first."""
        device, done = device or default_device(), []
        if deep:
            for p in self._cited.values():
                done += p.run(name, device, True, rerun, say)
        self.log_claim_edits()
        declared = self.declared()
        old = self.results
        rows = {r.run: r for r in old.rows} if old and old.set == name else {}
        machine = dict(old.ran_on) if old and old.set == name else {}
        for d in declared:
            have = rows.get(d.id())
            if have is not None and have.code == self.code_hash and not rerun:
                continue
            rec = self.fresh(d, name) if not rerun else None
            if rec is None:
                rec = self.execute(d, name, device)
                done.append(rec)
                say(f"{self.stem}: {d.name} {json.dumps({k: v for k, v in d.args.items() if not str(v).startswith(('fn:', 'data:'))})}"
                    f" seed={d.seed} {rec.status} in {rec.seconds}s" + (f" ({rec.error})" if rec.error else ""))
            if rec.status == "done":
                rows[rec.id] = to_row(rec)
                keep = [rows[x.id()] for x in declared if x.id() in rows]      # only runs the paper still declares
                machine = {**machine, **rec.ran_on, "packages": {**machine.get("packages", {}), **rec.ran_on["packages"]}}
                self._write_results(Results(set=name, ran_on=machine, rows=keep))
        return done

    def _write_results(self, results):
        parts = []
        for p in self.parts:
            if p.kind == "results":
                continue
            if p.kind == "prose" and parts and parts[-1].kind == "prose":
                parts[-1] = Prose(text=parts[-1].text + p.text)
            else:
                parts.append(p)
        if parts and parts[-1].kind == "prose":
            parts[-1] = Prose(text=parts[-1].text.rstrip("\n") + "\n\n")
        elif parts:
            parts.append(Prose(text="\n\n"))
        self.parts = parts + [results, Prose(text="\n")]
        self._text = self.render()
        with open(self.path, "w") as f:
            f.write(self._text)


class _Cited:
    """A cite together with the paper it resolved to."""

    def __init__(self, cite, owner):
        self.cite, self.name, self.at = cite, cite.name, cite.at
        self.paper = owner._cited.get(cite.name)
