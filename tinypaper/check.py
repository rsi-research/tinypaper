"""paper.check(): is the paper well formed, can it be rerun, and what do its results say?"""
import os
from typing import Any
from .model import Strict
from .records import differs, evaluate, source_runs, to_row
from .store import Unreachable, data_file


class Report(Strict):
    paper: str
    ok: bool = False                       # nothing is wrong or unreachable
    valid: bool = False                    # the paper follows the format
    problems: list[str] = []
    evidence: str = "none"                 # of the paper's own results: none, reported, or logged
    claims: dict[str, dict[str, str]] = {}     # claim -> {source: status}; the source "paper" is the results block
    runs: dict[str, dict[str, int]] = {}       # source -> how many declared runs are fresh, stale, missing
    ran_differently: dict[str, list[str]] = {}
    challenged_by: list[dict[str, str]] = []
    log: dict[str, Any] = {}
    declared: int = 0
    lines: int = 0
    cites: list[str] = []
    installs: list[str] = []
    git_installs: list[str] = []

    def text(self):
        out = [f"{os.path.basename(self.paper)}: {'ok' if self.ok else 'NOT OK'}"]
        out += ["  problem: " + p for p in self.problems]
        if not self.valid and not self.claims:
            return "\n".join(out)
        out.append(f"  {self.lines} lines of code, {self.declared} runs declared, cites {', '.join(self.cites) or 'nothing'}")
        for source, r in self.runs.items():
            label = f"paper ({self.evidence})" if source == "paper" else f"logs `{source}`"
            out.append(f"  {label}: {r['fresh']} fresh, {r['stale']} stale, {r['missing']} missing")
            out += [f"    {c}: {s[source]}" for c, s in self.claims.items() if source in s]
            out += ["    ran differently: " + m for m in self.ran_differently.get(source, [])]
        out += [f"  challenged: claim {c['challenges']} by {c['paper']} claim {c['claim']}" for c in self.challenged_by]
        if self.log:
            lg = self.log
            out.append(f"  run log: {lg['runs']} runs over {lg['versions']} versions of the code, {lg['undeclared']} no longer declared"
                       + (f"; claims edited after results existed: {', '.join(lg['claims_edited'])}" if lg["claims_edited"] else ""))
        if self.git_installs:
            out.append("  installed from git: " + ", ".join(self.git_installs))
        return "\n".join(out)


def all_papers(paper, seen=None):
    seen = {} if seen is None else seen
    if paper.path not in seen:
        seen[paper.path] = paper
        for p in paper._cited.values():
            all_papers(p, seen)
    return seen


def env_mismatch(envs, where):
    """How a machine differed from the envs its runs name."""
    found = set()
    for env in envs:
        for pin in env.packages:
            if "==" in pin:
                name, want = pin.split("==")
                got = where.get("packages", {}).get(name, "unknown")
                if got.split("+")[0] != want:
                    found.add(f"{name} {got}, env `{env.name}` pins {want}")
        if env.hardware and ("gpu" in env.hardware.lower()) != where.get("device", "cpu").startswith("cuda"):
            found.add(f"device {where.get('device')}" + (f" ({where['gpu']})" if "gpu" in where else "") + f", env `{env.name}` says {env.hardware}")
    return sorted(found)


def sources(paper):
    """What a paper's claims can be evaluated on: the paper's own results, then other record sets in the logs."""
    own = paper.results.set if paper.results else None
    names = set(paper.logs.sets)
    if not paper.declared():                      # a paper with no runs is evaluated on what its cites have
        for p in paper._cited.values():
            names |= set(p.logs.sets) - {p.results.set if p.results else None}
    return ["paper"] + sorted(names - {own})


class PaperChecks:
    def check(self):
        """Check the paper. Runs nothing.

        It fails if the paper breaks the format, if something needed to rerun it is unreachable
        (a cited paper, a data file), if a claim's `differs=` is wrong, if a claim raises, or if the
        logs contradict what the paper reports. Claims are evaluated on the paper's reported
        results, completed from the logs when those are loaded, and on every other record set."""
        report = Report(paper=self.path)
        problems = report.problems
        if self._unreachable:
            problems += [f"unreachable: {e}" for e in self._unreachable]
            return report
        try:
            declared = self.declared()
        except Unreachable as e:
            problems.append(f"unreachable: {e}")
            return report
        except Exception as e:
            problems.append(f"invalid: {type(e).__name__}: {e}")
            return report
        problems += [f"invalid: {p}" for p in self.problems]
        for c in self.claims:
            if c.challenges:
                alias, _, target = c.challenges.partition(".")
                if alias not in self._cited or target not in {x.id for x in self._cited[alias].claims}:
                    problems.append(f"invalid: claim {c.id} challenges `{c.challenges}`, which is not a cited paper's claim")
        report.valid = not problems
        for p in all_papers(self).values():                # every data file, here or in a cited paper
            for spec in p.data.values():
                for f in (spec.files or {}).values():
                    try:
                        data_file(f.url, f.sha256, verify=True)
                    except Unreachable as e:
                        problems.append(f"unreachable: {e}")
        res, envs = self.results, {d.env.name: d.env for d in declared if d.env}.values()
        if res:                                            # do the logs agree with what the paper reports?
            logged = self.logs.sets.get(res.set)
            hits = 0
            for row in res.rows:
                rec = logged.runs.get(row.run) if logged else None
                if rec is not None and rec.code == row.code and rec.status == "done":
                    hits += 1
                    if to_row(rec).out != row.out:
                        problems.append(f"the paper reports {row.out} for run {row.experiment} {row.run[:8]}, the logs say {to_row(rec).out}")
            report.evidence = "logged" if res.rows and hits == len(res.rows) else "reported"
        for source in sources(self):
            runs, offset = source_runs(self, source)
            state = dict(fresh=0, stale=0, missing=0)
            for d in declared:
                run = runs.get(d.id(offset))
                code = (run._record.code if run._record else run._row.code) if run else None
                state["missing" if run is None else "fresh" if code == self.code_hash else "stale"] += 1
            report.runs[source] = state
            where = [res.ran_on] if source == "paper" and res else list({str(r.ran_on): r.ran_on for r in runs.values()}.values())
            if source != "paper" or res:
                report.ran_differently[source] = sorted({m for w in where for m in env_mismatch(envs, w)})
            for c in self.claims:
                status, touched = evaluate(self, c, source)
                report.claims.setdefault(c.id, {})[source] = status
                if status in ("holds", "fails") and differs(touched) != c.differs:
                    problems.append(f"invalid: claim {c.id} says differs={','.join(c.differs) or 'nothing'}; "
                                    f"the runs it read differ in {','.join(differs(touched)) or 'nothing'} ({source})")
                if status.startswith("error"):
                    problems.append(f"claim {c.id} raised an error on {source}: {status[7:]}")
        events = self.logs.events
        if events:
            runs = [e for e in events if e["event"] == "run"]
            known = {d.id() for d in declared} | {d.id(1000) for d in declared}
            report.log = dict(runs=len(runs), undeclared=len({e["id"] for e in runs} - known), versions=len({e["code"] for e in runs}),
                              claims_edited=sorted({e["claim"] for e in events if e["event"] == "claim_edit" and e["records_existed"] and not e["first"]}))
        report.declared, report.lines = len(declared), self.lines
        report.cites = sorted(os.path.basename(p.path) for p in self._cited.values())
        report.installs = sorted({pin for e in self.envs.values() for pin in e.packages})
        report.git_installs = [p for p in report.installs if p.startswith("git+")]
        report.challenged_by = self.challenges()
        report.ok = not problems
        return report

    def challenges(self):
        """Claims of other papers in this vault that challenge a claim of this paper."""
        found = []
        for other in self.vault.papers.values():
            for c in other.claims:
                alias, _, target = (c.challenges or "").partition(".")
                if alias in other._cited and other._cited[alias].path == self.path:
                    found.append(dict(paper=os.path.basename(other.path), claim=c.id, challenges=target))
        return found
