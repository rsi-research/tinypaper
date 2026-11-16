"""paper.check(): is the paper well formed, can it be rerun, and what does its log say?"""
import os
from typing import Any
from .model import Strict
from .records import differs, evaluate
from .store import Unreachable, data_file


class Report(Strict):
    paper: str
    ok: bool = False                       # nothing is wrong or unreachable
    valid: bool = False                    # the paper follows the format
    problems: list[str] = []
    claims: dict[str, dict[str, str]] = {}     # claim -> {log: status}; the log "paper" is the author's, in the file
    calls: dict[str, dict[str, int]] = {}      # log -> how many of the paper's calls are logged, and missing
    ran_differently: dict[str, list[str]] = {}
    challenged_by: list[dict[str, str]] = []
    history: dict[str, Any] = {}
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
        out.append(f"  {self.lines} lines of code, {self.declared} calls, cites {', '.join(self.cites) or 'nothing'}")
        for log, n in self.calls.items():
            label = "the paper's log" if log == "paper" else f"local log `{log}`"
            out.append(f"  {label}: {n['logged']} calls logged, {n['missing']} missing")
            out += [f"    {c}: {s[log]}" for c, s in self.claims.items() if log in s]
            out += ["    ran differently: " + m for m in self.ran_differently.get(log, [])]
        out += [f"  challenged: claim {c['challenges']} by {c['paper']} claim {c['claim']}" for c in self.challenged_by]
        h = self.history
        if h.get("runs"):
            out.append(f"  history: {h['runs']} runs made"
                       + (f", {h['no_longer_made']} of them for calls the paper no longer makes" if h.get("no_longer_made") else "")
                       + (f"; claims edited after results existed: {', '.join(h['claims_edited'])}" if h.get("claims_edited") else ""))
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
    """How a machine differed from the envs its calls name."""
    found = set()
    for env in envs:
        for pin in env.packages:
            if "==" in pin:
                name, want = pin.split("==")
                got = where.get("packages", {}).get(name, "unknown")
                if got.split("+")[0] != want:
                    found.add(f"{name} {got}, env `{env.name}` pins {want}")
        if env.hardware and where.get("device") and ("gpu" in env.hardware.lower()) != where["device"].startswith("cuda"):
            found.add(f"device {where['device']}" + (f" ({where['gpu']})" if "gpu" in where else "") + f", env `{env.name}` says {env.hardware}")
    return sorted(found)


class PaperChecks:
    def check(self):
        """Check the paper. Runs nothing.

        It fails if the paper breaks the format, if something needed to rerun it is unreachable (a
        cited paper, a data file), if a claim raises, or if a claim's `differs=` is wrong. Claims
        are evaluated on the paper's log and on every local rerun. A cited paper is always read
        from its own log."""
        report = Report(paper=self.path)
        problems = report.problems
        if self._unreachable:
            problems += [f"unreachable: {e}" for e in self._unreachable]
            return report
        try:
            declared, calls = self.declared(), self.calls()
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
        envs = list({d.env.name: d.env for d in declared if d.env}.values())
        for log in [None] + sorted(self.local.logs):
            label = log or "paper"
            entries, offset = self.entries(log)
            logged = sum(1 for d in calls if d.id(offset) in entries and entries[d.id(offset)].status == "done")
            report.calls[label] = dict(logged=logged, missing=len(calls) - logged)
            where = (self.log.ran_on if self.log else {}) if log is None else self.local.logs[log].ran_on
            if where:
                report.ran_differently[label] = env_mismatch(envs, where)
            for c in self.claims:
                status, touched = evaluate(self, c, log)
                report.claims.setdefault(c.id, {})[label] = status
                if c.differs is not None and status in ("holds", "fails") and differs(touched) != c.differs:
                    problems.append(f"claim {c.id} says differs={','.join(c.differs) or 'nothing'}; "
                                    f"the calls it read differ in {','.join(differs(touched)) or 'nothing'} ({label})")
                if status.startswith("error"):
                    problems.append(f"claim {c.id} raised an error on {label}: {status[7:]}")
        events = self.local.history
        if events:
            runs = [e for e in events if e["event"] == "run"]
            now = {d.id() for d in calls} | {d.id(1000) for d in calls}
            report.history = dict(self.history_counts(), no_longer_made=sum(e["call"] not in now for e in runs))
        elif self.log:
            report.history = dict(self.log.history)        # no local history here: what the author's log recorded
        report.declared, report.lines = len(calls), self.lines
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
