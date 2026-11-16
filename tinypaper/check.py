"""paper.check(): is the paper well formed, can it be rerun, and what does its log say?"""
import os
from typing import Any
from .model import Strict
from .records import differ, evaluate
from .store import Unreachable, data_file


class Report(Strict):
    paper: str
    ok: bool = False                       # nothing is wrong or unreachable
    valid: bool = False                    # the file follows the format
    problems: list[str] = []
    claims: dict[str, str] = {}            # each claim's status on the paper's log
    compares: dict[str, list[str]] = {}    # for each claim that evaluates: what differs among the calls it read
    calls: int = 0                         # distinct calls the paper makes
    logged: int = 0                        # how many of them are in its log
    ran_differently: list[str] = []
    challenged_by: list[dict[str, str]] = []
    history: dict[str, Any] = {}
    lines: int = 0
    cites: list[str] = []

    def text(self):
        out = [f"{os.path.basename(self.paper)}: {'ok' if self.ok else 'NOT OK'}"]
        out += ["  problem: " + p for p in self.problems]
        if not self.valid and not self.claims:
            return "\n".join(out)
        out.append(f"  {self.lines} lines of code, cites {', '.join(self.cites) or 'nothing'}; {self.logged} of {self.calls} calls logged")
        for c, status in self.claims.items():
            how = self.compares.get(c)
            out.append(f"  {c}: {status}" + (f" (compares calls that differ in {', '.join(how)})" if how else ""))
        out += ["  ran differently: " + m for m in self.ran_differently]
        out += [f"  challenged: claim {c['challenges']} by {c['paper']} claim {c['claim']}" for c in self.challenged_by]
        h = self.history
        if h.get("runs"):
            out.append(f"  history: {h['runs']} runs made"
                       + (f", {h['no_longer_made']} of them for calls the paper no longer makes" if h.get("no_longer_made") else "")
                       + (f"; claims edited after results existed: {', '.join(h['claims_edited'])}" if h.get("claims_edited") else ""))
        return "\n".join(out)


def all_papers(paper, seen=None):
    seen = {} if seen is None else seen
    if paper.path not in seen:
        seen[paper.path] = paper
        for p in paper.cites.values():
            all_papers(p, seen)
    return seen


def version_mismatch(envs, where):
    """Pinned packages whose installed version, on the machine that wrote the log, was another one."""
    found = set()
    for env in envs:
        for pin in env.packages:
            if "==" in pin:
                name, want = pin.split("==")
                got = where.get("packages", {}).get(name, "unknown")
                if got.split("+")[0] != want:
                    found.add(f"{name} {got}, env `{env.name}` pins {want}")
    return sorted(found)


class PaperChecks:
    def check(self):
        """Check the paper. Runs nothing.

        It fails if the paper breaks the format, if something needed to rerun it is unreachable (a
        cited paper, a data file), or if a claim raises. Claims are evaluated on the paper's log; a
        cited paper is read from its own log."""
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
                if alias not in self.cites or target not in {x.id for x in self.cites[alias].claims}:
                    problems.append(f"invalid: claim {c.id} challenges `{c.challenges}`, which is not a cited paper's claim")
        report.valid = not problems
        for p in all_papers(self).values():                # every data file, here or in a cited paper
            for spec in p.data.values():
                for f in (spec.files or {}).values():
                    try:
                        data_file(f.url, f.sha256, verify=True)
                    except Unreachable as e:
                        problems.append(f"unreachable: {e}")
        logged = self.logged
        report.calls, report.logged = len(calls), sum(d.id in logged for d in calls)
        if self.log:
            report.ran_differently = version_mismatch({d.env.name: d.env for d in declared if d.env}.values(), self.log.ran_on)
        for c in self.claims:
            status, read = evaluate(self, c)
            report.claims[c.id] = status
            if status in ("holds", "fails") and differ(read):
                report.compares[c.id] = differ(read)
            if status.startswith("error"):
                problems.append(f"claim {c.id} raised an error: {status[7:]}")
        if self.local.history:
            now = {d.id for d in calls}
            report.history = dict(self.history_counts(), no_longer_made=sum(e["event"] == "run" and e["call"] not in now for e in self.local.history))
        elif self.log:
            report.history = dict(self.log.history)        # no local history here: what the paper's log recorded
        report.lines = self.lines
        report.cites = sorted(os.path.basename(p.path) for p in self.cites.values())
        report.challenged_by = self.challenges()
        report.ok = not problems
        return report

    def challenges(self):
        """Claims of other papers in this vault that challenge a claim of this paper."""
        found = []
        for other in self.vault.papers.values():
            for c in other.claims:
                alias, _, target = (c.challenges or "").partition(".")
                if alias in other.cites and other.cites[alias].path == self.path:
                    found.append(dict(paper=os.path.basename(other.path), claim=c.id, challenges=target))
        return found
