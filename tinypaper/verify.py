"""paper.verify(): is the paper well formed, can it be rerun, and which claims hold on its log?"""
import os
from .format import Strict
from .records import differ, evaluate
from .store import Unreachable, data_file


class Report(Strict):
    paper: str
    ok: bool = False                       # nothing is wrong or unreachable, and every claim holds
    valid: bool = False                    # the file follows the format
    problems: list[str] = []
    claims: dict[str, str] = {}            # each claim's status on the paper's log
    statements: dict[str, str] = {}        # each claim in words: its function's docstring
    compares: dict[str, list[str]] = {}    # for each claim that evaluates: what differs among the calls it read
    calls: int = 0                         # distinct calls the paper makes
    logged: int = 0                        # how many of them are in its log
    ran_differently: list[str] = []
    challenged_by: list[dict[str, str]] = []
    lines: int = 0
    cites: list[str] = []
    notes: list[str] = []

    def text(self):
        out = [f"{os.path.basename(self.paper)}: {'ok' if self.ok else 'NOT OK'}"]
        out += ["  problem: " + p for p in self.problems]
        if not self.valid and not self.claims:
            return "\n".join(out)
        out.append(f"  {self.lines} lines of code, cites {', '.join(self.cites) or 'nothing'}; {self.logged} of {self.calls} calls logged")
        for c, status in self.claims.items():
            how = self.compares.get(c)
            out.append(f"  {c}: {status}" + (f" (compares calls that differ in {', '.join(how)})" if how else ""))
            if self.statements.get(c):
                out.append(f"      {self.statements[c]}")
        out += ["  note: " + n for n in self.notes]
        out += ["  ran differently: " + m for m in self.ran_differently]
        out += [f"  challenged: claim {c['challenges']} by {c['paper']} claim {c['claim']}" for c in self.challenged_by]
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


class PaperVerify:
    def verify(self):
        """Verify the paper. Runs no experiments, but executes the paper's code blocks and claim functions.

        It fails if the paper breaks the format, if something needed to rerun it is unreachable (a
        cited paper, a data file), or if a claim does not hold. Claims are evaluated on the paper's
        log, which is taken as given; a cited paper is read from its own log."""
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
        if None in logged:
            problems.append("invalid: the log has an entry that matches no call the paper declares (its fn, args and seed)")
        if self.log and any(e.call is None for e in self.log.entries):
            report.notes.append("some log entries have no call id; paper.tidy() writes them in")
        if self.log:
            report.ran_differently = version_mismatch({d.env.name: d.env for d in declared if d.env}.values(), self.log.ran_on)
        for c in self.claims:
            status, read = evaluate(self, c)
            report.claims[c.id] = status
            if c.statement:
                report.statements[c.id] = c.statement
            else:
                report.notes.append(f"claim {c.id} has no docstring saying what it claims")
            if status in ("holds", "fails") and differ(read):
                report.compares[c.id] = differ(read)
            if status.startswith("error"):
                problems.append(f"claim {c.id} raised an error: {status[7:]}")
        import re
        written = re.search(r"^Claims: .*\n", self._text or "", re.M)
        if written and written.group(0) != self.status_line():
            problems.append(f"invalid: the paper says `{written.group(0).strip()}` but its log gives `{self.status_line().strip()}`; paper.tidy() rewrites the line")
        report.lines = self.lines
        report.cites = sorted(os.path.basename(p.path) for p in self.cites.values())
        report.challenged_by = self.challenges()
        report.ok = not problems and all(s == "holds" for s in report.claims.values())
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
