"""paper.reproduce(): rerun a paper as far as this machine gets in the time given, and count how far that was."""
import re, time as clock
from statistics import median
from typing import Optional
from .model import Strict
from .records import EARLY, evaluate

TOLERANCE = 1e-3            # how far a rerun's emitted value may be from the paper's, relative to it


def seconds(t):
    """A time limit in seconds, from a number or text such as "90s", "10m" or "2h"."""
    if t is None or isinstance(t, (int, float)):
        return t
    m = re.fullmatch(r"\s*([\d.]+)\s*([smh]?)\s*", str(t))
    if not m:
        raise ValueError(f"cannot read the time limit {t!r}; write it like 90s, 10m or 2h")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def about(s):
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


class Reproduction(Strict):
    """How far a rerun got, as counts."""
    paper: str
    device: str
    verified: bool = False                 # `verify` found nothing wrong with the format or what the paper needs
    problems: list[str] = []
    calls: int = 0                         # distinct calls the paper makes
    started: int = 0                       # calls that began on this machine
    compared: int = 0                      # calls whose first emitted points were compared with the paper's log
    matched: int = 0                       # of those, calls whose first points all agree
    rerun: int = 0                         # calls rerun to the end
    points: int = 0                        # over the calls rerun to the end: logged points compared,
    points_agree: int = 0                  #   and how many agree before the first difference in each call
    claims: dict[str, str] = {}            # each claim on the rerun: holds, fails, or not attempted
    speed: Optional[float] = None          # this machine's time for the first steps, relative to the author's
    needed: Optional[float] = None         # estimated seconds still needed here to rerun everything

    @property
    def held(self):
        return sum(s == "holds" for s in self.claims.values())

    @property
    def attempted(self):
        return sum(s in ("holds", "fails") for s in self.claims.values())

    def text(self):
        out = [f"{self.paper} (on {self.device}): {'verified' if self.verified else 'NOT verified'}"]
        out += ["  problem: " + p for p in self.problems]
        out.append(f"  {self.started} of {self.calls} calls started; first points match the paper in {self.matched} of {self.compared} compared")
        out.append(f"  {self.rerun} of {self.calls} calls rerun to the end"
                   + (f"; {self.points_agree} of {self.points} logged points agree" if self.points else ""))
        out.append(f"  {self.attempted} of {len(self.claims)} claims attempted, {self.held} held")
        out += [f"    {c}: {s}" for c, s in self.claims.items()]
        if self.needed:
            out.append(f"  about {about(self.needed)} more is needed here to rerun everything")
        return "\n".join(out)


def agree(mine, ref, upto=None):
    """How many emitted points of a rerun agree with the paper's, before the first that does not, and how many were compared."""
    n = min(len(mine), len(ref), upto or len(ref))
    for i in range(n):
        for key, want in ref[i].items():
            value = mine[i].get(key)
            if key != "t" and not (abs(value - want) <= TOLERANCE * abs(want) if isinstance(want, float) and isinstance(value, float) else value == want):
                return i, n
    return n, n


class PaperReproduce:
    def reproduce(self, time=None, device=None, full=True, say=print):
        """Rerun the paper on this machine within a time limit, and report how far it got.

        `time` is seconds, or text such as "10m"; without it everything is rerun. First, every call
        is taken its first few emitted points and compared with the paper's log, which also
        measures this machine's speed. With full=True the time left then goes to whole claims,
        cheapest first; a claim is started only if its calls are expected to fit. Calling again
        continues: calls already finished are kept beside the vault.

        The paper is never changed, and cited papers are read from their own logs."""
        from .paper import default_device
        begin, limit = clock.time(), seconds(time)
        deadline = None if limit is None else begin + limit
        device = device or default_device()
        result = Reproduction(paper=self.stem, device=device)
        report = self.verify()
        result.verified, result.problems = not report.problems, list(report.problems)
        if not report.valid:
            return result
        calls, refs = self.calls(), self.logged
        result.calls = len(calls)
        self.local.rerun = {i: e for i, e in self.local.rerun.items() if i in {d.id for d in calls}}    # drop calls the paper no longer makes

        def keep(d, got):
            if got.entry:
                self.local.rerun[d.id] = got.entry
                self.save_local()

        def left():
            return float("inf") if deadline is None else deadline - clock.time()

        # every call, its first few points: does it start, does it match, and how fast is this machine?
        ratios, errors = [], 0
        for d in calls:
            if left() <= 0:
                break
            ref, mine = refs.get(d.id), self.local.rerun.get(d.id)
            if mine is not None:
                emitted = mine.emitted
            else:
                if ref is not None and not ref.emitted and ref.seconds * (median(ratios) if ratios else 1) > left():
                    continue                       # it emits nothing as it runs, so it cannot be stopped: do not start it
                got = self._execute(d, device, self.local.rerun, max_emits=EARLY, deadline=deadline)
                if got.status == "error":
                    errors += 1
                    result.problems.append(f"{d.name} failed to start: {got.error}")
                    continue
                if got.entry:
                    keep(d, got)
                emitted = got.emitted
            result.started += 1
            if ref is None or not ref.emitted or not emitted:
                continue
            k = min(len(emitted), len(ref.emitted), EARLY)
            if ref.emitted[k - 1].get("t") and k == min(len(ref.emitted), EARLY):
                ratios.append(emitted[k - 1]["t"] / ref.emitted[k - 1]["t"])
            ok, n = agree(emitted, ref.emitted, EARLY)
            result.compared, result.matched = result.compared + 1, result.matched + (ok == n)
        result.speed = round(median(ratios), 2) if ratios else None

        def estimate(d):
            ref = refs.get(d.id)
            return 0 if ref is None else ref.seconds * (result.speed or 1)

        def missing():
            return [d for d in calls if d.id not in self.local.rerun]

        if result.speed:
            say(f"{self.stem}: this machine takes about {result.speed}x the author's time for the first steps; rerunning everything "
                f"needs about {about(sum(estimate(d) for d in missing()))}" + ("" if deadline is None else f", and {about(max(left(), 0))} is left"))

        # whole claims, cheapest first, as long as they are expected to fit
        costs = {}
        if full:
            plans = []
            for c in self.claims:
                status, read = evaluate(self, c)
                # a claim needs every call of each experiment it reads; if it cannot be read from the paper's log, assume all
                names = {x.experiment for x in read if x.paper.path == self.path}
                need = (self.calls(names) if names else []) if status in ("holds", "fails") else list(calls)
                plans.append((sum(estimate(d) for d in need if d.id not in self.local.rerun), c.id, need))
            for _, cid, need in sorted(plans, key=lambda p: p[0]):
                costs[cid] = sum(estimate(d) for d in need if d.id not in self.local.rerun)
                if costs[cid] > left():
                    say(f"{self.stem}: claim {cid} not attempted (needs about {about(costs[cid])} here)")
                    continue
                for d in need:
                    if d.id not in self.local.rerun and left() > 0:
                        got = self._execute(d, device, self.local.rerun, deadline=deadline)
                        keep(d, got)
                        say(f"{self.stem}: {d.name} seed={d.seed} {got.status} in {got.seconds}s")
            if deadline is None:
                for d in missing():
                    keep(d, self._execute(d, device, self.local.rerun))

        for c in self.claims:
            status = evaluate(self, c, self.local.rerun)[0]
            if status not in ("holds", "fails"):
                cost = costs.get(c.id) or sum(estimate(d) for d in missing())
                status = "not attempted" + (f" (needs about {about(cost)} here)" if cost else "")
            result.claims[c.id] = status
        result.rerun = len(calls) - len(missing())
        result.needed = round(sum(estimate(d) for d in missing()), 1)
        for d in calls:                            # every point the paper logged, for the calls that were rerun to the end
            ref, mine = refs.get(d.id), self.local.rerun.get(d.id)
            if ref is not None and ref.emitted and mine is not None:
                ok, n = agree(mine.emitted, ref.emitted)
                result.points, result.points_agree = result.points + n, result.points_agree + ok
        return result
