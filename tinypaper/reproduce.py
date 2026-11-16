"""paper.reproduce(): rerun a paper as far as this machine gets in the time given, and say how far that was."""
import re, time as clock
from statistics import median
from typing import Optional
from .model import Strict
from .records import PREFIX, evaluate

TIERS = ["valid", "checked", "starts", "prefix", "partial", "full", "robust"]


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
    paper: str
    device: str
    log: str                               # the local log that holds this rerun
    tier: Optional[str] = None             # the highest tier reached
    coverage: dict[str, str] = {}          # what was done inside each tier
    claims: dict[str, str] = {}            # each claim: holds, fails, or not attempted
    warnings: list[str] = []
    speed: Optional[float] = None          # this machine's time for the first steps, relative to the author's
    needed: Optional[float] = None         # estimated seconds still needed here to rerun everything

    def text(self):
        out = [f"{self.paper}: {self.tier or 'not valid'} (on {self.device})"]
        out += [f"  {t}: {c}" for t, c in self.coverage.items()]
        out += [f"  claim {c}: {s}" for c, s in self.claims.items()]
        if self.needed:
            out.append(f"  about {about(self.needed)} more is needed here to rerun everything")
        return "\n".join(out + ["  warning: " + w for w in self.warnings])


def same_machine(a, b):
    """Same Python, device and GPU, and the same version of every package both list."""
    shared = set(a.get("packages", {})) & set(b.get("packages", {}))
    return all(a.get(x) == b.get(x) for x in ["python", "device", "gpu"]) and all(a["packages"][k] == b["packages"][k] for k in shared)


def compare(mine, ref, siblings, exact, upto=PREFIX):
    """How many emitted points of a rerun agree with the paper's, in order, and how many were compared.

    On the same machine and versions values must be equal. Otherwise a value must be within 0.1%
    of the paper's, or failing that inside the range of the paper's seeds at that point, widened
    on each side by that range's width (1% of the value if there is one seed)."""
    n = min(len(mine), len(ref), upto or len(ref))
    for i in range(n):
        for key, want in ref[i].items():
            if key == "t":
                continue
            value = mine[i].get(key)
            if not isinstance(want, float) or exact:
                ok = value == want
            else:
                others = [s[i][key] for s in siblings if len(s) > i and isinstance(s[i].get(key), float)]
                spread = max(max(others) - min(others) if len(others) > 1 else 0.01 * abs(want), 1e-4 * abs(want))
                ok = abs(value - want) <= 1e-3 * abs(want) or min(others) - spread <= value <= max(others) + spread
            if not ok:
                return i, n
    return n, n


class PaperReproduce:
    def reproduce(self, time=None, device=None, seeds="recorded", name=None, upto=None, say=print):
        """Rerun the paper on this machine within a time limit and report the highest tier reached.

        valid    the paper parses and follows the format
        checked  `check` passes
        starts   every call starts here: data loads and the first point is emitted
        prefix   the first points each call emits match the paper's log
        partial  at least one claim was rerun in full
        full     every call was rerun and every claim evaluated
        robust   the same with seeds the paper never used (seeds="fresh")

        `time` is seconds, or text such as "10m"; without it everything is rerun. Every call is first
        taken a few steps, which also measures this machine's speed. The time left goes to whole
        claims, cheapest first, and a claim is started only if its calls are expected to fit.
        Calling again continues: calls already finished are kept.

        The rerun is a local log named `name`. The paper is never changed, and cited papers are
        read from their own logs."""
        from .paper import default_device
        begin, limit = clock.time(), seconds(time)
        deadline = None if limit is None else begin + limit
        device = device or default_device()
        offset = 1000 if seeds == "fresh" else 0
        name = name or (f"repro-{device}" + ("-fresh" if offset else ""))
        result = Reproduction(paper=self.stem, device=device, log=name)
        report = self.check()
        reached = {t: False for t in TIERS}
        reached["valid"], reached["checked"] = report.valid, report.ok
        cover = result.coverage

        def finish():
            # a full rerun is stronger evidence than the tiers below it, so it stands if one of them was missed
            top = max([TIERS.index(t) for t in ("full", "robust") if reached[t]], default=-1)
            for i, t in enumerate(TIERS):
                if not reached[t] and i > top:
                    break
                if not reached[t] and t in ("checked", "starts", "prefix"):
                    result.warnings.append(f"reached {TIERS[top]} although `{t}` did not pass: {cover.get(t) or '; '.join(report.problems)}")
                if reached[t]:
                    result.tier = t
            return result

        if not report.valid or upto in ("valid", "checked"):
            return finish()
        calls = self.calls()
        with self.locked():
            self.note_claim_edits()
            self.save_local()
        refs = self.entries(None)[0]
        author = self.log.ran_on if self.log else {}

        def done(d):
            e = self.entries(name)[0].get(d.id(offset))
            return e if e is not None and e.status == "done" else None

        # every call, a few steps: does it start, does it match, and how fast is this machine?
        started = errors = compared = matched = values = values_ok = skipped = 0
        exact, ratios = True, []
        for d in calls:
            if deadline is not None and clock.time() > deadline:
                skipped += 1
                continue
            ref = refs.get(d.id())
            entry = done(d)
            if entry is None:
                if ref is not None and not ref.emitted and deadline is not None and ref.seconds * (median(ratios) if ratios else 1) > deadline - clock.time():
                    skipped += 1                   # it emits nothing as it runs, so it cannot be stopped: do not start it
                    continue
                entry, machine = self.execute(d, device, max_emits=1 if upto == "starts" else PREFIX, deadline=deadline, offset=offset, log=name)
                self.keep(name, entry, machine, offset)
            else:
                machine = self.local.logs[name].ran_on
            if entry.status == "error":
                errors += 1
                say(f"{self.stem}: {d.name} failed to start: {entry.error}")
                continue
            started += 1
            if ref is None or not ref.emitted or not entry.emitted:
                continue
            k = min(len(entry.emitted), len(ref.emitted), PREFIX)
            if ref.emitted[k - 1].get("t") and k == min(len(ref.emitted), PREFIX):
                ratios.append(entry.emitted[k - 1]["t"] / ref.emitted[k - 1]["t"])
            if not offset and upto != "starts":
                same = same_machine(machine, author)
                siblings = [r.emitted for r in refs.values() if r.fn == ref.fn and r.args == ref.args]
                ok, n = compare(entry.emitted, ref.emitted, siblings, same)
                compared, matched, values, values_ok, exact = compared + 1, matched + (ok == n), values + n, values_ok + ok, exact and same
        result.speed = round(median(ratios), 2) if ratios else None
        cover["starts"] = f"{started} of {len(calls)} calls started" + (f"; {skipped} not started for lack of time" if skipped else "")
        reached["starts"] = started == len(calls) and not errors
        if upto == "starts":
            return finish()
        if offset:
            cover["prefix"] = "not compared: these are new seeds"
            reached["prefix"] = reached["starts"]
        else:
            rule = "exactly" if exact else "within 0.1%, or within the spread between seeds"
            cover["prefix"] = (f"{matched} of {compared} calls match the paper over their first emitted points "
                               f"({values_ok} of {values} points, {rule})" if compared else "no emitted points to compare")
            reached["prefix"] = matched == compared and not errors and not skipped

        def estimate(d):
            ref = refs.get(d.id())
            return None if ref is None else ref.seconds * (result.speed or 1)

        def left():
            return float("inf") if deadline is None else deadline - clock.time()

        def missing():
            return [d for d in calls if not done(d)]

        def rerun(d):
            entry, machine = self.execute(d, device, deadline=deadline, offset=offset, log=name)
            self.keep(name, entry, machine, offset)
            return entry

        total = sum(estimate(d) or 0 for d in missing())
        if result.speed:
            say(f"{self.stem}: this machine takes about {result.speed}x the author's time for the first steps; "
                f"rerunning everything needs about {about(total)}" + ("" if deadline is None else f", and {about(max(left(), 0))} is left"))
        if upto == "prefix":
            result.needed = round(total, 1)
            return finish()

        # whole claims, cheapest first, as long as they are expected to fit
        plans, costs = [], {}
        for c in self.claims:
            status, touched = evaluate(self, c)
            # a claim needs every call of each experiment it reads; if it cannot be read from the paper's log, assume all
            names = {x.experiment for x in touched if x.paper.path == self.path}
            need = self.calls(names) if status in ("holds", "fails") and names else list(calls) if status not in ("holds", "fails") else []
            plans.append((sum(estimate(d) or 0 for d in need if not done(d)), c.id, need))
        for _, cid, need in sorted(plans, key=lambda p: p[0]):
            cost = costs[cid] = sum(estimate(d) or 0 for d in need if not done(d))
            if cost > left():
                say(f"{self.stem}: claim {cid} not attempted (needs about {about(cost)} here)")
                continue
            for d in need:
                if not done(d) and left() > 0:
                    entry = rerun(d)
                    say(f"{self.stem}: {d.name} seed={d.seed} {entry.status} in {entry.seconds}s")
        if deadline is None:
            for d in missing():
                rerun(d)

        tried = held = 0
        for c in self.claims:
            status = evaluate(self, c, name)[0]
            if status in ("holds", "fails"):
                tried, held = tried + 1, held + (status == "holds")
            else:
                cost = costs.get(c.id) or sum(estimate(d) or 0 for d in missing())
                status = "not attempted" + (f" (needs about {about(cost)} here)" if cost else f" ({status})")
            result.claims[c.id] = status
        finished = len(calls) - len(missing())
        result.needed = round(sum(estimate(d) or 0 for d in missing()), 1)
        text = f"{tried} of {len(self.claims)} claims attempted, {held} held; {finished} of {len(calls)} calls rerun"
        cover["partial"] = text
        reached["partial"] = tried > 0 or (not self.claims and finished > 0)
        reached["full"] = finished == len(calls) and tried == len(self.claims)
        if reached["full"]:
            points = agree = 0
            if not offset:                         # every point the paper logged, not only the first ones
                same = same_machine(self.local.logs[name].ran_on, author) if name in self.local.logs else True
                for d in calls:
                    ref, mine = refs.get(d.id()), done(d)
                    if ref is not None and ref.emitted and mine is not None:
                        siblings = [r.emitted for r in refs.values() if r.fn == ref.fn and r.args == ref.args]
                        ok, n = compare(mine.emitted, ref.emitted, siblings, same, upto=None)
                        points, agree = points + n, agree + ok
            cover["full"] = text + (f"; {agree} of {points} logged points agree with the paper up to the first difference in each call" if points else "")
            if offset:
                cover["robust"] = text + ", with new seeds"
                reached["robust"] = True
        return finish()
