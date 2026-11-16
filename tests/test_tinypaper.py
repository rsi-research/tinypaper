"""Tests on a toy vault that needs only numpy. Run with pytest, or directly with python."""
import hashlib, os, re, shutil, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tinypaper as tp
from tinypaper import store
from tinypaper.format import EXAMPLE, Log

A = '''---
title: Toy root
tinypaper: 0.5
date: 2026-11-16
---

Some prose.

```env
name: cpu
hardware: 1 CPU
packages: [ "numpy==0.0.1" ]
```

```data
name: nums
load: load
files:
  nums:
    url: file://DATA
    sha256: HASH
```

```python code=all
from dataclasses import dataclass
import numpy as np
from tinypaper import emit


@dataclass
class Nums:
    x: np.ndarray


def load(files):
    return Nums(np.loadtxt(files["nums"]))


def fit(d, scale, seed, steps=20, device="cpu"):
    rng = np.random.default_rng(seed)
    for step in range(steps):
        emit(step=step, loss=float(scale * rng.random()))
    return dict(mean=float(d.x.mean() * scale), top=np.float64(d.x.max() * scale))


def make(d):
    return dict(w=d.x * 10)


def use(init):
    return dict(total=float(sum(init["w"])))
```

```python experiments
call("small", fit, d=nums, scale=1, env=cpu, seeds=2)
call("big", fit, d=nums, scale=2, env=cpu, seeds=2)
call("pair", fit, d=nums, scale=1, env=cpu, seeds=2)
pre = call("pre", make, d=nums)
call("post", use, init=pre)
```

```python claim=c1 stated=2026-11-16
def c1(r):
    return r.big[0].out["mean"] > r.small[0].out["mean"]
```

```python claim=c2 stated=2026-11-16
def c2(r):
    return r.post[0].out["total"] == 60.0 and r.pair[1].out == r.small[1].out
```
'''

B = '''---
title: Toy challenge
tinypaper: 0.5
date: 2026-11-16
---

```cite
name: a
at: ./a.md
```

```python code=use
from a import fit
```

```python experiments
call("huge", fit, d=a.nums, scale=3, seeds=1)
```

```python claim=not_so stated=2026-11-16 challenges=a.c1
def not_so(r):
    return r.huge[0].out["mean"] > r.a.big[0].out["mean"]
```
'''

C = '''---
title: Toy analysis
tinypaper: 0.5
date: 2026-11-16
---

```cite
name: a
at: ./a.md
```

```python claim=only stated=2026-11-16
def only(r):
    return r.a.big[0].out["mean"] == 2 * r.a.small[0].out["mean"]
```
'''


def setup(papers="ab"):
    d = tempfile.mkdtemp()
    store.CACHE = os.path.join(d, "cache")
    data = os.path.join(d, "nums.txt")
    open(data, "w").write("1\n2\n3\n")
    digest = hashlib.sha256(open(data, "rb").read()).hexdigest()
    v = os.path.join(d, "toy")
    os.makedirs(v)
    for name, text in dict(a=A, b=B, c=C).items():
        if name in papers:
            open(os.path.join(v, name + ".md"), "w").write(text.replace("DATA", data).replace("HASH", digest))
    return d, os.path.join(v, "a.md"), os.path.join(v, "b.md")


def edit(path, old, new):
    s = open(path).read()
    assert old in s, old
    open(path, "w").write(s.replace(old, new))


def quiet(*a):
    pass


def paper(path):
    return tp.Paper.load(path)


def run(path):
    """Stand in for an author: run every call the log lacks, any way at all, and write the log into
    the paper. TinyPaper has no runner; this uses the step `reproduce` is built on."""
    p = paper(path)
    kept, got, machine = dict(p.logged), [], dict(p.log.ran_on) if p.log else {}
    for d in p.calls():
        if d.id not in kept:
            got.append(p._execute(d, "cpu", kept))
            machine = {**machine, **got[-1].machine}
            if got[-1].entry:
                kept[d.id] = got[-1].entry
    if got:
        p._write_log(Log(ran_on=machine, entries=[kept[d.id] for d in p.calls() if d.id in kept]))
    return got


def test_parse_and_render():
    d, a, b = setup()
    p = paper(a)
    assert p.render() == open(a).read()
    assert [x.name for x in p.code] == ["all"] and list(p.envs) == ["cpu"] and [c.id for c in p.claims] == ["c1", "c2"]
    assert p.data["nums"].files["nums"].sha256 and p.header.title == "Toy root" and list(paper(b).cites) == ["a"]
    edit(a, "hardware: 1 CPU", "hardware: 1 CPU\ncolour: red")
    assert any("colour" in x for x in paper(a).problems)          # an unknown field is an error
    edit(a, "claim=c1 stated=2026-11-16", "claim=c1 stated=2026-11-16 differs=scale")
    assert any("unknown attribute" in x for x in paper(a).problems)


def test_log_and_verify():
    d, a, b = setup()
    rep = paper(a).verify()
    assert rep.valid and not rep.ok and (rep.calls, rep.logged) == (6, 0) and rep.claims["c1"].startswith("open"), rep.problems
    assert len(paper(a).declared()) == 8 and len(paper(a).calls()) == 6       # two experiments share two calls
    assert len(run(a)) == 6
    p = paper(a)
    assert len(p.log.entries) == 6 and "```log" in open(a).read()
    big = [e for e in p.log.entries if e.args.get("scale") == 2][0]
    assert big.out == {"mean": 4.0, "top": 6.0} and len(big.emitted) == 20 and big.emitted[0]["step"] == 0
    rep = p.verify()
    assert rep.ok and rep.claims == dict(c1="holds", c2="holds") and rep.logged == 6, rep.problems
    assert rep.compares == {"c1": ["scale"], "c2": ["fn"]} and any("numpy" in m for m in rep.ran_differently)
    assert run(a) == []
    assert tp.Paper.parse(open(a).read(), a).render() == open(a).read()
    rep.text()


def test_code_change():
    d, a, b = setup()
    run(a)
    edit(a, "d.x * 10", "d.x * 20")                               # new code: every call is a different call
    rep = paper(a).verify()
    assert rep.logged == 0 and all(s.startswith("open") for s in rep.claims.values())
    run(a)
    rep = paper(a).verify()
    assert rep.claims["c2"] == "fails" and not rep.ok and not rep.problems     # a failing claim is not a format problem
    edit(a, "== 60.0", "== 120.0")
    rep = paper(a).verify()
    assert rep.claims["c2"] == "holds" and rep.ok                 # a claim is not code a call depends on: the log stands
    assert len(paper(a).log.entries) == 6                         # the log holds only the calls the paper makes


def test_invalid_and_too_large():
    d, a, b = setup()
    edit(a, "    return dict(w=d.x * 10)", "    return dict(w=d.x * 10, raw=list(range(20000)))")
    said = [g.error for g in run(a) if g.error]
    assert any("the limit is" in s and "summary" in s for s in said)
    rep = paper(a).verify()
    assert rep.logged == 4 and rep.claims["c2"].startswith("open")
    edit(a, 'call("big", fit, d=nums, scale=2', 'call("big", fit, device="cuda", d=nums, scale=2')
    rep = paper(a).verify()
    assert not rep.valid and any("device" in x for x in rep.problems)


def test_unreachable():
    d, a, b = setup()
    run(a), run(b)
    assert paper(b).verify().claims["not_so"] == "holds"
    os.remove(os.path.join(d, "nums.txt"))                        # the data's URL is dead, the cache still has it
    assert paper(b).verify().ok
    shutil.rmtree(os.path.join(d, "cache"))
    rep = paper(b).verify()
    assert not rep.ok and any("data file" in x for x in rep.problems)
    open(os.path.join(d, "nums.txt"), "w").write("1\n2\n3\n")
    edit(b, "at: ./a.md", "at: ./a.md\ncode: abcd")
    rep = paper(b).verify()
    assert not rep.ok and "pinned" in rep.problems[0]
    edit(b, "code: abcd", "code: '" + paper(a).code_hash + "'")
    assert paper(b).verify().ok
    os.rename(a, a + ".gone")
    rep = paper(b).verify()
    assert not rep.ok and "unreachable" in rep.problems[0]


def test_challenges_and_vault():
    d, a, b = setup("abc")
    v = tp.Vault(os.path.dirname(a))
    assert v.name == "toy" and list(v.papers) == ["a", "b", "c"] and v["a"].path == a
    assert paper(a).verify().challenged_by == [dict(paper="b.md", claim="not_so", challenges="c1")]
    assert v.tidy() == ["a", "b", "c"] and v.tidy() == [] and "\nCites: [a](a.md)\n" in open(b).read()
    text = open(b).read()                                         # a viewer hides the fence's words, so tidy labels each block
    assert "**Code `use`**\n\n```python code=use" in text and "**Experiments**\n\n```python experiments" in text
    assert "**Claim `not_so`** (stated 2026-11-16, challenges `a.c1`)\n\n```python claim=not_so" in text
    edit(b, "def not_so(r):\n", 'def not_so(r):\n    """Scale 3 gives a larger mean than scale 2."""\n')
    assert v.tidy() == ["b"] and "challenges `a.c1`): Scale 3 gives a larger mean than scale 2.\n\n```python claim" in open(b).read()
    assert v["b"].verify().statements == {"not_so": "Scale 3 gives a larger mean than scale 2."} and open(b).read().count("**Claim") == 1
    assert "\nClaims: ❌ 0 of 2 hold\n\n" in open(a).read()        # nothing is logged yet
    for x in v:
        run(x.path)
    assert any("paper.tidy()" in x for x in v["a"].verify().problems)          # the line it carries is now out of date
    assert v.tidy() == ["a", "b", "c"] and "\nClaims: ✅ 2 of 2 hold\n\nSome prose" in open(a).read()
    assert "\nClaims: ✅ 1 of 1 hold\n\nCites: [a](a.md)\n\n" in open(b).read()
    reports = v.verify()
    assert all(r.ok for r in reports.values()) and reports["c"].claims == {"only": "holds"}
    assert [x.stem for x in v["a"].cited_by] == ["b", "c"] and v["c"].cited_by == []
    assert v.claims("holds") == {"a.c1": "holds", "a.c2": "holds", "b.not_so": "holds", "c.only": "holds"} and v.claims("fails") == {}
    r = v["c"].reproduce(device="cpu", say=quiet)                 # a paper with no calls: its claims read the cited logs
    assert r.claims == {"only": "holds"} and (r.calls, r.rerun) == (0, 0), r


def test_reproduce():
    d, a, b = setup()
    run(a)
    assert not os.path.exists(os.path.join(d, "toy", "logs"))     # the author left nothing beside the paper
    r = paper(a).reproduce(time=0.0, device="cpu", say=quiet)     # no time: nothing is started
    assert r.verified and r.started == 0 and r.claims["c1"].startswith("not attempted"), r
    r = paper(a).reproduce(device="cpu", full=False, say=quiet)   # the first points only
    assert (r.started, r.compared, r.matched, r.attempted) == (6, 4, 4, 0) and r.rerun == 2, r   # two calls emit nothing, so they finished
    r = paper(a).reproduce(time="10m", device="cpu", say=quiet)   # continue the same rerun
    assert r.rerun == 6 and r.claims == dict(c1="holds", c2="holds") and r.needed == 0 and (r.points, r.points_agree) == (80, 80), r
    assert len(paper(a).log.entries) == 6 and paper(a).verify().ok       # the paper itself is untouched
    old = re.search(r'"emitted":\[\{"step":0,"loss":[^,]+,', open(a).read()).group(0)
    edit(a, old, '"emitted":[{"step":0,"loss":0.123,')            # a logged point the rerun does not give
    r = paper(a).reproduce(device="cpu", say=quiet)
    assert (r.compared, r.matched) == (4, 3) and (r.points, r.points_agree) == (80, 60) and r.held == 2, r
    r.text()


def test_log_written_by_hand():
    d, a, b = setup()
    run(a)
    paper(a).tidy()
    full = open(a).read()
    assert "\n\n**Log**\n\n```log\n" in full
    edit(a, full[full.index("```log"):], re.sub(r'"call":"[0-9a-f]+",', "", full[full.index("```log"):]))    # entries without ids
    p = paper(a)
    assert all(e.call is None for e in p.log.entries)
    rep = p.verify()
    assert rep.ok and rep.logged == 6 and rep.claims == dict(c1="holds", c2="holds") and rep.notes, rep.problems
    r = p.reproduce(device="cpu", full=False, say=quiet)
    assert (r.compared, r.matched) == (4, 4), r
    assert p.tidy() and open(a).read() == full and not paper(a).tidy()         # tidy writes the ids in
    edit(a, '"fn":"fn:use","args":{"init"', '"fn":"fn:use","args":{"other":1,"init"')
    edit(a, re.search(r'"call":"[0-9a-f]+","fn":"fn:use"', open(a).read()).group(0), '"fn":"fn:use"')
    rep = paper(a).verify()
    assert not rep.ok and any("matches no call" in x for x in rep.problems)     # an entry for a call the paper does not make


def test_example_in_format():
    d, a, b = setup()
    x = os.path.join(d, "toy", "x.md")
    open(x, "w").write(EXAMPLE)                                   # the paper format.py shows
    rep = paper(x).verify()
    assert rep.valid and not rep.problems and (rep.calls, rep.logged) == (4, 1), rep.problems
    edit(x, EXAMPLE[EXAMPLE.index("```log"):], "")
    points = None
    for dcl in paper(x).calls():                                  # the recipe in Log's docstring
        with tp.capture() as points:
            out = dcl.fn(seed=dcl.seed, **dcl.kw)
        assert len(points) == 14 and "t" in points[0] and out["lr"] == dcl.args["lr"]
    run(x)
    rep = paper(x).verify()
    assert rep.ok and rep.claims == {"larger_is_faster": "holds"}, rep.problems


def test_cite_from_another_vault():
    d, a, b = setup()
    elsewhere = os.path.join(d, "elsewhere.md")
    shutil.move(a, elsewhere)                                     # the cited paper lives in another vault
    edit(b, "at: ./a.md", f"at: ./a.md\nsource: file://{elsewhere}")
    rep = paper(b).verify()
    assert not rep.problems and os.path.exists(a), rep.problems             # it was copied into this vault
    edit(b, "at: ./a.md", "at: https://example.org/a.md")
    assert "local path" in paper(b).verify().problems[0]


def test_source():
    d, a, b = setup()
    edit(a, "def use(init):\n    return dict(total=float(sum(init[\"w\"])))",
         "class Thing:\n    def size(self):\n        return 1\n\n\ndef use(init):\n    from tinypaper import source\n"
         "    return dict(total=float(sum(init[\"w\"])), chars=len(source(use)), cls=source(Thing).splitlines()[0])")
    run(a)
    out = [e for e in paper(a).log.entries if e.fn == "fn:use"][0].out
    assert out["chars"] > 50 and out["cls"] == "class Thing:", out   # a call can read its own source, for example to charge for its length


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
