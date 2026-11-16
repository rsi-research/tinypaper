"""Tests on a toy vault that needs only numpy. Run with pytest, or directly with python."""
import hashlib, os, re, shutil, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tinypaper as tp
from tinypaper import store

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
    return paper(path).run(device="cpu", say=quiet)


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


def test_run_and_check():
    d, a, b = setup()
    rep = paper(a).check()
    assert rep.ok and (rep.calls, rep.logged) == (6, 0) and rep.claims["c1"].startswith("open"), rep.problems
    assert len(paper(a).declared()) == 8 and len(paper(a).calls()) == 6       # two experiments share two calls
    assert len(run(a)) == 6
    p = paper(a)
    assert len(p.log.entries) == 6 and "```log" in open(a).read()
    big = [e for e in p.log.entries if e.args.get("scale") == 2][0]
    assert big.out == {"mean": 4.0, "top": 6.0} and len(big.emitted) == 20 and big.emitted[0]["step"] == 0
    rep = p.check()
    assert rep.ok and rep.claims == dict(c1="holds", c2="holds") and rep.logged == 6, rep.problems
    assert rep.compares == {"c1": ["scale"], "c2": ["fn"]} and any("numpy" in m for m in rep.ran_differently)
    assert rep.history["runs"] == 6 and p.log.history["runs"] == 6 and run(a) == []
    assert tp.Paper.parse(open(a).read(), a).render() == open(a).read()
    rep.text()
    shutil.rmtree(os.path.join(d, "toy", "logs"))                 # a clone has the paper and nothing else
    tp.paper.PAPERS.clear()
    rep = paper(a).check()
    assert rep.ok and rep.claims["c1"] == "holds" and rep.history["runs"] == 6


def test_code_change_and_history():
    d, a, b = setup()
    run(a)
    edit(a, "d.x * 10", "d.x * 20")                               # new code: every call is a different call
    rep = paper(a).check()
    assert rep.logged == 0 and all(s.startswith("open") for s in rep.claims.values())
    run(a)
    rep = paper(a).check()
    assert rep.claims["c2"] == "fails" and rep.history["runs"] == 12 and rep.history["no_longer_made"] == 6
    edit(a, "== 60.0", "== 120.0")                                # edit a claim after seeing the result
    run(a)
    rep = paper(a).check()
    assert rep.claims["c2"] == "holds" and rep.history["claims_edited"] == ["c2"]
    assert len(paper(a).log.entries) == 6                         # the log holds only the calls the paper makes


def test_invalid_and_too_large():
    d, a, b = setup()
    edit(a, "    return dict(w=d.x * 10)", "    return dict(w=d.x * 10, raw=list(range(20000)))")
    said = []
    paper(a).run(device="cpu", say=said.append)
    assert any("the limit is" in s and "summary" in s for s in said)
    rep = paper(a).check()
    assert rep.logged == 4 and rep.claims["c2"].startswith("open")
    edit(a, 'call("big", fit, d=nums, scale=2', 'call("big", fit, device="cuda", d=nums, scale=2')
    rep = paper(a).check()
    assert not rep.valid and any("device" in x for x in rep.problems)


def test_unreachable():
    d, a, b = setup()
    run(a), run(b)
    assert paper(b).check().claims["not_so"] == "holds"
    os.remove(os.path.join(d, "nums.txt"))                        # the data's URL is dead, the cache still has it
    assert paper(b).check().ok
    shutil.rmtree(os.path.join(d, "cache"))
    rep = paper(b).check()
    assert not rep.ok and any("data file" in x for x in rep.problems)
    open(os.path.join(d, "nums.txt"), "w").write("1\n2\n3\n")
    edit(b, "at: ./a.md", "at: ./a.md\ncode: abcd")
    rep = paper(b).check()
    assert not rep.ok and "pinned" in rep.problems[0]
    edit(b, "code: abcd", "code: '" + paper(a).code_hash + "'")
    assert paper(b).check().ok
    os.rename(a, a + ".gone")
    rep = paper(b).check()
    assert not rep.ok and "unreachable" in rep.problems[0]


def test_challenges_and_vault():
    d, a, b = setup("abc")
    v = tp.Vault(os.path.dirname(a))
    assert v.name == "toy" and list(v.papers) == ["a", "b", "c"] and v["a"].path == a
    assert paper(a).check().challenged_by == [dict(paper="b.md", claim="not_so", challenges="c1")]
    assert v.tidy() == ["b", "c"] and v.tidy() == [] and "\nCites: [a](a.md)\n" in open(b).read()
    v.run(device="cpu", say=quiet)
    reports = v.check()
    assert all(r.ok for r in reports.values()) and reports["c"].claims == {"only": "holds"}
    r = v["c"].reproduce(device="cpu", say=quiet)                 # a paper with no calls: its claims read the cited logs
    assert r.claims == {"only": "holds"} and (r.calls, r.rerun) == (0, 0), r


def test_reproduce():
    d, a, b = setup()
    run(a)
    shutil.rmtree(os.path.join(d, "toy", "logs"))                 # reproduce from the paper alone
    tp.paper.PAPERS.clear()
    r = paper(a).reproduce(time=0.0, device="cpu", say=quiet)     # no time: nothing is started
    assert r.checked and r.started == 0 and r.claims["c1"].startswith("not attempted"), r
    r = paper(a).reproduce(device="cpu", full=False, say=quiet)   # the first points only
    assert (r.started, r.compared, r.matched, r.attempted) == (6, 4, 4, 0) and r.rerun == 2, r   # two calls emit nothing, so they finished
    r = paper(a).reproduce(time="10m", device="cpu", say=quiet)   # continue the same rerun
    assert r.rerun == 6 and r.claims == dict(c1="holds", c2="holds") and r.needed == 0 and (r.points, r.points_agree) == (80, 80), r
    assert len(paper(a).log.entries) == 6 and paper(a).check().ok       # the paper itself is untouched
    old = re.search(r'"emitted":\[\{"step":0,"loss":[^,]+,', open(a).read()).group(0)
    edit(a, old, '"emitted":[{"step":0,"loss":0.123,')            # a logged point the rerun does not give
    r = paper(a).reproduce(device="cpu", say=quiet)
    assert (r.compared, r.matched) == (4, 3) and (r.points, r.points_agree) == (80, 60) and r.held == 2, r
    r.text()


WORKER = '''
import sys
sys.path.insert(0, sys.argv[1])
import tinypaper as tp
tp.store.CACHE = sys.argv[3]
print(len(tp.Paper.load(sys.argv[2]).run(device="cpu", say=lambda *a: None)))
'''


def test_parallel_runs():
    d, a, b = setup()
    edit(a, "    rng = np.random.default_rng(seed)\n", "    rng = np.random.default_rng(seed)\n    import time\n    time.sleep(0.4)\n")
    paper(a).load_data("nums")                                    # fetch the data once, before the workers start
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workers = [subprocess.Popen([sys.executable, "-c", WORKER, root, a, store.CACHE], stdout=subprocess.PIPE, text=True) for _ in range(3)]
    counts = [int(w.communicate()[0].strip().splitlines()[-1]) for w in workers]
    tp.paper.PAPERS.clear()
    p = paper(a)
    runs = [e for e in p.local.history if e["event"] == "run"]
    assert sum(counts) == 6 and len(runs) == 6 and len({e["call"] for e in runs}) == 6, (counts, len(runs))    # each call ran once
    assert len(p.log.entries) == 6 and p.check().ok and max(counts) < 6, counts                               # and the work was shared


def test_cite_from_another_vault():
    d, a, b = setup()
    elsewhere = os.path.join(d, "elsewhere.md")
    shutil.move(a, elsewhere)                                     # the cited paper lives in another vault
    edit(b, "at: ./a.md", f"at: ./a.md\nsource: file://{elsewhere}")
    rep = paper(b).check()
    assert rep.ok and os.path.exists(a), rep.problems             # it was copied into this vault
    edit(b, "at: ./a.md", "at: https://example.org/a.md")
    assert "local path" in paper(b).check().problems[0]


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
