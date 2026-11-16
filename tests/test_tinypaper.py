"""Tests on a toy vault that needs only numpy. Run with pytest, or directly with python."""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tinypaper as tp
from tinypaper import store

A = '''---
title: Toy root
tinypaper: 0.4
date: 2026-11-15
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
run("small", fit, d=nums, scale=1, env=cpu, seeds=2)
run("big", fit, d=nums, scale=2, env=cpu, seeds=2)
run("pair", fit, d=nums, scale=1, env=cpu, seeds=2)
pre = run("pre", make, d=nums)
run("post", use, init=pre)
```

```python claim=c1 stated=2026-11-15 differs=scale
def c1(r):
    return r.big[0].out["mean"] > r.small[0].out["mean"]
```

```python claim=c2 stated=2026-11-15
def c2(r):
    return r.post[0].out["total"] == 60.0 and r.pair[1].out == r.small[1].out
```
'''

B = '''---
title: Toy challenge
tinypaper: 0.4
date: 2026-11-15
---

```cite
name: a
at: ./a.md
```

```python code=use
from a import fit
```

```python experiments
run("huge", fit, d=a.nums, scale=3, seeds=1)
```

```python claim=not_so stated=2026-11-15 differs=scale challenges=a.c1
def not_so(r):
    return r.huge[0].out["mean"] > r.a.big[0].out["mean"]
```
'''

C = '''---
title: Toy analysis
tinypaper: 0.4
date: 2026-11-15
---

```cite
name: a
at: ./a.md
```

```python claim=only stated=2026-11-15
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


def test_parse_render_dump():
    d, a, b = setup()
    p = paper(a)
    assert p.render() == open(a).read()
    assert [x.name for x in p.code] == ["all"] and list(p.envs) == ["cpu"] and [c.id for c in p.claims] == ["c1", "c2"]
    assert p.data["nums"].files["nums"].sha256 and p.header.title == "Toy root"
    again = tp.Paper.from_dump(json.loads(json.dumps(p.dump())), folder=os.path.dirname(a))
    assert again.render() == open(a).read() and again.code_hash == p.code_hash
    for part in again.parts:                       # written from the fields alone, it parses to the same paper
        if part.kind != "prose":
            part.raw = None
    regen = tp.Paper.parse(again.render(), a)
    assert regen.code_hash == p.code_hash and [c.model_dump(exclude={"raw"}) for c in regen.claims] == [c.model_dump(exclude={"raw"}) for c in p.claims]
    edit(a, "hardware: 1 CPU", "hardware: 1 CPU\ncolour: red")
    assert any("colour" in x for x in paper(a).problems)          # an unknown field is an error


def test_run_and_check():
    d, a, b = setup()
    rep = paper(a).check()
    assert rep.ok and rep.calls["paper"] == dict(logged=0, missing=6) and rep.claims["c1"]["paper"].startswith("open"), rep.problems
    assert len(paper(a).declared()) == 8 and len(paper(a).calls()) == 6       # two experiments share two calls
    assert len(paper(a).run(name="mine", device="cpu", say=quiet)) == 6
    p = paper(a)
    assert p.log.name == "mine" and len(p.log.entries) == 6 and "```log" in open(a).read()
    big = [e for e in p.log.entries if e.args.get("scale") == 2][0]
    assert big.out == {"mean": 4.0, "top": 6.0} and len(big.emitted) == 20 and big.emitted[0]["step"] == 0
    rep = p.check()
    assert rep.ok and {c: s["paper"] for c, s in rep.claims.items()} == dict(c1="holds", c2="holds"), rep.problems
    assert rep.calls["paper"] == dict(logged=6, missing=0) and any("numpy" in m for m in rep.ran_differently["paper"])
    assert rep.history["runs"] == 6 and p.log.history["runs"] == 6 and p.run(name="mine", device="cpu", say=quiet) == []
    assert tp.Paper.parse(open(a).read(), a).render() == open(a).read()
    rep.text()
    shutil.rmtree(os.path.join(d, "toy", "logs"))                 # a clone has the paper and nothing else
    tp.paper.PAPERS.clear()
    rep = paper(a).check()
    assert rep.ok and rep.claims["c1"]["paper"] == "holds" and rep.history["runs"] == 6


def test_code_change_and_history():
    d, a, b = setup()
    paper(a).run(name="mine", device="cpu", say=quiet)
    edit(a, "d.x * 10", "d.x * 20")                               # new code: every call is a different call
    rep = paper(a).check()
    assert rep.calls["paper"] == dict(logged=0, missing=6) and all(s["paper"].startswith("open") for s in rep.claims.values())
    paper(a).run(name="mine", device="cpu", say=quiet)
    rep = paper(a).check()
    assert rep.claims["c2"]["paper"] == "fails" and rep.history["runs"] == 12 and rep.history["no_longer_made"] == 6
    edit(a, "== 60.0", "== 120.0")                                # edit a claim after seeing the result
    paper(a).run(name="mine", device="cpu", say=quiet)
    rep = paper(a).check()
    assert rep.claims["c2"]["paper"] == "holds" and rep.history["claims_edited"] == ["c2"]
    assert len(paper(a).log.entries) == 6                         # the log holds only the calls the paper makes


def test_invalid_and_differs():
    d, a, b = setup()
    paper(a).run(name="mine", device="cpu", say=quiet)
    edit(a, "claim=c1 stated=2026-11-15 differs=scale", "claim=c1 stated=2026-11-15 differs=steps")
    rep = paper(a).check()
    assert rep.valid and not rep.ok and any("differs" in x for x in rep.problems)     # wrong, but the file is well formed
    edit(a, "claim=c1 stated=2026-11-15 differs=steps", "claim=c1 stated=2026-11-15")
    assert paper(a).check().ok                                    # the annotation is optional
    edit(a, 'run("big", fit, d=nums, scale=2', 'run("big", fit, device="cuda", d=nums, scale=2')
    rep = paper(a).check()
    assert not rep.valid and any("device" in x for x in rep.problems)


def test_too_large():
    d, a, b = setup()
    edit(a, "    return dict(w=d.x * 10)", "    return dict(w=d.x * 10, raw=list(range(20000)))")
    said = []
    paper(a).run(name="mine", device="cpu", say=said.append)
    assert any("the limit is" in s and "summary" in s for s in said)
    rep = paper(a).check()
    assert rep.calls["paper"]["missing"] == 2 and rep.claims["c2"]["paper"].startswith("open")


def test_unreachable():
    d, a, b = setup()
    paper(b).run(name="mine", device="cpu", deep=True, say=quiet)
    assert paper(b).check().claims["not_so"]["paper"] == "holds"
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
    v.run(name="mine", device="cpu", say=quiet)
    reports = v.check()
    assert all(r.ok for r in reports.values()) and reports["c"].claims == {"only": {"paper": "holds"}}
    lines = [json.loads(l) for l in open(v.to_jsonl(os.path.join(d, "vault.jsonl")))]
    assert [l["path"] for l in lines] == ["a.md", "b.md", "c.md"] and lines[0]["parts"][-2]["kind"] == "log"
    r = paper(os.path.join(os.path.dirname(a), "c.md")).reproduce(device="cpu", say=quiet)
    assert r.tier == "full" and r.claims == {"only": "holds"}, r


def test_reproduce():
    d, a, b = setup()
    assert paper(a).reproduce(upto="checked").tier == "checked"
    paper(a).run(name="mine", device="cpu", say=quiet)
    shutil.rmtree(os.path.join(d, "toy", "logs"))                 # reproduce from the paper alone
    tp.paper.PAPERS.clear()
    r = paper(a).reproduce(upto="starts", device="cpu", name="r1", say=quiet)
    assert r.tier == "starts" and r.coverage["starts"] == "6 of 6 calls started"
    r = paper(a).reproduce(time=0.0, device="cpu", name="r2", say=quiet)        # no time: nothing is started
    assert r.tier == "checked" and r.claims["c1"].startswith("not attempted"), r
    r = paper(a).reproduce(upto="prefix", device="cpu", name="r3", say=quiet)
    assert r.tier == "prefix" and "4 of 4 calls match the paper" in r.coverage["prefix"] and "exactly" in r.coverage["prefix"], r
    assert {e.status for e in paper(a).local.logs["r3"].entries.values() if e.args.get("scale")} == {"stopped"}
    r = paper(a).reproduce(time="10m", device="cpu", name="r3", say=quiet)      # continue the same rerun
    assert r.tier == "full" and r.claims == dict(c1="holds", c2="holds") and r.needed == 0, r
    assert "80 of 80 logged points agree" in r.coverage["full"]
    seen = paper(a).check().claims["c1"]
    assert paper(a).log.name == "mine" and seen["paper"] == "holds" and seen["r3"] == "holds" and seen["r1"].startswith("open"), seen
    r = paper(a).reproduce(seeds="fresh", device="cpu", say=quiet)
    assert r.tier == "robust" and r.log == "repro-cpu-fresh" and r.claims["c1"] == "holds", r
    old = re.search(r'"emitted":\[\{"step":0,"loss":[^,]+,', open(a).read()).group(0)
    edit(a, old, '"emitted":[{"step":0,"loss":0.123,')            # a logged point the rerun does not give
    r = paper(a).reproduce(upto="prefix", device="cpu", name="r4", say=quiet)
    assert r.tier == "starts" and "3 of 4 calls match" in r.coverage["prefix"], r
    r = paper(a).reproduce(device="cpu", name="r4", say=quiet)
    assert r.tier == "full" and any("`prefix` did not pass" in w for w in r.warnings) and "60 of 80 logged points" in r.coverage["full"], r
    r.text()


WORKER = '''
import sys
sys.path.insert(0, sys.argv[1])
import tinypaper as tp
tp.store.CACHE = sys.argv[3]
print(len(tp.Paper.load(sys.argv[2]).run(name="mine", device="cpu", say=lambda *a: None)))
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


def test_tidy_and_source():
    d, a, b = setup("abc")
    v = tp.Vault(os.path.dirname(a))
    assert v.tidy() == ["b", "c"] and v.tidy() == []
    assert "\nCites: [a](a.md)\n" in open(b).read() and paper(b).check().ok
    edit(a, "def use(init):\n    return dict(total=float(sum(init[\"w\"])))",
         "class Thing:\n    def size(self):\n        return 1\n\n\ndef use(init):\n    from tinypaper import source\n"
         "    return dict(total=float(sum(init[\"w\"])), chars=len(source(use)), cls=source(Thing).splitlines()[0])")
    paper(a).run(name="mine", device="cpu", say=quiet)
    out = [e for e in paper(a).log.entries if e.fn == "fn:use"][0].out
    assert out["chars"] > 50 and out["cls"] == "class Thing:", out   # a call can read its own source, for example to charge for its length


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
