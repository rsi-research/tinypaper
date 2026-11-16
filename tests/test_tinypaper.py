"""Tests on a toy vault that needs only numpy. Run with pytest, or directly with python."""
import hashlib, json, os, re, shutil, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tinypaper as tp
from tinypaper import records, store

A = '''---
title: Toy root
tinypaper: 0.3
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
    return dict(vals=d.x * scale, mean=float(d.x.mean() * scale))


def make(d):
    return dict(w=d.x * 10)


def use(init):
    return dict(total=float(init["w"].sum()))
```

```python experiments
run("small", fit, d=nums, scale=1, env=cpu, seeds=2)
run("big", fit, d=nums, scale=2, env=cpu, seeds=2)
pre = run("pre", make, d=nums)
run("post", use, init=pre)
```

```python claim=c1 stated=2026-11-15 differs=scale
def c1(r):
    return r.big[0].out["mean"] > r.small[0].out["mean"]
```

```python claim=c2 stated=2026-11-15
def c2(r):
    return r.post[0].out["total"] == 60.0
```

```python claim=c3 stated=2026-11-15 differs=scale
def c3(r):
    return float(r.big[0].out["vals"].sum()) == 2 * float(r.small[0].out["vals"].sum())
```
'''

B = '''---
title: Toy challenge
tinypaper: 0.3
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
tinypaper: 0.3
date: 2026-11-15
---

```cite
name: a
at: ./a.md
```

```python claim=only stated=2026-11-15 differs=scale
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
    open(os.path.join(v, "tinypaper.toml"), "w").write(f'name = "toy"\nstore = "{d}/store"\n')
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
    assert [x.name for x in p.code] == ["all"] and list(p.envs) == ["cpu"] and [c.id for c in p.claims] == ["c1", "c2", "c3"]
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
    assert rep.ok and rep.evidence == "none" and rep.claims["c1"]["paper"].startswith("open"), rep.problems
    assert len(paper(a).run(name="mine", device="cpu", say=quiet)) == 6
    p = paper(a)
    assert p.results.set == "mine" and len(p.results.rows) == 6 and "```results" in open(a).read()
    row = [r for r in p.results.rows if r.experiment == "big"][0]
    assert row.out == {"mean": 4.0} and row.logged == ["vals"] and len(row.prefix["loss"]) == 10
    rep = p.check()
    assert rep.ok and rep.evidence == "logged", rep.problems
    assert {c: s["paper"] for c, s in rep.claims.items()} == dict(c1="holds", c2="holds", c3="holds")
    assert rep.runs["paper"] == dict(fresh=6, stale=0, missing=0) and any("numpy" in m for m in rep.ran_differently["paper"])
    assert rep.log["runs"] == 6 and p.run(name="mine", device="cpu", say=quiet) == []
    assert tp.Paper.parse(open(a).read(), a).render() == open(a).read()
    rep.text()


def test_paper_alone_then_logs():
    d, a, b = setup()
    paper(a).run(name="mine", device="cpu", say=quiet)
    assert paper(a).push() == "toy/a.json"
    shutil.rmtree(os.path.join(d, "toy", "logs"))                 # a clone: the paper, without logs
    tp.paper.PAPERS.clear()
    rep = paper(a).check()
    assert rep.ok and rep.evidence == "reported"
    assert rep.claims["c1"]["paper"] == "holds" and rep.claims["c3"]["paper"].startswith("open") and "logs" in rep.claims["c3"]["paper"]
    assert paper(a).load_logs()
    rep = paper(a).check()
    assert rep.evidence == "logged" and rep.claims["c3"]["paper"] == "holds"
    edit(a, '"mean":4.0', '"mean":5.0')                           # a reported number that the logs contradict
    rep = paper(a).check()
    assert not rep.ok and any("the logs say" in x for x in rep.problems)


def test_stale_and_log():
    d, a, b = setup()
    paper(a).run(name="mine", device="cpu", say=quiet)
    edit(a, "d.x * 10", "d.x * 20")
    rep = paper(a).check()
    assert rep.runs["paper"]["stale"] == 6 and all(s["paper"].startswith("open") for s in rep.claims.values())
    paper(a).run(name="mine", device="cpu", say=quiet)
    rep = paper(a).check()
    assert rep.claims["c2"]["paper"] == "fails" and rep.log["versions"] == 2
    edit(a, "== 60.0", "== 120.0")                                # edit a claim after seeing the result
    edit(a, 'run("small", fit, d=nums, scale=1', 'run("small", fit, d=nums, scale=4')
    paper(a).run(name="mine", device="cpu", say=quiet)
    rep = paper(a).check()
    assert rep.claims["c2"]["paper"] == "holds" and rep.log["claims_edited"] == ["c2"] and rep.log["undeclared"] == 2
    assert len(paper(a).results.rows) == 6                        # the paper reports only the runs it still declares


def test_invalid():
    d, a, b = setup()
    paper(a).run(name="mine", device="cpu", say=quiet)
    edit(a, "claim=c1 stated=2026-11-15 differs=scale", "claim=c1 stated=2026-11-15 differs=steps")
    rep = paper(a).check()
    assert not rep.ok and any("differs" in x for x in rep.problems)
    edit(a, 'run("big", fit, d=nums, scale=2', 'run("big", fit, device="cuda", d=nums, scale=2')
    rep = paper(a).check()
    assert not rep.valid and any("device" in x for x in rep.problems)


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
    out = v.to_jsonl(os.path.join(d, "vault.jsonl"), logs=True)
    lines = [json.loads(l) for l in open(out)]
    assert [l["path"] for l in lines] == ["a.md", "b.md", "c.md"] and "mine" in lines[0]["logs"]["sets"]
    r = paper(os.path.join(os.path.dirname(a), "c.md")).reproduce(device="cpu", say=quiet)
    assert r.tier == "full" and r.claims == {"only": "holds"}, r


def test_reproduce():
    d, a, b = setup()
    assert paper(a).reproduce(upto="checked").tier == "checked"
    paper(a).run(name="mine", device="cpu", say=quiet)
    shutil.rmtree(os.path.join(d, "toy", "logs"))                 # reproduce from the paper alone
    tp.paper.PAPERS.clear()
    r = paper(a).reproduce(upto="starts", device="cpu", name="r1", say=quiet)
    assert r.tier == "starts" and r.coverage["starts"] == "6 of 6 runs started"
    r = paper(a).reproduce(time=0.0, device="cpu", name="r2", say=quiet)        # no time: nothing is started
    assert r.tier == "checked" and r.claims["c1"].startswith("not attempted"), r
    r = paper(a).reproduce(upto="prefix", device="cpu", name="r3", say=quiet)
    assert r.tier == "prefix" and "4 of 4 runs match the paper" in r.coverage["prefix"] and "exactly" in r.coverage["prefix"], r
    assert {x.status for x in paper(a).logs.sets["r3"].runs.values() if x.experiment == "small"} == {"stopped"}
    r = paper(a).reproduce(time="10m", device="cpu", name="r3", say=quiet)      # continue the same rerun
    assert r.tier == "full" and r.claims == dict(c1="holds", c2="holds", c3="holds") and r.needed == 0, r
    assert "```results" in open(a).read() and paper(a).results.set == "mine"     # the paper itself is untouched
    r = paper(a).reproduce(seeds="fresh", device="cpu", say=quiet)
    assert r.tier == "robust" and r.set == "repro-cpu-fresh" and r.claims["c1"] == "holds", r
    edit(a, re.search(r'"prefix":\{"step":\[0,1,2,3,4,5,6,7,8,9\],"loss":\[[^,]+,', open(a).read()).group(0),
         '"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[0.123,')              # a reported value the rerun does not give
    r = paper(a).reproduce(upto="prefix", device="cpu", name="r4", say=quiet)
    assert r.tier == "starts" and "3 of 4 runs match" in r.coverage["prefix"], r
    r = paper(a).reproduce(device="cpu", name="r4", say=quiet)
    assert r.tier == "full" and any("`prefix` did not pass" in w for w in r.warnings), r
    r.text()


def test_two_agents_one_store():
    d, a, b = setup()
    paper(a).run(name="mine", device="cpu", say=quiet)
    other = os.path.join(d, "clone")
    shutil.copytree(os.path.dirname(a), os.path.join(other, "toy"))
    a2 = os.path.join(other, "toy", "a.md")
    paper(a).reproduce(device="cpu", name="agent-1", say=quiet)
    paper(a2).reproduce(device="cpu", name="agent-2", say=quiet)
    paper(a).push()
    paper(a2).push()                                              # merged with what the first agent uploaded
    there = json.load(open(os.path.join(d, "store", "toy", "a.json")))
    assert set(there["sets"]) == {"mine", "agent-1", "agent-2"}
    assert paper(a).load_logs() and set(paper(a).logs.sets) == {"mine", "agent-1", "agent-2"}
    rep = paper(a).check()
    assert rep.ok and rep.claims["c1"] == {"paper": "holds", "agent-1": "holds", "agent-2": "holds"}


def test_large_arrays():
    d, a, b = setup()
    records.LARGE = 8                                             # treat even small arrays as too large for the logs file
    try:
        paper(a).run(name="mine", device="cpu", say=quiet)
        assert '"$file"' in open(paper(a).logs_path).read()
        paper(a).push()
        shutil.rmtree(os.path.join(d, "toy", "logs"))
        tp.paper.PAPERS.clear()
        assert paper(a).load_logs() and paper(a).check().claims["c3"]["paper"] == "holds"
    finally:
        records.LARGE = 10 * 1024 * 1024


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
