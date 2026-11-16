"""Vault: a folder of papers."""
import json, os
from . import store

VAULTS = {}


class Vault:
    """A flat folder of papers. Its name and its store come from the nearest tinypaper.toml."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        cfg = store.config(self.path)
        self.name, self.store = cfg["name"], cfg["store"]

    @classmethod
    def of(cls, folder):
        folder = os.path.abspath(folder)
        if folder not in VAULTS:
            VAULTS[folder] = cls(folder)
        return VAULTS[folder]

    @property
    def papers(self):
        """{name: Paper} for every paper in the folder, cited papers before the papers that cite them."""
        from .paper import Paper
        found = {}
        for f in sorted(os.listdir(self.path)):
            path = os.path.join(self.path, f)
            if f.endswith(".md") and os.path.isfile(path) and open(path, errors="ignore").read(4) == "---\n":
                try:
                    paper = Paper.load(path)
                except Exception:
                    continue
                if paper.header is not None:
                    found[paper.stem] = paper
        ordered = {}

        def add(p):
            for c in p._cited.values():
                if c.stem in found and c.stem not in ordered:
                    add(c)
            ordered[p.stem] = p
        for p in found.values():
            add(p)
        return ordered

    def __getitem__(self, name):
        from .paper import Paper
        return Paper.load(os.path.join(self.path, name if name.endswith(".md") else name + ".md"))

    def __iter__(self):
        return iter(self.papers.values())

    def __len__(self):
        return len(self.papers)

    def check(self):
        return {name: p.check() for name, p in self.papers.items()}

    def run(self, name="author", device=None, say=print):
        return [rec for p in self for rec in p.run(name, device, say=say)]

    def tidy(self):
        """Bring every paper's derived parts up to date. Returns the names of the papers that changed."""
        return [name for name, p in self.papers.items() if p.tidy()]

    def load_logs(self):
        return {name: p.load_logs(deep=False) for name, p in self.papers.items()}

    def push(self):
        return [p.push() for p in self if p.logs.sets or p.logs.events]

    def to_jsonl(self, path, logs=False):
        """Every paper as one line of JSON, optionally with its logs."""
        with open(path, "w") as f:
            for p in self:
                f.write(json.dumps(p.dump(logs=logs)) + "\n")
        return path
