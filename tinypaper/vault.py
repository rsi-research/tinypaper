"""Vault: a folder of papers."""
import os

VAULTS = {}


class Vault:
    """A flat folder of papers. Its name is the folder's name."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path)

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
            for c in p.cites.values():
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

    def run(self, device=None, say=print):
        """Run every paper, cited papers first."""
        return [got for p in self for got in p.run(device, say=say)]

    def tidy(self):
        """Bring every paper's derived parts up to date. Returns the names of the papers that changed."""
        return [name for name, p in self.papers.items() if p.tidy()]
