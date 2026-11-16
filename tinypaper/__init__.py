"""TinyPaper: papers that can be checked by running them. See SPEC.md."""
from .records import emit, Stop, Run, Runs
from .store import Unreachable
from .paper import Paper
from .vault import Vault
from .check import Report
from .reproduce import Reproduction, TIERS


def check(path):
    return Paper.load(path).check()


def run_paper(path, **kw):
    return Paper.load(path).run(**kw)


def reproduce(path, **kw):
    return Paper.load(path).reproduce(**kw)


__all__ = ["Paper", "Vault", "Report", "Reproduction", "TIERS", "Unreachable", "emit", "Stop", "Run", "Runs",
           "check", "run_paper", "reproduce"]
