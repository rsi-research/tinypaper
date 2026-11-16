"""The typed parts of a paper and of its logs. Each is a strict model: an unknown field is an error."""
import hashlib
from dataclasses import dataclass
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Header(Strict):
    title: str
    tinypaper: str
    date: str

    @field_validator("tinypaper", "date", mode="before")
    @classmethod
    def text(cls, v):
        return str(v)


class Prose(Strict):
    kind: Literal["prose"] = "prose"
    text: str


class Block(Strict):
    raw: Optional[str] = None      # the block exactly as written, so a paper renders back unchanged


class Cite(Block):
    kind: Literal["cite"] = "cite"
    name: str
    at: str                        # a path relative to the citing paper: a cited paper is always a local file
    source: Optional[str] = None   # where the file came from, if another vault; it is fetched from there if missing
    code: Optional[str] = None     # if given, the cited paper's code hash must equal it

    @field_validator("code", mode="before")
    @classmethod
    def text(cls, v):
        return None if v is None else str(v)


class Env(Block):
    kind: Literal["env"] = "env"
    name: str
    hardware: str = ""
    packages: list[str] = []
    pitfalls: list[str] = []


class DataFile(Strict):
    url: str
    sha256: str


class Data(Block):
    kind: Literal["data"] = "data"
    name: str
    load: str
    files: Optional[dict[str, DataFile]] = None
    source: Optional[str] = Field(None, alias="from")
    args: dict[str, Any] = {}

    @model_validator(mode="after")
    def one_origin(self):
        if (self.files is None) == (self.source is None):
            raise ValueError("a data block has either `files` or `from`")
        return self


class Code(Block):
    kind: Literal["code"] = "code"
    name: str
    source: str


class Experiments(Block):
    kind: Literal["experiments"] = "experiments"
    source: str


class Claim(Block):
    kind: Literal["claim"] = "claim"
    id: str
    stated: str
    differs: list[str] = []
    challenges: Optional[str] = None
    source: str

    @property
    def hash(self):
        return sha(self.source)[:16]


class Row(Strict):
    """One run as the paper reports it: its identity and the small values it returned."""
    run: str
    experiment: str
    fn: str
    args: dict[str, Any]
    seed: Optional[int]
    code: str                      # hash of the code that produced it
    seconds: float
    out: Any = None                # the small part of what the run returned
    logged: list[str] = []         # the parts too large for the paper; they are in the logs
    prefix: dict[str, list] = {}   # the first emitted values, for comparing a rerun
    prefix_t: Optional[float] = None


class Results(Block):
    """The results block: the author's runs, written by the runner."""
    kind: Literal["results"] = "results"
    set: str
    ran_on: dict[str, Any]
    rows: list[Row] = []


class Record(Strict):
    """One run in full, as the logs hold it."""
    id: str
    experiment: str
    fn: str
    args: dict[str, Any]
    seed: Optional[int]
    env: Optional[str] = None
    code: str
    status: str                    # done, stopped or error
    error: Optional[str] = None
    seconds: float
    ran_on: dict[str, Any]
    emitted: list[dict[str, Any]] = []
    out: Any = None
    owner: Optional[str] = None    # while status is "running": the host and process doing it
    started: Optional[float] = None


class RecordSet(Strict):
    """One machine's runs of a paper."""
    seed_offset: int = 0
    runs: dict[str, Record] = {}


class Logs(Strict):
    """Everything about a paper's runs that does not fit in the paper. Optional, kept in the store."""
    sets: dict[str, RecordSet] = {}
    events: list[dict[str, Any]] = []      # the run log: every run made and every claim edited


@dataclass
class Declared:
    """One run an experiments block declares. It holds live functions, so it is not a model."""
    paper: Any
    name: str
    fn: Any
    kw: dict
    env: Any
    seed: Optional[int]
    args: dict

    def id(self, offset=0):
        import json
        seed = None if self.seed is None else self.seed + offset
        return sha(json.dumps([self.name, "fn:" + self.fn.__name__, self.args, seed], sort_keys=True))[:16]
