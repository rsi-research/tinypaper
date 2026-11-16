"""The typed parts of a paper. Each is a strict model: an unknown field is an error."""
import hashlib, json
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
    raw: Optional[str] = None      # the block exactly as written; the paper renders back from these


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
    challenges: Optional[str] = None
    source: str

    @property
    def hash(self):
        return sha(self.source)[:16]


class Entry(Strict):
    """What happened when one call ran: what its function returned and what it emitted."""
    call: str                      # the call's identity
    fn: str
    args: dict[str, Any]
    seed: Optional[int] = None
    seconds: float = 0
    out: Any = None
    emitted: list[dict[str, Any]] = []


class Log(Block):
    """The log block: the author's log, written into the paper by the runner."""
    kind: Literal["log"] = "log"
    ran_on: dict[str, Any] = {}
    history: dict[str, Any] = {}   # counts from the history: runs made, claims edited after results existed
    entries: list[Entry] = []


class Local(Strict):
    """What the tools keep beside the vault for one paper. None of it is part of the paper."""
    rerun: dict[str, Entry] = {}               # the calls a reproduction has finished on this machine
    history: list[dict[str, Any]] = []         # every run made and every claim edited, in order
    running: dict[str, dict[str, Any]] = {}    # calls some process is running now, and which process


@dataclass
class Declared:
    """One call an experiments block declares, under one experiment name. It holds live functions."""
    paper: Any
    name: str
    fn: Any
    kw: dict
    env: Any
    seed: Optional[int]
    args: dict

    @property
    def id(self):
        """The call's identity: its function, arguments, seed and code. The experiment name is not part of it."""
        return sha(json.dumps(["fn:" + self.fn.__name__, self.args, self.seed, self.paper.code_hash], sort_keys=True))[:16]


@dataclass
class Outcome:
    """One execution of a call: its entry if it finished, and how it ended."""
    entry: Optional[Entry]
    status: str                    # done, stopped or error
    error: Optional[str]
    emitted: list
    seconds: float
    machine: dict
