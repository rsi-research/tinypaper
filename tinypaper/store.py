"""Where things outside the vault are found: data files by URL, and logs in the store."""
import json, os, re, urllib.request
from .model import Logs, sha

CACHE = os.environ.get("TINYPAPER_CACHE", os.path.expanduser("~/.cache/tinypaper"))


class Unreachable(Exception):
    """Something a paper needs to be rerun cannot be found, or its hash does not match."""


def file_sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_env(start):
    """Read KEY=VALUE lines from the nearest `.env` above `start` into the environment.

    This is where tokens go. Variables that are already set are left alone."""
    d = os.path.abspath(start)
    while True:
        f = os.path.join(d, ".env")
        if os.path.isfile(f):
            for line in open(f):
                key, eq, value = line.strip().partition("=")
                if eq and key and not key.startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip("'\""))
            return
        if os.path.dirname(d) == d:
            return
        d = os.path.dirname(d)


def config(start):
    """The vault's name and store, from the nearest tinypaper.toml above `start`.

    Without one the name is the folder's name and there is no store. $TINYPAPER_STORE overrides the store."""
    load_env(start)
    d = start = os.path.abspath(start)
    out = dict(name=os.path.basename(start), store=None)
    while True:
        f = os.path.join(d, "tinypaper.toml")
        if os.path.exists(f):
            import tomllib
            found = tomllib.load(open(f, "rb"))
            out["name"] = found.get("name", out["name"])
            store = found.get("store")
            out["store"] = os.path.join(d, store) if store and not re.match(r"https?://|/", store) else store
            break
        if os.path.dirname(d) == d:
            break
        d = os.path.dirname(d)
    out["store"] = os.environ.get("TINYPAPER_STORE") or out["store"]
    return out


def get(url):
    request = urllib.request.Request(url)
    token = os.environ.get("TINYPAPER_TOKEN") or ("huggingface.co" in url and os.environ.get("HF_TOKEN"))
    if token:                             # a private store: the token comes from the environment, never from a paper
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=120) as r:
        return r.read()


def fetch(store, rel):
    """Bytes of `rel` in a store (a directory or an http base), or None."""
    if re.match(r"https?://", store):
        try:
            return get(store.rstrip("/") + "/" + rel)
        except Exception:
            return None
    path = os.path.join(store, rel)
    return open(path, "rb").read() if os.path.isfile(path) else None


def data_file(url, digest, verify=False):
    """Local path of a data file: the cache, then its URL. The hash must match."""
    path = os.path.join(CACHE, "files", digest)
    if os.path.exists(path) and (not verify or file_sha(path) == digest):
        return path
    import hashlib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part, h = path + ".part", hashlib.sha256()
    try:                                   # straight to disk, so a file larger than memory is fine
        with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                h.update(chunk)
                f.write(chunk)
    except Exception:
        h = None
    if h is None or h.hexdigest() != digest:
        if os.path.exists(part):
            os.remove(part)
        raise Unreachable(f"data file {url} (sha256 {digest[:12]}) not found with a matching hash")
    os.replace(part, path)
    return path


def merge(theirs, ours):
    """Two sets of logs as one. Where both have the same run, ours is kept."""
    out = Logs.model_validate(theirs.model_dump())
    for name, rs in ours.sets.items():
        if name not in out.sets:
            out.sets[name] = rs.model_copy(deep=True)
        else:
            out.sets[name].runs.update(rs.runs)
            out.sets[name].seed_offset = rs.seed_offset
    seen = {json.dumps(e, sort_keys=True) for e in out.events}
    out.events += [e for e in ours.events if json.dumps(e, sort_keys=True) not in seen]
    return out


HF = re.compile(r"https://huggingface.co/(datasets/)?([^/]+/[^/]+)/resolve/([^/]+)/?$")


def upload(store, files, message, expect=None):
    """Write {path: bytes} to a store. For Hugging Face this is one commit, refused if the
    repository moved past commit `expect` in the meantime; returns False on a refusal."""
    m = HF.match(store)
    if m:
        from huggingface_hub import CommitOperationAdd, HfApi
        from huggingface_hub.utils import HfHubHTTPError
        api, kind = HfApi(token=os.environ.get("HF_TOKEN")), "dataset" if m.group(1) else "model"
        try:
            api.create_commit(m.group(2), repo_type=kind, revision=m.group(3), commit_message=message, parent_commit=expect,
                              operations=[CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in files.items()])
        except HfHubHTTPError as e:
            if expect and getattr(e.response, "status_code", None) in (409, 412):
                return False
            raise
        return True
    if re.match(r"https?://", store):
        raise ValueError("a store that can be written is a directory or a Hugging Face repository")
    for k, v in files.items():
        os.makedirs(os.path.dirname(os.path.join(store, k)), exist_ok=True)
        with open(os.path.join(store, k), "wb") as f:
            f.write(v)
    return True


def head(store):
    """The store's current commit, if it has commits."""
    m = HF.match(store)
    if not m:
        return None
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    return api.repo_info(m.group(2), repo_type="dataset" if m.group(1) else "model", revision=m.group(3)).sha


def at_commit(store, commit):
    """The same store, read at one commit."""
    m = HF.match(store)
    return store if not (m and commit) else store.rstrip("/").rsplit("/", 1)[0] + "/" + commit
