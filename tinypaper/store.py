"""Fetching what a paper names by address: data files, and cited papers from another vault."""
import hashlib, os, urllib.request

CACHE = os.environ.get("TINYPAPER_CACHE", os.path.expanduser("~/.cache/tinypaper"))


class Unreachable(Exception):
    """Something a paper needs to be rerun cannot be found, or its hash does not match."""


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def data_file(url, digest, verify=False):
    """Local path of a data file: the cache, then its URL. The hash must match."""
    path = os.path.join(CACHE, "files", digest)
    if os.path.exists(path) and (not verify or file_sha(path) == digest):
        return path
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
