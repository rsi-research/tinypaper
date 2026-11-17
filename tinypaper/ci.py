"""What a CI job runs on a vault: verify the papers a change touches, and rerun them briefly.

    python -m tinypaper.ci example-vault --base <commit> --time 2m
    python -m tinypaper.ci my-vault --base <commit> --append-only     # for a shared vault: also refuse changes to published papers

The job needs no secrets and must not be given any: it executes the code in submitted papers."""
import argparse, os, subprocess, sys
from .vault import Vault


def touched(vault, base):
    """Names of the papers changed since commit `base`, and the papers that cite them. All papers
    if there is no base, or if anything outside the vault changed (the tools themselves)."""
    papers = vault.papers
    if not base or set(base) == {"0"}:
        return list(papers), "no base commit"
    try:
        out = subprocess.run(["git", "diff", "--name-only", base, "HEAD"], capture_output=True, text=True, check=True, cwd=vault.path).stdout
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True, cwd=vault.path).stdout.strip()
    except Exception:
        return list(papers), "the base commit could not be compared"
    files = [os.path.join(top, f) for f in out.split()]
    inside = [f for f in files if os.path.dirname(f) == vault.path and f.endswith(".md")]
    if len(inside) < len(files):
        return list(papers), "files outside the vault changed"
    changed = {os.path.basename(f)[:-3] for f in inside}
    more = True
    while more:                            # a paper that cites a changed paper may have gone stale
        more = False
        for name, p in papers.items():
            if name not in changed and any(c.stem in changed for c in p.cites.values()):
                changed.add(name)
                more = True
    return [n for n in papers if n in changed], "changed papers and the papers that cite them"


def changed_existing(vault, base):
    """Papers that existed at commit `base` and were changed or deleted since: [(status, file name)].
    A vault is append-only, so for a shared vault this should be empty."""
    out = subprocess.run(["git", "diff", "--name-status", "--no-renames", base, "HEAD", "--", "."], capture_output=True, text=True, check=True, cwd=vault.path).stdout
    found = []
    for line in out.splitlines():
        status, _, name = line.partition("\t")
        if name.endswith(".md") and "/" not in os.path.relpath(os.path.join(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True, cwd=vault.path).stdout.strip(), name), vault.path) and status[0] in "MD":
            found.append((status[0], os.path.basename(name)))
    return found


def install(papers):
    """Install the packages the papers' environments pin."""
    pins = sorted({pin for p in papers for e in p.envs.values() for pin in e.packages})
    if pins:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *pins], check=False)
    return pins


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("vault")
    ap.add_argument("--base", default=os.environ.get("BASE"))
    ap.add_argument("--time", default="2m", help="time limit for rerunning each paper")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--append-only", action="store_true", help="fail if a paper that already existed was changed or deleted")
    args = ap.parse_args(argv)
    vault = Vault(args.vault)
    if args.append_only and args.base and set(args.base) != {"0"}:
        broken = changed_existing(vault, args.base)
        if broken:
            for status, name in broken:
                print(f"{name} was {'changed' if status == 'M' else 'deleted'}: a vault is append-only. Publish a new paper that cites or challenges it instead.")
            return 1
    names, why = touched(vault, args.base)
    papers = [vault[n] for n in names]
    pins = [] if args.no_install else install(papers)
    lines = [f"## TinyPaper: `{vault.name}`", "",
             f"{len(papers)} of {len(vault.papers)} papers ({why}). Installed: {', '.join(pins) or 'nothing'}. "
             f"Each paper was rerun for up to {args.time} on this runner.", "",
             "| Paper | Verify | Claims on its log | First points match | Calls rerun | Claims rerun | Still needed |", "|---|---|---|---|---|---|---|"]
    ok, details = True, []
    for p in papers:
        report = p.verify()
        ok &= report.ok
        held = sum(s == "holds" for s in report.claims.values())
        r = p.reproduce(time=args.time, say=lambda *a: None) if report.valid else None
        need = "" if not r or not r.needed else f"about {r.needed / 60:.0f} min"
        lines.append(f"| `{p.stem}` | {'ok' if report.ok else '**not ok**'} | {held} of {len(report.claims)} hold | "
                     + (f"{r.matched} of {r.compared} calls | {r.rerun} of {r.calls} | {r.held} of {r.attempted} hold | {need} |" if r else "not run | | | |"))
        details += ["", f"<details><summary>{p.stem}</summary>", "", "```", report.text()] + ([r.text()] if r else []) + ["```", "</details>"]
        print(report.text(), flush=True)
        if r:
            print(r.text(), flush=True)
    summary = "\n".join(lines + details) + "\n"
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary)
    print("\n" + "\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
