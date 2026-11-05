"""pocomc's resolution swept on one benchmark problem.

Each ``(n_effective, n_active)`` setting runs at several seeds, and every run's
moments, evidence, cost and (on the problems with separated modes) the share
of its draws in the minor mode are saved; scoring against the reference waits
for ``--report``, so a sweep need not wait for its reference::

    python benchmarks/sweep_pocomc.py alpha_gp --out results/hpc/sweep_alpha_gp.json
    python benchmarks/sweep_pocomc.py --report results/hpc/sweep_alpha_gp.json

``docs/benchmarks.md``'s sweep tables are this report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench_samplers import (
    BUDGETS,
    REFERENCES,
    TOLERANCE,
    errors,
    run_pocomc,
    usable_reference,
)
from problems import PROBLEMS, mode_share

SETTINGS = [(1024, 512), (2048, 1024), (4096, 2048)]
SEEDS = (100, 101, 102)


def sweep(name, settings=SETTINGS, seeds=SEEDS, out=None, **lengths):
    """``lengths`` (``n_total``, ``n_evidence``) shorten the runs, for tests."""
    problem, _ = PROBLEMS[name]()
    summary = mode_share(name, problem)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    saved = json.loads(out.read_text()) if out and out.exists() else {}
    runs = saved.get("runs", [])
    done = {(r["n_effective"], r["n_active"], r["seed"]) for r in runs}
    for n_effective, n_active in settings:
        for seed in seeds:
            if (n_effective, n_active, seed) in done:
                continue  # resume an interrupted sweep
            r = run_pocomc(
                problem,
                BUDGETS[name],
                seed,
                summary=summary,
                n_effective=n_effective,
                n_active=n_active,
                **lengths,
            )
            r.update(n_effective=n_effective, n_active=n_active, seed=seed)
            runs.append(r)
            share = f", mode share {r['mode_share']:.2f}" if "mode_share" in r else ""
            print(
                f"{name} {n_effective}/{n_active} seed {seed}: {r['wall_s']:.0f} s, "
                f"log Z {r['logz']:.2f}{share}",
                flush=True,
            )
            if out:  # each run is saved as it ends
                out.write_text(
                    json.dumps(
                        dict(problem=name, names=problem.names, runs=runs), indent=1
                    )
                )
    return dict(problem=name, names=problem.names, runs=runs)


def report(saved, refs):
    name = saved["problem"]
    ref = usable_reference(name, saved.get("names"), refs.get(name))
    print(f"| {name} | bias / width per seed | mode share | log Z | wall time |")
    print("|---|---|---|---|---|")
    by_setting = {}
    for r in saved["runs"]:
        by_setting.setdefault((r["n_effective"], r["n_active"]), []).append(r)
    for (ne, na), rs in sorted(by_setting.items()):
        if ref is None:
            scores = "no reference"
        else:
            pairs = [errors(r, ref) for r in rs]
            scores = ", ".join(
                f"{b:.2f} / {w:.2f}" + ("" if max(b, w) <= TOLERANCE else " ✗")
                for b, w in pairs
            )
        share = ", ".join(f"{r['mode_share']:.2f}" for r in rs if "mode_share" in r)
        logz = ", ".join(f"{r['logz']:.2f}" for r in rs)
        wall = (
            f"{min(r['wall_s'] for r in rs):.0f}–{max(r['wall_s'] for r in rs):.0f} s"
        )
        print(f"| pocomc {ne}/{na} | {scores} | {share or '—'} | {logz} | {wall} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("problem", nargs="?", choices=list(PROBLEMS))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument(
        "--settings",
        nargs="*",
        default=[f"{a}/{b}" for a, b in SETTINGS],
        help="n_effective/n_active pairs",
    )
    ap.add_argument("--report", type=Path, help="print a saved sweep and exit")
    ap.add_argument("--references", type=Path, default=REFERENCES)
    args = ap.parse_args(argv)
    if args.report:
        refs = (
            json.loads(args.references.read_text()) if args.references.exists() else {}
        )
        return report(json.loads(args.report.read_text()), refs)
    if args.problem is None:
        ap.error("give a problem to sweep, or --report")
    settings = [tuple(int(v) for v in s.split("/")) for s in args.settings]
    return sweep(args.problem, settings, tuple(args.seeds), args.out)


if __name__ == "__main__":
    main()
