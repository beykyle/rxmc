"""Sampler settings swept on one benchmark problem: pocomc's resolution, and on
the problems the prior-started samplers' defaults are chosen on, dynesty's.

Each setting runs at several seeds, and every run's moments, evidence, cost
and (on the problems with separated modes) the share of its draws in the
minor mode are saved; scoring against the reference waits for ``--report``,
so a sweep need not wait for its reference::

    python benchmarks/sweep_pocomc.py alpha_gp --out results/hpc/sweep_alpha_gp.json
    python benchmarks/sweep_pocomc.py --report results/hpc/sweep_alpha_gp.json
    python benchmarks/sweep_pocomc.py --grid results/hpc/sweep_{gp,modes,elastic}.json

A setting is named by its label, ``"pocomc 1024/512"`` (or just ``1024/512``)
or ``"dynesty rslice 250"``.  ``docs/benchmarks.md``'s sweep tables are these
reports: ``--report`` per problem, ``--grid`` the worst over the seeds of
several problems side by side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench_samplers import (
    BUDGETS,
    MARK,
    REFERENCES,
    errors,
    run_dynesty,
    run_pocomc,
    usable_reference,
    verdict,
)
from problems import PROBLEMS, mode_share

#: pocomc's (n_effective, n_active): the sweep on every problem but those below
SETTINGS = [(1024, 512), (2048, 1024), (4096, 2048)]
#: on the problems where the defaults of the samplers that start from the
#: prior are chosen (bench_samplers.DYNESTY, POCOMC): dynesty's and pocomc's
#: defaults and the settings about them
PRIOR_PROBLEMS = ("gp", "modes", "elastic")
PRIOR_SETTINGS = [
    ("dynesty", dict(sample="rwalk", nlive=200)),
    ("dynesty", dict(sample="rwalk", nlive=500)),
    ("dynesty", dict(sample="rslice", nlive=250)),
    ("dynesty", dict(sample="rslice", nlive=500)),
    ("pocomc", dict(n_effective=512, n_active=256)),
    ("pocomc", dict(n_effective=1024, n_active=512)),
    ("pocomc", dict(n_effective=2048, n_active=1024)),
]
SEEDS = (100, 101, 102)


def setting(s):
    """``(sampler, settings)`` of a setting given as that pair, a pocomc
    ``(n_effective, n_active)`` pair, or a label."""
    if isinstance(s, str):
        words = s.split()
        if len(words) == 1:
            words = ["pocomc", *words]
        if words[0] == "pocomc" and len(words) == 2:
            ne, na = (int(v) for v in words[1].split("/"))
            return "pocomc", dict(n_effective=ne, n_active=na)
        if words[0] == "dynesty" and len(words) == 3:
            return "dynesty", dict(sample=words[1], nlive=int(words[2]))
        raise ValueError(f"not a setting: {s!r}")
    if isinstance(s[0], str):
        return s[0], dict(s[1])
    return "pocomc", dict(n_effective=int(s[0]), n_active=int(s[1]))


def label(s):
    """A setting's label, ``"pocomc 1024/512"`` or ``"dynesty rslice 250"``."""
    sampler, kw = setting(s)
    if sampler == "pocomc":
        return f"pocomc {kw['n_effective']}/{kw['n_active']}"
    return f"dynesty {kw['sample']} {kw['nlive']}"


def run_label(r):
    """The label of a saved run (one saved before dynesty was swept is pocomc)."""
    return r.get("setting") or label((r["n_effective"], r["n_active"]))


def settings_for(name):
    """The settings swept on problem ``name``."""
    if name in PRIOR_PROBLEMS:
        return list(PRIOR_SETTINGS)
    return [setting(s) for s in SETTINGS]


def sweep(name, settings=None, seeds=SEEDS, out=None, **lengths):
    """``settings`` (default ``settings_for(name)``) as :func:`setting` reads
    them; ``lengths`` (``n_total``, ``n_evidence``) shorten pocomc runs, for
    tests."""
    problem, _ = PROBLEMS[name]()
    summary = mode_share(name, problem)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    saved = json.loads(out.read_text()) if out and out.exists() else {}
    runs = saved.get("runs", [])
    done = {(run_label(r), r["seed"]) for r in runs}
    for s in settings_for(name) if settings is None else settings:
        sampler, kw = setting(s)
        name_s = label(s)
        for seed in seeds:
            if (name_s, seed) in done:
                continue  # resume an interrupted sweep
            if sampler == "pocomc":
                r = run_pocomc(
                    problem, BUDGETS[name], seed, summary=summary, **kw, **lengths
                )
            else:
                r = run_dynesty(problem, BUDGETS[name], seed, summary=summary, **kw)
            r.update(sampler=sampler, setting=name_s, seed=seed, **kw)
            runs.append(r)
            share = f", mode share {r['mode_share']:.2f}" if "mode_share" in r else ""
            print(
                f"{name} {name_s} seed {seed}: {r['wall_s']:.0f} s, "
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


def _by_setting(runs):
    """Runs grouped by label, in the order the labels first appear."""
    groups = {}
    for r in runs:
        groups.setdefault(run_label(r), []).append(r)
    return groups


def report(saved, refs):
    name = saved["problem"]
    ref = usable_reference(name, saved.get("names"), refs.get(name))
    print(f"| {name} | bias / width per seed | mode share | log Z | wall time |")
    print("|---|---|---|---|---|")
    for name_s, rs in _by_setting(saved["runs"]).items():
        if ref is None:
            scores = "no reference"
        else:
            pairs = [errors(r, ref) for r in rs]
            scores = ", ".join(
                f"{b:.2f} / {w:.2f}" + MARK[verdict(max(b, w), ref)] for b, w in pairs
            )
        share = ", ".join(f"{r['mode_share']:.2f}" for r in rs if "mode_share" in r)
        logz = ", ".join(f"{r['logz']:.2f}" for r in rs)
        wall = (
            f"{min(r['wall_s'] for r in rs):.0f}–{max(r['wall_s'] for r in rs):.0f} s"
        )
        print(f"| {name_s} | {scores} | {share or '—'} | {logz} | {wall} |")


def grid(saved, refs):
    """Several problems' sweeps side by side: per setting and problem, the
    worst bias and the worst width error over the seeds."""
    cells, order = {}, []
    names = [s["problem"] for s in saved]
    for s in saved:
        ref = usable_reference(s["problem"], s.get("names"), refs.get(s["problem"]))
        for name_s, rs in _by_setting(s["runs"]).items():
            if name_s not in order:
                order.append(name_s)
            if ref is None:
                cells[name_s, s["problem"]] = "no reference"
                continue
            b, w = (max(e) for e in zip(*(errors(r, ref) for r in rs)))
            cells[name_s, s["problem"]] = (
                f"{b:.2f} / {w:.2f}" + MARK[verdict(max(b, w), ref)]
            )
    print("| setting | " + " | ".join(f"`{n}`" for n in names) + " |")
    print("|---|" + "---|" * len(names))
    for name_s in order:
        row = [cells.get((name_s, n), "") for n in names]
        print(f"| {name_s} | " + " | ".join(row) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("problem", nargs="?", choices=list(PROBLEMS))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument(
        "--settings",
        nargs="*",
        help="labels, e.g. 'pocomc 1024/512' (or 1024/512), 'dynesty rslice 250' "
        "(default: the problem's own, settings_for)",
    )
    ap.add_argument("--report", type=Path, help="print a saved sweep and exit")
    ap.add_argument(
        "--grid", type=Path, nargs="+", help="print saved sweeps side by side and exit"
    )
    ap.add_argument("--references", type=Path, default=REFERENCES)
    args = ap.parse_args(argv)
    if args.report or args.grid:
        refs = (
            json.loads(args.references.read_text()) if args.references.exists() else {}
        )
        if args.grid:
            return grid([json.loads(p.read_text()) for p in args.grid], refs)
        return report(json.loads(args.report.read_text()), refs)
    if args.problem is None:
        ap.error("give a problem to sweep, or --report or --grid")
    settings = [setting(s) for s in args.settings] if args.settings else None
    return sweep(args.problem, settings, tuple(args.seeds), args.out)


if __name__ == "__main__":
    main()
