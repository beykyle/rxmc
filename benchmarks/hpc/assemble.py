"""Assemble the cluster tasks' pieces into the files the benchmark reads.

::

    python benchmarks/hpc/assemble.py

* each problem's reference, whole or pooled from its parts
  (``reference.pool_parts``), goes into ``results/hpc/references.json``;
* the sweep pieces of each problem become ``results/hpc/sweep_<problem>.json``;
* the seed-check pieces become ``results/hpc/gp_seed_check.json``, with each
  seed set's log Z differences.

Then ``merge_results.py`` folds ``results/hpc/references.json`` and the
sampler pieces (``results/hpc/run/*/*.json``) into the committed results.
A missing piece is reported and its problem skipped, so a partial set of
results can still be looked at.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
OUT = HERE.parent / "results" / "hpc"


def _pieces(kind, problem):
    from task import tasks

    want = len(tasks(kind, problem))
    paths = [OUT / kind / problem / f"{i}.json" for i in range(want)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"{kind} {problem}: missing {missing}; skipped")
        return None
    return [json.loads(p.read_text()) for p in paths]


def references():
    from reference import POPULATION, pool_parts

    refs = {}
    for problem_dir in sorted((OUT / "ref").glob("*")):
        problem = problem_dir.name
        parts = _pieces("ref", problem)
        if parts is None:
            continue
        if parts[0]["part"] == "whole":  # quadrature or a long NUTS run
            refs[problem] = parts[0]["result"]
            continue
        spec = POPULATION[problem]
        runs = [p["result"] for p in parts if p["part"] == "pocomc"]
        dyn = [p["result"] for p in parts if p["part"] == "dynesty"]
        settings = {k: spec[k] for k in ("n_effective", "n_active")}
        print(f"{problem}:")
        refs[problem] = pool_parts(
            runs, spec["agree"], settings, dyn[0] if dyn else None
        )
        refs[problem]["wall_s"] = sum(r["wall_s"] for r in runs + dyn)
    (OUT / "references.json").write_text(json.dumps(refs, indent=1))
    return refs


def sweeps():
    for problem_dir in sorted((OUT / "sweep").glob("*")):
        pieces = _pieces("sweep", problem_dir.name)
        if pieces is None:
            continue
        runs = [r for p in pieces for r in p["runs"]]
        merged = dict(problem=pieces[0]["problem"], names=pieces[0]["names"], runs=runs)
        (OUT / f"sweep_{problem_dir.name}.json").write_text(
            json.dumps(merged, indent=1)
        )


def seed_check():
    from gp_seed_check import with_differences

    pieces = _pieces("seed", "gp")
    if pieces is None:
        return
    by_base = {}
    for p in pieces:
        by_base.setdefault(str(p["base"]), {})[p["rung"]] = p["result"]
    out = {base: with_differences(rungs) for base, rungs in by_base.items()}
    (OUT / "gp_seed_check.json").write_text(json.dumps(out, indent=1))


def main():
    references()
    sweeps()
    seed_check()


if __name__ == "__main__":
    main()
