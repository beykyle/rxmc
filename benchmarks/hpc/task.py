"""One cluster task: the run that SLURM array index ``i`` of a job stands for.

::

    python benchmarks/hpc/task.py KIND PROBLEM [INDEX]   # INDEX defaults to
                                                          # $SLURM_ARRAY_TASK_ID
    python benchmarks/hpc/task.py KIND PROBLEM --count    # how many tasks
    python benchmarks/hpc/task.py KIND PROBLEM --list     # what each one is

KIND is

* ``ref``   -- PROBLEM's reference: for a population reference (``omp`` and
               the GP problems) one part of it, a pocomc seed or one of the
               dynesty seeds that check them (pooled later by assemble.py);
               otherwise the whole reference, quadrature or a long NUTS run;
* ``run``   -- one sampler of the benchmark on PROBLEM;
* ``sweep`` -- one (setting, seed) of the pocomc resolution sweep;
* ``seed``  -- one (seed set, rung) of gp_discrepancy's ladder (PROBLEM is
               ignored; use ``gp``).

Every task writes ``benchmarks/results/hpc/<kind>/<problem>/<index>.json`` and
nothing else, so any number of them can run at once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))
OUT = BENCH / "results" / "hpc"


def tasks(kind, problem):
    """The runs a KIND job of PROBLEM is made of, in array-index order."""
    if kind == "ref":
        from reference import POPULATION

        if problem not in POPULATION:
            return [("whole", None)]
        spec = POPULATION[problem]
        dyn = spec.get("corroborate", {}).get("seeds", ())
        return [("pocomc", seed) for seed in spec["seeds"]] + [
            ("dynesty", seed) for seed in dyn
        ]
    if kind == "run":
        from bench_samplers import BUDGETS

        return list(BUDGETS[problem]["samplers"])
    if kind == "sweep":
        from sweep_pocomc import SEEDS, SETTINGS

        return [(s, seed) for s in SETTINGS for seed in SEEDS]
    if kind == "seed":
        from gp_seed_check import SEED_SETS

        return [(base, i) for base in SEED_SETS for i in range(3)]
    raise ValueError(f"unknown kind {kind!r}")


def run(kind, problem, index):
    task = tasks(kind, problem)[index]
    out = OUT / kind / problem / f"{index}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if kind == "ref":
        import time

        from problems import PROBLEMS, mode_share
        from reference import POPULATION, dynesty_part, pocomc_part

        spec = POPULATION.get(problem)
        p, _ = PROBLEMS[problem]()
        part, seed = task
        summary = mode_share(problem, p)
        if part == "whole":
            from reference import reference

            t0 = time.perf_counter()
            result = reference(problem, p)
            result["wall_s"] = time.perf_counter() - t0
            result["names"] = p.names
        elif part == "pocomc":
            settings = {k: spec[k] for k in ("n_effective", "n_active")}
            result = pocomc_part(p, seed, settings, summary)
        else:
            settings = {k: v for k, v in spec["corroborate"].items() if k != "seeds"}
            result = dynesty_part(p, seed, settings, summary)
        out.write_text(
            json.dumps(dict(problem=problem, part=part, names=p.names, result=result))
        )
    elif kind == "run":
        import bench_samplers

        bench_samplers.main(
            ["--problems", problem, "--only", task, "--out", str(out),
             "--references", str(OUT / "no-references.json")]
        )  # fmt: skip
    elif kind == "sweep":
        import sweep_pocomc

        sweep_pocomc.sweep(problem, [task[0]], (task[1],), out)
    elif kind == "seed":
        import gp_seed_check
        from problems import n40ca_ladder

        base, i = task
        rung, result = gp_seed_check.run_rung(n40ca_ladder(), base, i)
        out.write_text(json.dumps(dict(base=base, rung=rung, result=result)))
    print(f"{kind} {problem} task {index} ({task}) -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["ref", "run", "sweep", "seed"])
    ap.add_argument("problem")
    ap.add_argument("index", nargs="?", type=int)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    todo = tasks(args.kind, args.problem)
    if args.count:
        return print(len(todo))
    if args.list:
        return print("\n".join(f"{i}: {t}" for i, t in enumerate(todo)))
    index = args.index
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    run(args.kind, args.problem, index)


if __name__ == "__main__":
    main()
