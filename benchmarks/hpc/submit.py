"""Submit any subset of the benchmark as SLURM array jobs, one task per run.

::

    python benchmarks/hpc/submit.py --dry-run                     # everything
    python benchmarks/hpc/submit.py --problems omp alpha_gp --kinds ref run
    python benchmarks/hpc/submit.py --problems omp --kinds run --samplers NUTS pocomc
    python benchmarks/hpc/submit.py --problems alpha_gp --kinds sweep --settings 1024/512
    python benchmarks/hpc/submit.py --kinds seed --seed-sets 20

One array job per (kind, problem), whose tasks are the selected entries of
``task.py KIND PROBLEM --list``; selecting samplers, settings, seeds or seed
sets keeps their array indices, so a result lands in the same file whatever
subset produced it.  Kinds: ``ref``, ``run``, ``sweep`` (any problem) and
``seed`` (gp_discrepancy's ladder; ``--problems`` does not apply).  Run it
from inside the repository with the environment active.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from task import tasks  # noqa: E402

#: walltime per task, about twice the 8-core-laptop estimates; problems not
#: listed take the kind's default
TIME = {
    "ref": {"omp": "4:00:00", "alpha_gp": "6:00:00", "n40ca_gp": "2:00:00",
            "hierarchy": "3:00:00", "elastic": "2:00:00", "default": "1:00:00"},
    "run": {"omp": "2:00:00", "alpha_gp": "3:00:00", "n40ca_gp": "1:00:00",
            "hierarchy": "3:00:00", "default": "1:00:00"},
    "sweep": {"omp": "3:00:00", "alpha_gp": "4:00:00", "n40ca_gp": "1:00:00",
              "hierarchy": "3:00:00", "default": "1:00:00"},
    "seed": {"default": "1:00:00"},
}  # fmt: skip


def _selected(kind, entry, args):
    if kind == "run":
        return not args.samplers or entry in args.samplers
    if kind == "sweep":
        (ne, na), seed = entry
        return (not args.settings or f"{ne}/{na}" in args.settings) and (
            not args.seeds or seed in args.seeds
        )
    if kind == "seed":
        return not args.seed_sets or entry[0] in args.seed_sets
    return True


def commands(args):
    """The sbatch command lines for the selection, one per array job."""
    from bench_samplers import BUDGETS

    jobs = []
    for kind in args.kinds:
        for problem in ["gp"] if kind == "seed" else (args.problems or list(BUDGETS)):
            indices = [
                i
                for i, e in enumerate(tasks(kind, problem))
                if _selected(kind, e, args)
            ]
            if not indices:
                continue
            time = args.time or TIME[kind].get(problem, TIME[kind]["default"])
            jobs.append(
                [
                    "sbatch",
                    f"--array={','.join(map(str, indices))}",
                    f"--job-name={kind}_{problem}",
                    f"--time={time}",
                    str(HERE / "job.sbatch"),
                    "python",
                    str(HERE / "task.py"),
                    kind,
                    problem,
                ]
            )
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--problems", nargs="*", help="default: every benchmark problem")
    ap.add_argument(
        "--kinds", nargs="*", default=["ref", "run", "sweep"],
        choices=["ref", "run", "sweep", "seed"],
    )  # fmt: skip
    ap.add_argument("--samplers", nargs="*", help="run: these samplers only")
    ap.add_argument("--settings", nargs="*", help="sweep: n_effective/n_active pairs")
    ap.add_argument("--seeds", type=int, nargs="*", help="sweep: these seeds only")
    ap.add_argument("--seed-sets", type=int, nargs="*", help="seed: these sets only")
    ap.add_argument("--time", help="walltime per task, overriding the table")
    ap.add_argument("--dry-run", action="store_true", help="print, don't submit")
    args = ap.parse_args(argv)
    (HERE.parent / "results" / "hpc" / "logs").mkdir(parents=True, exist_ok=True)
    for cmd in commands(args):
        print(shlex.join(cmd))
        if not args.dry_run:
            subprocess.run(
                cmd, check=True, cwd=HERE.parents[1]
            )  # job.sbatch uses relative paths


if __name__ == "__main__":
    main()
