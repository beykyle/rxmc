"""Submit any subset of the benchmark as SLURM array jobs, one task per run.

::

    python benchmarks/hpc/submit.py --all --dry-run               # everything
    python benchmarks/hpc/submit.py --problems omp alpha_gp --kinds ref run
    python benchmarks/hpc/submit.py --problems omp --kinds run --samplers NUTS pocomc
    python benchmarks/hpc/submit.py --problems alpha_gp --kinds sweep --settings 1024/512
    python benchmarks/hpc/submit.py --kinds seed --seed-sets 20
    python benchmarks/hpc/submit.py --problems alpha_gp --kinds ref --indices 12-52

One array job per (kind, problem), whose tasks are the selected entries of
``task.py KIND PROBLEM --list``; selecting samplers, settings, seeds or seed
sets keeps their array indices, so a result lands in the same file whatever
subset produced it.  Kinds: ``ref``, ``run``, ``sweep`` (any problem) and
``seed`` (gp_discrepancy's ladder; ``--problems`` does not apply).  Run it
from inside the repository with the environment active.

Site settings are options, not edits to ``job.sbatch``: ``--account``,
``--partition``, ``--constraint``, ``--venv`` (the venv ``env.sh``
activates; default ``~/rxmc-venv``) and ``--sbatch`` for any other ``sbatch``
option.  Keep them in a file outside the repository, one argument per line,
and pass it as ``@file``::

    # ~/rxmc-hpc/site.args
    --account=my-account
    --constraint=my-node-feature

    python benchmarks/hpc/submit.py @$HOME/rxmc-hpc/site.args --all --dry-run
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
    "ref": {"omp": "4:00:00", "alpha_gp": "2:00:00", "n40ca_gp": "1:00:00",
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
        from sweep_pocomc import label

        s, seed = entry
        wanted = {label(w) for w in args.settings or ()}
        return (not wanted or label(s) in wanted) and (
            not args.seeds or seed in args.seeds
        )
    if kind == "seed":
        return not args.seed_sets or entry[0] in args.seed_sets
    return True


def site_options(args):
    """The site's ``sbatch`` options: account, partition, constraint, the venv
    (exported to ``env.sh`` as ``RXMC_VENV``) and any others given."""
    opts = [
        f"--{name}={value}"
        for name in ("account", "partition", "constraint")
        if (value := getattr(args, name, None))
    ]
    if venv := getattr(args, "venv", None):
        opts.append(f"--export=ALL,RXMC_VENV={Path(venv).expanduser().resolve()}")
    return opts + list(getattr(args, "sbatch", None) or [])


def indices(spec):
    """Array indices from ``"3,5,12-52"``."""
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


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
                and (not getattr(args, "indices", None) or i in args.indices)
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
                    *site_options(args),
                    str(HERE / "job.sbatch"),
                    "python",
                    str(HERE / "task.py"),
                    kind,
                    problem,
                ]
            )
    return jobs


def parse_args(argv=None):
    """The command line, ``@file`` arguments read in place."""
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], fromfile_prefix_chars="@"
    )
    from bench_samplers import BUDGETS

    ap.add_argument("--problems", nargs="*", choices=list(BUDGETS))
    ap.add_argument(
        "--all", action="store_true", help="every benchmark problem (else --problems)"
    )
    ap.add_argument(
        "--kinds", nargs="*", default=["ref", "run", "sweep"],
        choices=["ref", "run", "sweep", "seed"],
    )  # fmt: skip
    ap.add_argument("--samplers", nargs="*", help="run: these samplers only")
    ap.add_argument(
        "--settings", nargs="*",
        help="sweep: these settings only, by label ('pocomc 1024/512' or 1024/512, "
        "'dynesty rslice 250')",
    )  # fmt: skip
    ap.add_argument("--seeds", type=int, nargs="*", help="sweep: these seeds only")
    ap.add_argument("--seed-sets", type=int, nargs="*", help="seed: these sets only")
    ap.add_argument(
        "--indices", type=indices,
        help="these array indices only, of every job selected (e.g. 12-52)",
    )  # fmt: skip
    ap.add_argument("--time", help="walltime per task, overriding the table")
    site = ap.add_argument_group("site settings (or put them in an @file)")
    site.add_argument("--account", help="sbatch --account")
    site.add_argument("--partition", help="sbatch --partition")
    site.add_argument("--constraint", help="sbatch --constraint (node features)")
    site.add_argument(
        "--venv", help="the venv env.sh activates (default $RXMC_VENV or ~/rxmc-venv)"
    )
    site.add_argument(
        "--sbatch", action="append", metavar="OPTION",
        help="any other sbatch option, as --sbatch=--qos=x (repeatable)",
    )  # fmt: skip
    ap.add_argument("--dry-run", action="store_true", help="print, don't submit")
    args = ap.parse_args(argv)
    if not (args.problems or args.all or args.kinds == ["seed"]):
        ap.error("name --problems, or --all for every problem")
    return args


def main(argv=None):
    args = parse_args(argv)
    (HERE.parent / "results" / "hpc" / "logs").mkdir(parents=True, exist_ok=True)
    for cmd in commands(args):
        print(shlex.join(cmd))
        if not args.dry_run:
            subprocess.run(
                cmd, check=True, cwd=HERE.parents[1]
            )  # job.sbatch uses relative paths


if __name__ == "__main__":
    main()
