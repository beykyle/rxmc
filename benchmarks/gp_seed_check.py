"""gp_discrepancy's cross-section ladder under other seeds.

The notebook fits its three rungs with pocomc at 4096/2048 and seeds 10, 11
and 12.  This reruns the ladder at the notebook's settings with three other
sets of seeds and records what the notebook quotes: each rung's
log Z, the differences between rungs, and the medians and 68 % intervals of
the four volume parameters::

    python benchmarks/gp_seed_check.py --out results/hpc/gp_seed_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from bench_samplers import run_pocomc
from problems import RUNGS, n40ca_ladder

#: the notebook's resolution for the ladder
SETTINGS = dict(n_effective=4096, n_active=2048)
#: the notebook uses 10 + rung; these are three other sets
SEED_SETS = (20, 30, 40)
VOLUME = ("Vv", "Wv", "Rv", "av")


def _quantiles(x, w, qs=(0.16, 0.5, 0.84)):
    """Weighted quantiles, each draw's weight centred on its value (equal
    weights give the usual midpoint median)."""
    order = np.argsort(x)
    w = w[order] / w.sum()
    cdf = np.cumsum(w) - 0.5 * w
    return [float(np.interp(q, cdf, x[order])) for q in qs]


def volume_summary(problem):
    cols = [problem.names.index(n) for n in VOLUME]

    def summary(theta, weights):
        return {
            "volume": {
                n: _quantiles(theta[:, c], weights) for n, c in zip(VOLUME, cols)
            }
        }

    return summary


def run_rung(base, i, settings=SETTINGS, ladder=None):
    """Rung ``i`` of the ladder at seed ``base + i``, as the notebook seeds them;
    only that rung is built unless a whole ``ladder`` is given."""
    rung = RUNGS[i]
    problem = (ladder or n40ca_ladder([rung]))[rung]
    r = run_pocomc(problem, {}, base + i, summary=volume_summary(problem), **settings)
    # as the notebook prints it: a density of the cross sections, not their logs
    logz = r["logz"] + problem.log_jacobian()
    print(f"seeds {base}+: {rung}: log Z {logz:.2f}", flush=True)
    return rung, dict(
        logz=logz, logz_err=r["logz_err"], wall_s=r["wall_s"], volume=r["volume"]
    )


def with_differences(rungs):
    """The rungs of one seed set, with the log Z differences between them."""
    names = list(rungs)
    return {
        **rungs,
        "delta_logz": {
            f"{b} - {a}": rungs[b]["logz"] - rungs[a]["logz"]
            for a, b in zip(names, names[1:])
        },
    }


def check(seed_sets=SEED_SETS, out=None, settings=SETTINGS):
    ladder = n40ca_ladder()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out and out.exists() else {}
    for base in seed_sets:
        if str(base) in results:
            continue  # resume
        rungs = dict(run_rung(base, i, settings, ladder) for i in range(len(RUNGS)))
        results[str(base)] = with_differences(rungs)
        if out:
            out.write_text(json.dumps(results, indent=1))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed-sets", type=int, nargs="*", default=list(SEED_SETS))
    args = ap.parse_args(argv)
    check(tuple(args.seed_sets), args.out)


if __name__ == "__main__":
    main()
