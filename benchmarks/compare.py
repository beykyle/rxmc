"""Side-by-side table of two ``bench_log_posterior.py`` result files.

    python benchmarks/compare.py BASELINE.json CANDIDATE.json

Ratios are candidate / baseline times (below 1 is faster), and the value
column is the largest relative difference of ``log_posterior`` at the shared
parameter points.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def _ms(t):
    return 1e3 * t["median"]


def main(base_path, cand_path):
    base, cand = (json.load(open(p)) for p in (base_path, cand_path))
    print(f"baseline:  {base['backend']} rxmc {base['rxmc']} jitr {base.get('jitr')}")
    print(f"candidate: {cand['backend']} rxmc {cand['rxmc']} jitr {cand.get('jitr')}")
    head = (
        f"{'problem':<9}{'max rel dV':>11}{'first(s)':>16}{'call ms':>20}"
        f"{'ratio':>7}{'grad ms':>20}{'ratio':>7}{'B=256 θ/s':>22}{'emcee s':>16}"
    )
    print(head)
    print("-" * len(head))
    for name, b in base["problems"].items():
        c = cand["problems"].get(name)
        if c is None:
            continue
        vb, vc = np.array(b["values"]), np.array(c["values"])
        rel = float(np.max(np.abs(vc - vb) / np.maximum(1.0, np.abs(vb))))
        first = f"{b['first_call_s']:.2f}/{c['first_call_s']:.2f}"
        call = f"{_ms(b['log_posterior']):.3f}/{_ms(c['log_posterior']):.3f}"
        r_call = c["log_posterior"]["median"] / b["log_posterior"]["median"]
        grad = f"{_ms(b['grad']):.3f}/{_ms(c['grad']):.3f}"
        r_grad = c["grad"]["median"] / b["grad"]["median"]
        tb, tc = b["batch"]["256"]["theta_per_s"], c["batch"]["256"]["theta_per_s"]
        batch = f"{tb:.0f}/{tc:.0f}"
        em = ""
        if "emcee" in b and "emcee" in c:
            em = f"{b['emcee']['wall_s']:.1f}/{c['emcee']['wall_s']:.1f}"
        print(
            f"{name:<9}{rel:>11.1e}{first:>16}{call:>20}{r_call:>7.2f}"
            f"{grad:>20}{r_grad:>7.2f}{batch:>22}{em:>16}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:3])
