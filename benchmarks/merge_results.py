"""Merge per-job benchmark results into the committed result files.

On a cluster each job writes a file of its own (``bench_samplers.py --out``
for samplers, ``--reference --references`` for references), so jobs never
race on one file.  This merges them::

    python benchmarks/hpc/assemble.py
    python benchmarks/merge_results.py benchmarks/results/hpc/references.json \
        benchmarks/results/hpc/run/*/*.json

A sampler file holds ``{"meta": ..., "problems": {name: {"ndim", "names",
"samplers": {...}}}}`` and is merged sampler by sampler into
``results/samplers.json``; a references file holds ``{name: reference}`` and
is merged problem by problem into ``results/references.json``.  Entries from
the inputs replace the committed ones; two inputs that both carry the same
sampler on the same problem (or the same reference) are refused, since one
would silently win.  Problems and samplers no input mentions are left as
they are.  A problem whose parameter names change (a redefined problem)
loses the sampler rows no input replaces, since they belong to the old
problem; anything that is neither a sampler file nor a file of references
(a sweep, say) is refused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def merge(inputs, samplers_path, references_path):
    samplers = json.loads(samplers_path.read_text()) if samplers_path.exists() else {}
    samplers.setdefault("problems", {})
    references = (
        json.loads(references_path.read_text()) if references_path.exists() else {}
    )
    seen = {}
    for path in inputs:
        data = json.loads(Path(path).read_text())
        if "problems" in data:  # a sampler run
            for name, entry in data["problems"].items():
                into = samplers["problems"].setdefault(
                    name,
                    dict(ndim=entry["ndim"], names=entry.get("names"), samplers={}),
                )
                new_names = entry.get("names")
                if new_names is not None and into.get("names") not in (None, new_names):
                    # a redefined problem: the old rows belong to the old problem
                    print(
                        f"{name}: parameters changed; its old sampler rows are dropped"
                    )
                    into["samplers"] = {
                        s: r
                        for s, r in into["samplers"].items()
                        if ("sampler", name, s) in seen
                    }
                into["ndim"] = entry["ndim"]
                if new_names is not None:
                    into["names"] = new_names
                for sampler, row in entry["samplers"].items():
                    key = ("sampler", name, sampler)
                    if key in seen:
                        raise ValueError(
                            f"{name}/{sampler} is in {seen[key]} and {path}"
                        )
                    seen[key] = path
                    into["samplers"][sampler] = row
            if "meta" in data:
                samplers["meta"] = data["meta"]
        else:  # references
            for name, ref in data.items():
                if not (isinstance(ref, dict) and {"mean", "sd", "method"} <= set(ref)):
                    raise ValueError(
                        f"{path}: {name!r} is not a reference (a sweep or seed-check "
                        "file?); give merge_results.py references and sampler runs only"
                    )
                key = ("reference", name)
                if key in seen:
                    raise ValueError(
                        f"the {name} reference is in {seen[key]} and {path}"
                    )
                seen[key] = path
                references[name] = ref
    return samplers, references


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--samplers", type=Path, default=RESULTS / "samplers.json")
    ap.add_argument("--references", type=Path, default=RESULTS / "references.json")
    args = ap.parse_args(argv)
    samplers, references = merge(args.inputs, args.samplers, args.references)
    args.samplers.write_text(json.dumps(samplers, indent=1))
    args.references.write_text(json.dumps(references, indent=1))
    print(f"merged {len(args.inputs)} files into {args.samplers} and {args.references}")


if __name__ == "__main__":
    main()
