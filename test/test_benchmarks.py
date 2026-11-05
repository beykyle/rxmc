"""The benchmark's problems, its result merging and its cluster drivers."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import merge_results  # noqa: E402
import problems  # noqa: E402


@pytest.mark.parametrize("name", ["omp", "alpha_gp", "n40ca_gp"])
def test_the_notebook_problems_build_and_are_finite(name):
    p, thetas = problems.PROBLEMS[name]()
    assert p.ndim == {"omp": 9, "alpha_gp": 9, "n40ca_gp": 7}[name]
    assert all(np.isfinite(p.log_posterior(t)) for t in thetas)


def test_the_ladder_has_three_rungs_sharing_the_data():
    ladder = problems.n40ca_ladder()
    assert list(ladder) == ["bare", "proportional error", "GP, growing amplitude"]
    ys = [p.constraints[0].y for p in ladder.values()]
    assert all(np.array_equal(ys[0], y) for y in ys)


def _sampler_file(path, name, sampler, mean):
    row = dict(mean=[mean], sd=[1.0])
    data = dict(
        meta={}, problems={name: dict(ndim=1, names=["x"], samplers={sampler: row})}
    )
    path.write_text(json.dumps(data))
    return path


def test_merge_replaces_what_the_inputs_carry_and_keeps_the_rest(tmp_path):
    base = tmp_path / "samplers.json"
    base.write_text(
        json.dumps(
            dict(
                problems=dict(
                    a=dict(ndim=1, names=["x"], samplers=dict(NUTS=dict(mean=[0.0]))),
                    b=dict(ndim=1, names=["x"], samplers=dict(emcee=dict(mean=[5.0]))),
                )
            )
        )
    )
    refs = tmp_path / "references.json"
    refs.write_text(json.dumps(dict(b=dict(mean=[5.0]))))
    one = _sampler_file(tmp_path / "1.json", "a", "NUTS", 1.0)
    two = _sampler_file(tmp_path / "2.json", "a", "pocomc", 2.0)
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(dict(a=dict(mean=[1.5]))))
    s, r = merge_results.merge([one, two, ref], base, refs)
    assert s["problems"]["a"]["samplers"]["NUTS"]["mean"] == [1.0]
    assert s["problems"]["a"]["samplers"]["pocomc"]["mean"] == [2.0]
    assert s["problems"]["b"]["samplers"]["emcee"]["mean"] == [5.0]  # untouched
    assert r == dict(a=dict(mean=[1.5]), b=dict(mean=[5.0]))


def test_merge_refuses_two_inputs_with_the_same_entry(tmp_path):
    one = _sampler_file(tmp_path / "1.json", "a", "NUTS", 1.0)
    two = _sampler_file(tmp_path / "2.json", "a", "NUTS", 2.0)
    with pytest.raises(ValueError, match="a/NUTS"):
        merge_results.merge([one, two], tmp_path / "s.json", tmp_path / "r.json")


@pytest.mark.slow
def test_the_sweep_and_the_seed_check_run_end_to_end(tmp_path):
    import gp_seed_check
    import sweep_pocomc

    out = tmp_path / "sweep.json"
    saved = sweep_pocomc.sweep("alpha_gp", [(64, 32)], (0,), out)
    assert saved["runs"][0]["mode_share_label"] == "share in the deeper family"
    assert json.loads(out.read_text())["runs"] == saved["runs"]
    sweep_pocomc.report(saved, {})
    res = gp_seed_check.check(
        (20,), tmp_path / "seed.json", dict(n_effective=64, n_active=32)
    )
    assert set(res["20"]["GP, growing amplitude"]["volume"]) == {"Vv", "Wv", "Rv", "av"}
    assert len(res["20"]["delta_logz"]) == 2


def test_every_array_job_maps_its_indices_to_runs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "hpc"))
    import task

    counts = {
        (k, p): len(task.tasks(k, p))
        for k in ("ref", "run", "sweep")
        for p in ("omp", "alpha_gp", "n40ca_gp")
    }
    assert set(counts.values()) == {4, 8, 9}
    assert len(task.tasks("seed", "gp")) == 9
    assert task.tasks("ref", "omp")[-1][0] == "dynesty"


def test_assemble_pools_reference_parts_and_joins_pieces(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "hpc"))
    import assemble

    monkeypatch.setattr(assemble, "OUT", tmp_path)
    ref_dir = tmp_path / "ref" / "n40ca_gp"
    ref_dir.mkdir(parents=True)
    run = dict(mean=[1.0, 2.0], sd=[1.0, 1.0], logz=-3.0, logz_err=0.1, wall_s=5.0)
    for i in range(3):
        part = dict(problem="n40ca_gp", part="pocomc", result=run)
        (ref_dir / f"{i}.json").write_text(json.dumps(part))
    dyn = dict(run, mean=[1.02, 2.0], method="dynesty")
    (ref_dir / "3.json").write_text(
        json.dumps(dict(problem="n40ca_gp", part="dynesty", result=dyn))
    )
    refs = assemble.references()  # the other problems have no pieces: skipped
    assert list(refs) == ["n40ca_gp"]
    assert refs["n40ca_gp"]["corroborated"] and refs["n40ca_gp"]["logz"] == -3.0
    seed_dir = tmp_path / "seed" / "gp"
    seed_dir.mkdir(parents=True)
    rungs = ["bare", "proportional error", "GP, growing amplitude"]
    for i, (base, r) in enumerate((b, r) for b in (20, 30, 40) for r in range(3)):
        piece = dict(base=base, rung=rungs[r], result=dict(logz=float(r)))
        (seed_dir / f"{i}.json").write_text(json.dumps(piece))
    assemble.seed_check()
    out = json.loads((tmp_path / "gp_seed_check.json").read_text())
    assert out["30"]["delta_logz"] == {
        "proportional error - bare": 1.0,
        "GP, growing amplitude - proportional error": 1.0,
    }
