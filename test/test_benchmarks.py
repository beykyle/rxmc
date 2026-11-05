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
    ref.write_text(json.dumps(dict(a=dict(mean=[1.5], sd=[1.0], method="m"))))
    s, r = merge_results.merge([one, two, ref], base, refs)
    assert s["problems"]["a"]["samplers"]["NUTS"]["mean"] == [1.0]
    assert s["problems"]["a"]["samplers"]["pocomc"]["mean"] == [2.0]
    assert s["problems"]["b"]["samplers"]["emcee"]["mean"] == [5.0]  # untouched
    assert r == dict(a=dict(mean=[1.5], sd=[1.0], method="m"), b=dict(mean=[5.0]))
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps(dict(problem="a", names=["x"], runs=[])))
    with pytest.raises(ValueError, match="not a reference"):
        merge_results.merge([sweep], base, refs)


def test_merge_refuses_two_inputs_with_the_same_entry(tmp_path):
    one = _sampler_file(tmp_path / "1.json", "a", "NUTS", 1.0)
    two = _sampler_file(tmp_path / "2.json", "a", "NUTS", 2.0)
    with pytest.raises(ValueError, match="a/NUTS"):
        merge_results.merge([one, two], tmp_path / "s.json", tmp_path / "r.json")


def test_mode_share_counts_the_notebooks_minor_mode():
    p, thetas = problems.PROBLEMS["alpha_gp"]()
    share = problems.mode_share("alpha_gp", p)
    theta = np.tile(thetas[0], (4, 1))
    theta[:2, p.names.index("V")] = [120.0, 130.0]  # the shallow family
    theta[2:, p.names.index("V")] = [160.0, 170.0]  # the deeper one
    out = share(theta, np.array([1.0, 1.0, 1.0, 3.0]))
    assert out["mode_share"] == pytest.approx(4 / 6)
    assert problems.mode_share("n40ca_gp", p) is None


@pytest.mark.slow
def test_the_sweep_runs_end_to_end(tmp_path):
    import sweep_pocomc

    out = tmp_path / "sweep.json"
    tiny = dict(n_total=128, n_evidence=128)
    saved = sweep_pocomc.sweep("linear", [(64, 32)], (0, 1), out, **tiny)
    assert [r["seed"] for r in saved["runs"]] == [0, 1]
    assert json.loads(out.read_text())["runs"] == saved["runs"]
    again = sweep_pocomc.sweep("linear", [(64, 32)], (0, 1), out, **tiny)
    assert again["runs"] == saved["runs"]  # resumed: nothing rerun
    sweep_pocomc.report(saved, {})


def test_the_seed_check_summarises_as_the_notebook_quotes():
    import gp_seed_check

    x = np.array([3.0, 1.0, 2.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    lo, mid, hi = gp_seed_check._quantiles(x, w)
    assert lo < mid < hi and 2.0 <= mid <= 3.0
    rungs = {"a": dict(logz=1.0), "b": dict(logz=4.0), "c": dict(logz=4.5)}
    out = gp_seed_check.with_differences(rungs)
    assert out["delta_logz"] == {"b - a": 3.0, "c - b": 0.5}


def test_every_array_job_maps_its_indices_to_runs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "hpc"))
    import task

    counts = {
        (k, p): len(task.tasks(k, p))
        for k in ("ref", "run", "sweep")
        for p in ("omp", "alpha_gp", "n40ca_gp")
    }
    assert set(counts.values()) == {6, 8, 9}
    assert len(task.tasks("seed", "gp")) == 9
    assert task.tasks("ref", "omp")[-1][0] == "dynesty"


def test_assemble_pools_reference_parts_and_joins_pieces(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "hpc"))
    import assemble

    monkeypatch.setattr(assemble, "OUT", tmp_path)
    ref_dir = tmp_path / "ref" / "n40ca_gp"
    ref_dir.mkdir(parents=True)
    run = dict(mean=[1.0, 2.0], sd=[1.0, 1.0], logz=-3.0, logz_err=0.1, wall_s=5.0)
    dyn = dict(run, mean=[1.02, 2.0], method="dynesty")
    for i, (part, result) in enumerate([("pocomc", run)] * 3 + [("dynesty", dyn)] * 3):
        piece = dict(problem="n40ca_gp", part=part, names=["a", "b"], result=result)
        (ref_dir / f"{i}.json").write_text(json.dumps(piece))
    refs = assemble.references()  # the other problems have no pieces: skipped
    assert list(refs) == ["n40ca_gp"]
    ref = refs["n40ca_gp"]
    assert ref["corroborated"] and ref["accepted"] and ref["logz"] == -3.0
    assert ref["names"] == ["a", "b"] and len(ref["corroboration"]["runs"]) == 3
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


def test_submit_selects_by_index_and_keeps_indices():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "hpc"))
    import submit

    def lines(*argv):
        args = submit.main.__globals__["argparse"].Namespace(
            problems=None, kinds=["ref", "run", "sweep"], samplers=None,
            settings=None, seeds=None, seed_sets=None, time=None,
        )  # fmt: skip
        for k, v in zip(argv[::2], argv[1::2]):
            setattr(args, k, v)
        return [" ".join(c[1:4]) for c in submit.commands(args)]

    only = lines("problems", ["omp"], "kinds", ["run"], "samplers", ["NUTS", "HMC"])
    assert only == ["--array=3,5 --job-name=run_omp --time=2:00:00"]
    sweep = lines("problems", ["linear"], "kinds", ["sweep"], "settings", ["2048/1024"])
    assert sweep == ["--array=3,4,5 --job-name=sweep_linear --time=1:00:00"]
    seed = lines("kinds", ["seed"], "seed_sets", [30])
    assert seed == ["--array=3,4,5 --job-name=seed_gp --time=1:00:00"]
    everything = lines()
    assert len(everything) == 3 * 8  # three kinds on every BUDGETS problem


def test_merge_drops_the_rows_of_a_redefined_problem(tmp_path):
    base = tmp_path / "samplers.json"
    old = dict(ndim=1, names=["old"], samplers=dict(emcee=dict(mean=[0.0])))
    base.write_text(json.dumps(dict(problems=dict(a=old))))
    new = tmp_path / "new.json"
    row = dict(mean=[1.0], sd=[1.0])
    new.write_text(
        json.dumps(
            dict(problems=dict(a=dict(ndim=1, names=["x"], samplers=dict(NUTS=row))))
        )
    )
    s, _ = merge_results.merge([new], base, tmp_path / "r.json")
    assert s["problems"]["a"]["names"] == ["x"]
    assert list(s["problems"]["a"]["samplers"]) == ["NUTS"]  # emcee's stale row gone
