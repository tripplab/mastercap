import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from mastercap.analysis import shell_volumes
from mastercap.cli import main
from mastercap.constraints import validate_geometry
from mastercap.generate import search
from mastercap.model import Atom, MastercapError, Structure
from mastercap.refinement import PushController, apply_radial_push, candidate_key


def metrics(c_error, s_error, total):
    return {"total_profile_error": total, "elements": {
        "C": {"profile_error": c_error, "count_difference": 0},
        "S": {"profile_error": s_error, "count_difference": 0}}}


def test_selection_objective_can_prefer_total_over_sulfur():
    lower_total = metrics(0.01, 1.0, 0.02)
    lower_sulfur = metrics(0.05, 0.2, 0.05)
    assert candidate_key(lower_total, "total") < candidate_key(lower_sulfur, "total")
    assert candidate_key(lower_sulfur, "worst-element") < candidate_key(lower_total, "worst-element")
    with pytest.raises(MastercapError):
        candidate_key(lower_total, "unknown")


def test_partial_push_prefers_largest_mismatch_and_caps_requested_moves():
    force = np.zeros((10, 3))
    direction = np.tile([1.0, 0, 0], (10, 1))
    details = apply_radial_push(force, np.ones(10), direction,
                                {"C": np.arange(8), "S": np.array([8, 9])},
                                {"C": np.arange(8)+2, "S": np.array([3, 5])},
                                strength=100, max_step=0.05, fraction=0.25)
    assert np.array_equal(np.flatnonzero(np.linalg.norm(force, axis=1)), [6, 7, 9])
    assert details["selected_atoms"] == details["atoms_pushed"] == 3
    assert details["max_requested_radial_move_angstrom"] <= 0.05
    assert np.max(np.linalg.norm(force, axis=1)) <= 0.05


def geometry(feasible=True, violations=0, components=1):
    return {"mandatory_pass": feasible, "distance_violations": violations, "global_components": components,
            "disconnected_residues": [] if feasible else [0], "outside_radial_bounds": 0,
            "isolated_atoms": 0, "center_error_angstrom": 0.0}


def test_immediate_feasibility_grows_strength_but_caps_it():
    history = []
    controller = PushController(0.005, 10, 10000, history)
    xyz = np.zeros((2, 3))
    for iteration in range(12):
        controller.start(iteration, xyz, {"strength": controller.strength})
        assert not controller.observe(iteration+1, geometry(), xyz)
    assert controller.strength == pytest.approx(0.02)
    assert all(push["recovery_iterations"] == 0 for push in history)
    assert controller.rollbacks == 0


def test_small_damage_repairs_without_strength_growth():
    controller = PushController(0.005, 10, 10000, [])
    xyz = np.zeros((2, 3))
    controller.start(3, xyz, {"strength": controller.strength})
    assert not controller.observe(4, geometry(False, 1, 2), xyz)
    assert not controller.observe(5, geometry(), xyz)
    assert controller.strength == 0.005
    assert controller.history[0]["recovery_iterations"] == 1
    assert controller.history[0]["next_state_classification"] == "small"


def test_recovery_limit_restores_only_coordinates():
    history = []
    controller = PushController(0.005, 2, 10000, history)
    xyz = np.arange(6.0).reshape(2, 3)
    original = xyz.copy()
    controller.start(7, xyz, {"strength": controller.strength})
    xyz += 0.1
    assert not controller.observe(8, geometry(False, 1, 2), xyz)
    assert not controller.observe(9, geometry(False, 1, 2), xyz)
    assert controller.observe(10, geometry(False, 1, 2), xyz)
    assert np.array_equal(xyz, original)
    assert controller.strength == 0.0025
    assert history[0]["rollback_reason"] == "recovery_limit"
    assert history[0]["rollback_after_repair_iterations"] == 2
    assert history[0]["recovery_iterations"] is None


def controlled_case(lower=0.1, all_inner=False):
    source = Structure([Atom("CA" if i%2 == 0 else "C", "C", "GLY", i//2) for i in range(4)],
                       np.array([[-2.4, 0, 0], [-0.8, 0, 0], [0.8, 0, 0], [2.4, 0, 0]]))
    edges = np.array([0.0, 2.0, 4.0])
    profiles = {"C": np.array([4.0, 0] if all_inner else [2.0, 2])/shell_volumes(edges)}
    analysis = SimpleNamespace(limits={("C", "C"): lower}, original=profiles)
    return source, edges, profiles, analysis


def controlled_search(source, edges, profiles, analysis, **kwargs):
    with patch("mastercap.generate.initialize", return_value=[]), patch("mastercap.generate.characteristic_forces"):
        return search(source, analysis, edges, profiles, np.zeros(3), np.random.default_rng(42),
                       time_limit=30, max_iterations=500, **kwargs)


def test_budget_counts_exactly_pushes_and_tiny_pushes_stay_feasible():
    source, edges, profiles, analysis = controlled_case()
    output, report = controlled_search(source, edges, profiles, analysis, refine_updates=5,
                                       refine_patience=0, refine_strength=1e-10)
    assert output is not None
    assert report["termination"] == "refinement_complete"
    assert report["refinement_stop_limit"] == "refine_updates"
    assert report["refinement_updates"] == report["profile_updates"] == len(report["push_history"]) == 5
    assert [p["iteration"] for p in report["push_history"]] == [0, 1, 2, 3, 4]
    assert report["rollback_count"] == 0
    assert all(p["recovery_iterations"] == 0 for p in report["push_history"])
    assert set(p["iteration"] for p in report["best_history"]) <= set(report["feasible_iterations"])
    assert report["profile_error_objective"] == "total"


def test_patience_stops_unproductive_pushes():
    source, edges, profiles, analysis = controlled_case()
    _, report = controlled_search(source, edges, profiles, analysis, refine_updates=50,
                                   refine_patience=2, refine_strength=1e-10)
    assert report["termination"] == "refinement_stalled"
    assert report["refinement_stop_limit"] == "refine_patience"
    assert report["profile_updates"] == 2


@pytest.mark.parametrize("mode", ["independent", "residue"])
def test_large_push_rolls_back_and_does_not_mutate_best_candidate(mode):
    source, edges, profiles, analysis = controlled_case(lower=1.59, all_inner=True)
    original = source.xyz.copy()
    messages = []
    progress = []
    output, report = controlled_search(source, edges, profiles, analysis, mode=mode,
                                       refine_strength=1e6, refine_fraction=0.25, refine_patience=0,
                                       refine_updates=50, stage_progress=messages.append,
                                       progress=lambda i, g, m: progress.append((i, g)), progress_interval=100)
    assert output is not None
    assert report["rollback_count"] > 0
    assert report["refinement_strength_final"] < 1e6
    assert report["termination"] == "refinement_stalled"
    assert report["refinement_stop_limit"] == "strength_floor"
    assert report["refinement_updates"] == report["rollback_count"]
    assert all(p["rollback"] and p["damage"]["distance_violations"] > 0 for p in report["push_history"])
    assert np.array_equal(source.xyz, original)
    assert np.array_equal(output.xyz, original)
    assert not np.shares_memory(output.xyz, source.xyz)
    assert validate_geometry(output, np.zeros(3), edges, analysis.limits, mode=mode)["mandatory_pass"]
    assert progress[-1][1]["final_search_state"]
    assert progress[-1][1]["export_candidate_available"]
    if mode == "residue":
        assert np.linalg.norm(output.xyz[1]-output.xyz[0]) == pytest.approx(1.6)
    json.dumps(report)  # Coordinate snapshots are private, never serialized.


@pytest.mark.parametrize("option,value", [
    ("--refine-updates", "-1"), ("--refine-patience", "-1"), ("--refine-recovery-limit", "-1"),
    ("--refine-strength", "-0.1"), ("--refine-strength", "nan"),
    ("--refine-max-step", "-0.1"), ("--refine-fraction", "0"), ("--refine-fraction", "1.1")])
def test_invalid_settings_are_rejected_before_input(option, value):
    with pytest.raises(SystemExit) as stopped:
        main(["generate", "--input", "missing.pdb", "--output-prefix", "out", option, value])
    assert stopped.value.code == 2


def test_export_with_new_options_passes_independent_validation(tmp_path, small_pdb, capsys):
    prefix = tmp_path/"adaptive"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--seed", "42", "--allow-approximate", "--refine-updates", "3", "--refine-patience", "0",
                 "--refine-strength", "1e-10", "--selection-objective", "total"]) == 2
    captured = capsys.readouterr().err
    assert "New best feasible candidate" in captured
    assert "median repair iterations=" in captured
    assert "rollbacks=" in captured
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["geometry"]["mandatory_pass"]
    assert report["search"]["refinement_updates"] == 3
    assert main(["validate", "--input", str(small_pdb), "--pdb", str(prefix.with_suffix(".pdb")),
                 "--report", str(prefix.with_suffix(".json")), "--output-prefix", str(tmp_path/"validation")]) == 2


def test_final_broken_state_is_labeled_and_not_exported(tmp_path, small_pdb, capsys):
    prefix = tmp_path/"rollback"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--seed", "42", "--allow-approximate", "--no-smoothing", "--refine-strength", "1000000",
                 "--refine-max-step", "1", "--refine-fraction", "1", "--refine-patience", "0",
                 "--progress-interval", "100"]) == 2
    output = capsys.readouterr().err
    assert "(final search state; not the exported candidate)" in output
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["geometry"]["mandatory_pass"]
    assert not report["search"]["last_search_geometry"]["mandatory_pass"]
    assert report["search"]["rollback_count"] > 0


@pytest.mark.parametrize("setting", [
    {"refine_updates": -1}, {"refine_patience": -1}, {"refine_recovery_limit": -1},
    {"refine_iterations": -1}, {"refine_strength": -1}, {"refine_strength": float("inf")},
    {"refine_strength": 1e308}, {"refine_max_step": -1}, {"refine_fraction": 0},
    {"refine_fraction": 1.1}, {"selection_objective": "unknown"}])
def test_library_rejects_invalid_refinement_settings(setting):
    source, edges, profiles, analysis = controlled_case()
    with pytest.raises(MastercapError):
        controlled_search(source, edges, profiles, analysis, **setting)


@pytest.mark.parametrize("setting", [{"refine_updates": 0}, {"refine_iterations": 0}])
def test_either_zero_budget_returns_first_feasible_geometry(setting):
    source, edges, profiles, analysis = controlled_case()
    _, report = controlled_search(source, edges, profiles, analysis, **setting)
    assert report["profile_updates"] == 0
    assert report["candidate_iteration"] == report["first_feasible_iteration"] == 0
    assert report["termination"] == "constraints_satisfied"
