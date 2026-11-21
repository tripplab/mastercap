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
from mastercap.refinement import (AcceptanceController, PushController, apply_radial_push,
                                  apply_crossing_push, candidate_key, bin_inventory)


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
    details, selected = apply_radial_push(force, np.ones(10), direction,
                                {"C": np.arange(8), "S": np.array([8, 9])},
                                {"C": np.arange(8)+2, "S": np.array([3, 5])},
                                strength=100, max_step=0.05, fraction=0.25)
    assert np.array_equal(np.flatnonzero(np.linalg.norm(force, axis=1)), [6, 7, 9])
    assert np.array_equal(np.sort(selected), [6, 7, 9])
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
    kwargs.setdefault("consolidate_iterations", 0)
    kwargs.setdefault("refine_strength_control", "acceptance")
    kwargs.setdefault("refine_min_step", 0)
    kwargs.setdefault("refine_selection", "mismatch")
    with patch("mastercap.generate.initialize", return_value=[]), patch("mastercap.generate.characteristic_forces", return_value={}):
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
                                       refine_acceptance="rollback", rollback_violation_fraction=0,
                                       refine_strength=1e6, refine_fraction=0.25, refine_patience=0,
                                       refine_updates=50, stage_progress=messages.append,
                                       progress=lambda i, g, m: progress.append((i, g)), progress_interval=100)
    assert output is not None
    assert report["rollback_count"] > 0
    assert report["refinement_strength_final"] < 1e6
    assert report["termination"] == "refinement_blocked"
    assert report["refinement_stop_limit"] == "damage_independent_of_push"
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
                 "--refine-acceptance", "rollback", "--consolidate-iterations", "0", "--rollback-violation-fraction", "0",
                 "--refine-selection", "mismatch",
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


@pytest.mark.parametrize("falling", [False, True])
def test_damage_controller_distinguishes_independent_and_responsive_damage(falling):
    controller = PushController(0.005, 10, 10000, [], violation_fraction=0)
    xyz = np.zeros((2, 3))
    for i in range(3):
        controller.start(2*i, xyz, {"strength": controller.strength})
        assert controller.observe(2*i+1, geometry(False, 100//2**i if falling else 100, 2), xyz)
    assert controller.blocked is (not falling)
    if not falling:
        assert controller.strength == pytest.approx(0.005/4)
        assert controller.blocked_evidence["reason"] == "damage_independent_of_push"


def test_responsive_damage_can_still_reach_strength_floor():
    controller = PushController(0.005, 10, 10000, [], violation_fraction=0)
    xyz = np.zeros((2, 3))
    for i in range(5):
        controller.start(2*i, xyz, {"strength": controller.strength})
        controller.observe(2*i+1, geometry(False, 1000//2**i, 2), xyz)
    assert controller.stalled
    assert not controller.blocked


@pytest.mark.parametrize("improved", [False, True])
def test_rollback_judges_recovered_profile_against_snapshot(improved):
    history = []
    controller = PushController(0.005, 10, 10000, history)
    xyz = np.zeros((2, 3))
    controller.start(0, xyz, {"strength": controller.strength}, (0.1, 0, 0, 0))
    xyz += 0.1
    assert not controller.observe(1, geometry(False, 1, 2), xyz)
    restored = controller.observe(2, geometry(), xyz, (0.09 if improved else 0.11, 0, 0, 0))
    assert restored is (not improved)
    assert history[0]["outcome"] == ("recovered_improved" if improved else "recovered_worse")
    assert np.allclose(xyz, 0.1 if improved else 0)


def test_restored_snapshots_do_not_count_as_new_feasible_candidates():
    source, edges, profiles, analysis = controlled_case()
    _, report = controlled_search(source, edges, profiles, analysis, refine_acceptance="rollback",
                                   refine_strength=1e-10, refine_updates=3, refine_patience=0)
    assert report["restored_snapshot_iterations"] == [2, 4]
    assert report["feasible_iterations"] == [0]
    assert [r["iteration"] for r in report["best_history"]] == [0]
    assert report["recoveries_worse"] == 3


@pytest.mark.parametrize("fraction,expected", [(0.9, 0.00625), (0.6, 0.005), (0.4, 0.0035)])
def test_acceptance_controller_targets_the_acceptance_band(fraction, expected):
    controller = AcceptanceController(0.005, [], control="acceptance")
    controller.start(0, np.zeros((2, 3)), {"acceptance_fraction": fraction})
    assert controller.strength == pytest.approx(expected)


def test_fixed_strength_remains_constant_after_ten_pushes():
    controller = AcceptanceController(0.005, [], control="fixed")
    xyz = np.zeros((2, 3))
    for iteration in range(10):
        controller.start(iteration, xyz, {"acceptance_fraction": 0.1}, (1, 0, 0, 0))
        controller.observe(iteration+1, geometry(), xyz, (1, 0, 0, 0))
    assert controller.strength == 0.005
    assert not controller.stalled


def test_gain_control_reduces_after_two_stalls_and_grows_on_gain():
    controller = AcceptanceController(0.005, [], control="gain")
    xyz = np.zeros((2, 3))
    for iteration in range(2):
        controller.start(iteration, xyz, {"acceptance_fraction": 0.9}, (1, 0, 0, 0))
        controller.observe(iteration+1, geometry(), xyz, (1, 0, 0, 0))
    assert controller.strength == pytest.approx(0.0035)
    controller.start(2, xyz, {"acceptance_fraction": 0.9}, (1, 0, 0, 0))
    controller.observe(3, geometry(), xyz, (0.9, 0, 0, 0))
    assert controller.strength == pytest.approx(0.0035*1.25)


def test_minimum_requested_step_survives_tiny_strength():
    force = np.zeros((4, 3))
    radial = np.ones(4)
    direction = np.tile([1.0, 0, 0], (4, 1))
    _, selected = apply_radial_push(force, radial, direction, {"C": np.arange(4)},
                                    {"C": radial+np.array([-1, 0, 0.001, 2])},
                                    1e-12, 0.05, 1, min_step=0.002)
    assert abs(force[0, 0]) == pytest.approx(0.002)
    assert force[3, 0] == pytest.approx(0.002)
    assert force[1, 0] == 0


def crossing_case(blocked=None, fraction=1, matched=False):
    radial = np.array([1.98, 1.99, 1.1, 2.5])
    edges = np.array([0, 2, 4])
    profiles = {"C": np.array([3, 1] if matched else [1, 3])/shell_volumes(edges)}
    force = np.zeros((4, 3))
    details, selected = apply_crossing_push(force, radial, np.tile([1.0, 0, 0], (4, 1)),
                                           {"C": np.arange(4)}, profiles, edges, 0.005, 0.05, fraction,
                                           blocked=blocked)
    return radial, edges, profiles, force, details, selected


def test_crossing_quotas_reduce_l1_count_error_by_twice_the_crossings():
    radial, edges, profiles, force, details, _ = crossing_case()
    _, before = bin_inventory(radial, {"C": np.arange(4)}, edges, profiles)
    _, after = bin_inventory(radial+force[:, 0], {"C": np.arange(4)}, edges, profiles)
    assert details["per_element"]["C"]["crossing_proposals"] == 2
    assert np.abs(before["C"][2]).sum()-np.abs(after["C"][2]).sum() == 4


def test_matching_bins_propose_no_crossing_or_drift_moves():
    _, _, _, force, details, selected = crossing_case(matched=True)
    assert len(selected) == 0
    assert not force.any()
    assert details["per_element"]["C"]["crossing_proposals"] == 0


def test_previous_rejections_are_skipped_by_crossing_selection():
    _, _, _, _, details, selected = crossing_case(blocked=np.array([True, False, False, False]))
    assert 0 not in selected
    assert details["per_element"]["C"]["crossing_proposals"] == 1


def test_drift_uses_remaining_fraction_budget_and_minimum_step():
    radial = np.array([1, 1.2, 1.4, 2.5])
    edges = np.array([0, 2, 4])
    profiles = {"C": np.array([1, 3])/shell_volumes(edges)}
    force = np.zeros((4, 3))
    details, selected = apply_crossing_push(force, radial, np.tile([1.0, 0, 0], (4, 1)),
                                           {"C": np.arange(4)}, profiles, edges, 1e-12, 0.05, 0.5)
    assert len(selected) == 2
    assert details["per_element"]["C"]["drift_proposals"] == 2
    assert np.all(force[selected, 0] == 0.002)


@pytest.mark.parametrize("mode", ["independent", "residue"])
def test_large_per_atom_push_preserves_geometry_and_seed_reproducibility(mode):
    results = []
    for _ in range(2):
        source, edges, profiles, analysis = controlled_case(lower=1.59, all_inner=True)
        output, report = controlled_search(source, edges, profiles, analysis, mode=mode,
                                           refine_strength=1e6, refine_updates=3, refine_patience=0)
        assert validate_geometry(output, np.zeros(3), edges, analysis.limits, mode=mode)["mandatory_pass"]
        assert report["rollback_count"] == 0
        assert any(p["atoms_undone"] > 0 for p in report["push_history"])
        for p in report["push_history"]:
            assert p["atoms_accepted"]+p["atoms_undone"] == p["atoms_proposed"]
            assert p["damage"]["mandatory_pass"]
            assert p["profile_before"] and p["profile_after"]
            assert set(p["force_components"]) == {"repair", "bridges", "radial_clip", "centering", "radial_push", "nearest_neighbor"}
        json.dumps(report)
        results.append((output.xyz.copy(), report["push_history"]))
    assert np.array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


@pytest.mark.parametrize("option,value", [
    ("--refine-slack", "-1"), ("--refine-slack", "nan"), ("--consolidate-iterations", "-1"),
    ("--rollback-violation-fraction", "-0.1"), ("--rollback-violation-fraction", "1.1"),
    ("--rollback-component-limit", "0"), ("--refine-acceptance", "invalid"), ("--neighbor-forces", "invalid")])
def test_new_options_reject_invalid_values_without_reading_input(option, value):
    with pytest.raises(SystemExit) as stopped:
        main(["generate", "--input", "missing.pdb", "--output-prefix", "out", option, value])
    assert stopped.value.code == 2


@pytest.mark.parametrize("setting", [
    {"refine_slack": -1}, {"refine_slack": float("nan")}, {"consolidate_iterations": -1},
    {"rollback_violation_fraction": -0.1}, {"rollback_violation_fraction": 1.1},
    {"rollback_component_limit": 0}, {"refine_acceptance": "invalid"},
    {"refine_strength_control": "invalid"}, {"refine_min_step": -1}, {"refine_min_step": 0.1},
    {"refine_min_step": float("nan")}, {"consolidate_target": 1.1},
    {"reconsolidate_below": -1}, {"reconsolidate_below": 1.1},
    {"reconsolidate_every": -1}, {"reconsolidate_iterations": -1}, {"refine_selection": "invalid"}])
def test_library_rejects_new_invalid_settings(setting):
    source, edges, profiles, analysis = controlled_case()
    with pytest.raises(MastercapError):
        controlled_search(source, edges, profiles, analysis, **setting)
