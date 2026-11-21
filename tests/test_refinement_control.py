"""Neighbor-force isolation before changing refinement acceptance."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from mastercap.constraints import GeometryMetadata, local_geometry, validate_geometry
from mastercap.refinement import accept_trial, consolidation_targets, slack_summary
from mastercap.cli import main
from mastercap.generate import characteristic_forces, search
from mastercap.model import Atom, MastercapError, Structure


def pair_case(lower=1.0, median=10.0):
    structure = Structure([Atom("C", "C", "GLY", i) for i in range(2)],
                          np.array([[0.0, 0, 0], [2.0, 0, 0]]))
    analysis = SimpleNamespace(limits={("C", "C"): lower},
                               neighbor_samples={"C-C": {"percentiles": {"50": median}}})
    return structure, analysis, GeometryMetadata.build(structure, analysis.limits)


def test_neighbor_target_is_clipped_below_contact_distance():
    structure, analysis, metadata = pair_case()
    force = np.zeros_like(structure.xyz)
    with patch("mastercap.generate.pair_forces") as apply:
        characteristic_forces(force, structure, analysis, metadata, margin=0.1)
    desired = apply.call_args.args[3]
    assert desired == pytest.approx(np.full(2, 3.4-0.1-0.003))


def test_neighbor_target_is_clipped_above_lower_limit():
    structure, analysis, metadata = pair_case(median=0.1)
    with patch("mastercap.generate.pair_forces") as apply:
        characteristic_forces(np.zeros_like(structure.xyz), structure, analysis, metadata)
    assert apply.call_args.args[3] == pytest.approx(np.full(2, 1.003))


def test_empty_neighbor_window_is_skipped():
    structure, analysis, metadata = pair_case(lower=3.4)
    force = np.zeros_like(structure.xyz)
    details = characteristic_forces(force, structure, analysis, metadata)
    assert not force.any()
    assert details["active_atoms"] == 0


def test_off_preserves_existing_force_and_skips_neighbor_queries():
    structure, analysis, metadata = pair_case()
    force = np.full_like(structure.xyz, 0.123)
    with patch("mastercap.generate.cKDTree") as tree:
        details = characteristic_forces(force, structure, analysis, metadata, mode="off")
    tree.assert_not_called()
    assert np.array_equal(force, np.full_like(force, 0.123))
    assert details["max_move_angstrom"] == 0


def test_halving_scaled_coefficient_halves_neighbor_move():
    structure, analysis, metadata = pair_case()
    moves = []
    for strength in (0.002, 0.001):
        force = np.zeros_like(structure.xyz)
        details = characteristic_forces(force, structure, analysis, metadata, mode="scaled",
                                        strength=strength, max_move=0.025,
                                        selected_indices=np.array([0]))
        assert np.array_equal(force[1], np.zeros(3))
        moves.append(details["max_move_angstrom"])
    assert moves[0] > 0
    assert moves[1] == pytest.approx(moves[0]/2)


def test_scaled_cap_is_applied_after_accumulating_all_pairs():
    # Every carbon contributes to the same sulfur: cap the accumulated vector.
    atoms = [Atom("C", "C", "GLY", i) for i in range(20)] + [Atom("S", "S", "CYS", 20)]
    xyz = np.zeros((21, 3))
    xyz[:20, 0] = np.linspace(1, 2, 20)
    structure = Structure(atoms, xyz)
    analysis = SimpleNamespace(limits={("C", "S"): 0.5},
                               neighbor_samples={"C-S": {"percentiles": {"50": 3.0}}})
    metadata = GeometryMetadata.build(structure, analysis.limits)
    force = np.zeros_like(xyz)
    details = characteristic_forces(force, structure, analysis, metadata, mode="scaled",
                                    strength=1, max_move=0.025, selected_indices=np.array([20]))
    assert not force[:20].any()
    assert np.linalg.norm(force[20]) == pytest.approx(0.025)
    assert details["max_move_angstrom"] == pytest.approx(0.025)


def test_invalid_neighbor_mode_is_rejected_before_initialization():
    structure, analysis, _ = pair_case()
    with pytest.raises(MastercapError, match="Neighbor forces"):
        search(structure, analysis, None, None, None, None, neighbor_forces="invalid")
    with pytest.raises(SystemExit) as stopped:
        main(["generate", "--input", "missing.pdb", "--output-prefix", "out",
              "--neighbor-forces", "invalid"])
    assert stopped.value.code == 2


@pytest.mark.parametrize("mode", ["scaled", "off", "fixed"])
def test_cli_neighbor_modes_reach_search_and_export_valid_pdb(tmp_path, small_pdb, mode):
    prefix = tmp_path/mode
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--seed", "42", "--allow-approximate", "--refine-updates", "1",
                 "--refine-strength", "1e-10", "--neighbor-forces", mode]) == 2
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["geometry"]["mandatory_pass"]
    assert report["search"]["refinement_settings"]["neighbor_forces"] == mode
    gap_keys = ("distance_gap_min_angstrom", "contact_gap_min_angstrom")
    before = report["search"]["slack_at_first_feasible"]
    after = report["search"]["slack_after_consolidation"]
    assert min(after[key] for key in gap_keys) > min(before[key] for key in gap_keys)
    assert min(after[key] for key in gap_keys) >= 0.01
    pushes = report["search"]["push_history"]
    assert len(pushes) == 1
    neighbor = pushes[0]["neighbor_forces"]
    assert neighbor["mode"] == mode
    if mode == "off":
        assert neighbor["max_move_angstrom"] == 0
    if mode == "scaled":
        assert neighbor["max_move_angstrom"] <= 0.025


def test_consolidation_targets_build_slack_without_changing_validation():
    from mastercap.generate import pair_forces
    structure, analysis, metadata = pair_case(lower=1.997)
    before = slack_summary(local_geometry(structure, analysis.limits, metadata=metadata), metadata)
    for _ in range(10):
        local = local_geometry(structure, analysis.limits, metadata=metadata)
        near, targets, contacts, upper = consolidation_targets(local, metadata, 0.02)
        force = np.zeros_like(structure.xyz)
        pair_forces(force, structure.xyz, near, targets, strength=0.3)
        pair_forces(force, structure.xyz, contacts, upper, strength=0.15)
        structure.xyz = np.round(structure.xyz+force, 3)
    after = slack_summary(local_geometry(structure, analysis.limits, metadata=metadata), metadata)
    assert after["distance_gap_min_angstrom"] > before["distance_gap_min_angstrom"]
    assert validate_geometry(structure, structure.xyz.mean(axis=0), np.array([0, 5]), analysis.limits)["mandatory_pass"]


def test_narrow_consolidation_window_uses_one_third_gap():
    structure, analysis, metadata = pair_case(lower=3.39)
    structure.xyz[1, 0] = 3.391
    local = local_geometry(structure, analysis.limits, metadata=metadata)
    near, targets, _, _ = consolidation_targets(local, metadata, 0.02)
    assert len(near) == 1
    assert targets[0] == pytest.approx(3.39+0.01/3)


def test_trial_rejects_broken_contact_and_keeps_feasible_snapshot():
    structure, analysis, metadata = pair_case()
    local = local_geometry(structure, analysis.limits, metadata=metadata)
    trial = structure.xyz.copy()
    trial[1, 0] = 4
    committed, report = accept_trial(structure.xyz, trial, local, metadata,
                                     structure.xyz.mean(axis=0), np.array([0, 10]))
    assert report["atoms_undone"] == 1
    assert report["undone_by_reason"]["contact_broken"] == 1
    assert np.array_equal(committed, structure.xyz)
    assert validate_geometry(Structure(structure.atoms, committed), structure.xyz.mean(axis=0),
                             np.array([0, 10]), analysis.limits)["mandatory_pass"]


def test_trial_rejects_entire_rigid_residue():
    structure = Structure([Atom("C", "C", "GLY", i//2) for i in range(4)],
                          np.array([[0.0, 0, 0], [1.2, 0, 0], [3.2, 0, 0], [4.4, 0, 0]]))
    metadata = GeometryMetadata.build(structure, {("C", "C"): 1.0})
    local = local_geometry(structure, {("C", "C"): 1.0}, mode="residue", metadata=metadata)
    trial = structure.xyz.copy()
    trial[2:] += [2, 0, 0]
    committed, report = accept_trial(structure.xyz, trial, local, metadata,
                                     structure.xyz.mean(axis=0), np.array([0, 10]), mode="residue")
    assert report["atoms_undone"] == 2
    assert np.array_equal(committed, structure.xyz)
    assert np.linalg.norm(committed[2]-committed[3]) == pytest.approx(1.2)


def test_trial_detects_new_collision_after_reverting_a_neighbor():
    structure = Structure([Atom("C", "C", "GLY", i) for i in range(3)],
                          np.array([[0.0, 0, 0], [1.2, 0, 0], [2.4, 0, 0]]))
    metadata = GeometryMetadata.build(structure, {("C", "C"): 1.0})
    local = local_geometry(structure, {("C", "C"): 1.0}, metadata=metadata)
    trial = structure.xyz+np.array([[0, 0, 0], [0.9, 0, 0], [2, 0, 0]])
    committed, report = accept_trial(structure.xyz, trial, local, metadata,
                                     structure.xyz.mean(axis=0), np.array([0, 10]))
    assert report["atoms_undone"] > 0
    assert validate_geometry(Structure(structure.atoms, committed), structure.xyz.mean(axis=0),
                             np.array([0, 10]), {("C", "C"): 1.0})["mandatory_pass"]


def test_trial_checks_large_rare_element_bound_beyond_contact_radius():
    structure = Structure([Atom("S", "S", "CYS", i) for i in range(2)],
                          np.array([[0.0, 0, 0], [9.0, 0, 0]]))
    metadata = GeometryMetadata.build(structure, {("S", "S"): 8.0})
    local = local_geometry(structure, {("S", "S"): 8.0}, metadata=metadata)
    assert len(local["pairs"]) == 0
    trial = structure.xyz.copy()
    trial[1, 0] = 7
    committed, report = accept_trial(structure.xyz, trial, local, metadata,
                                     structure.xyz.mean(axis=0), np.array([0, 10]))
    assert report["undone_by_reason"]["too_close"] == 1
    assert np.array_equal(committed, structure.xyz)


def test_mixed_trial_snapshot_pair_is_checked_after_contact_rejection():
    structure = Structure([Atom("C", "C", "GLY", i) for i in range(3)],
                          np.array([[0.0, 0, 0], [5.0, 0, 0], [6.5, 0, 0]]))
    metadata = GeometryMetadata.build(structure, {("C", "C"): 1.2})
    local = local_geometry(structure, {("C", "C"): 1.2}, metadata=metadata)
    assert not np.any(np.all(local["pairs"] == [0, 1], axis=1))
    trial = structure.xyz.copy()
    trial[:2, 0] = [4, 1]
    committed, report = accept_trial(structure.xyz, trial, local, metadata,
                                     structure.xyz.mean(axis=0), np.array([0, 10]))
    assert report["undone_by_reason"]["contact_broken"] > 0
    assert report["undone_by_reason"]["too_close"] > 0
    assert np.array_equal(committed, structure.xyz)


def test_trial_rejects_whole_push_when_center_tolerance_is_exceeded():
    structure, analysis, metadata = pair_case()
    local = local_geometry(structure, analysis.limits, metadata=metadata)
    committed, report = accept_trial(structure.xyz, structure.xyz+[1.1, 0, 0], local, metadata,
                                     structure.xyz.mean(axis=0), np.array([0, 10]))
    assert report["center_rejected"]
    assert report["atoms_accepted"] == 0
    assert np.array_equal(committed, structure.xyz)


def test_rigid_projected_trial_can_accept_moves_and_preserve_residue_shape(small_structure):
    from mastercap.generate import update_positions
    metadata = GeometryMetadata.build(small_structure, {("C", "C"): 1.59})
    local = local_geometry(small_structure, {("C", "C"): 1.59}, mode="residue", metadata=metadata)
    trial = small_structure.copy()
    force = np.tile([100.0, 0, 0], (4, 1))
    update_positions(trial, force, list(metadata.groups.values()), "residue")
    committed, report = accept_trial(small_structure.xyz, trial.xyz, local, metadata, np.zeros(3),
                                     np.array([0, 4]), mode="residue")
    assert report["atoms_accepted"] == 4
    assert np.max(np.linalg.norm(committed-small_structure.xyz, axis=1)) == pytest.approx(0.2)
    for group in metadata.groups.values():
        assert np.linalg.norm(committed[group[0]]-committed[group[1]]) == pytest.approx(1.6)
    assert validate_geometry(Structure(small_structure.atoms, committed), np.zeros(3), np.array([0, 4]),
                             {("C", "C"): 1.59}, mode="residue")["mandatory_pass"]


def test_acceptance_denominator_includes_selected_stationary_proposals():
    structure, analysis, metadata = pair_case()
    trial = structure.xyz.copy()
    trial[1, 0] = 4
    local = local_geometry(structure, analysis.limits, metadata=metadata)
    committed, report = accept_trial(structure.xyz, trial, local, metadata,
                                     structure.xyz.mean(axis=0), np.array([0, 10]),
                                     selected_indices=np.array([0, 1]))
    assert np.array_equal(committed, structure.xyz)
    assert report["atoms_proposed"] == 2
    assert report["atoms_accepted"] == report["atoms_undone"] == 1
    assert report["proposed_moving_atoms"] == 1
    assert report["atoms_moved"] == 0
    assert report["stationary_proposals"] == 1
    assert report["acceptance_fraction"] == 0.5


def test_too_close_rejections_identify_element_pairs_without_double_counting():
    structure = Structure([Atom(e,e,'ALA',i) for i,e in enumerate(('O','S','C','S'))],
                          np.array([[0.,0,0],[2.5,0,0],[8.,0,0],[10.5,0,0]]))
    limits = {('O','S'):1.5, ('C','S'):1.5}
    metadata = GeometryMetadata.build(structure, limits)
    local = local_geometry(structure, limits, metadata=metadata)
    trial = structure.xyz.copy()
    trial[1,0], trial[3,0] = 1., 9.
    _, record = accept_trial(structure.xyz,trial,local,metadata,structure.xyz.mean(axis=0),[0,20],
                              selected_indices=np.array([1,3]))
    assert record['undone_too_close_by_pair'] == {'C-S':1, 'O-S':1}
    assert record['per_element']['S']['undone_too_close_by_pair'] == {'C-S':1, 'O-S':1}
    assert record['per_element']['O']['undone_too_close_by_pair'] == {}
    assert record['atoms_proposed'] == 2
    assert record['atoms_accepted'] == 0
    assert record['atoms_undone'] == 2
    assert 0 <= record['acceptance_fraction'] <= 1
    assert sum(v['accepted']+v['undone'] for v in record['per_element'].values()) == record['atoms_proposed']


def test_noncontact_pair_builds_distance_slack_without_contact_pull():
    structure = Structure([Atom('O','O','ALA',0),Atom('S','S','CYS',1)],
                          np.array([[0.,0,0],[3.47,0,0]]))
    metadata = GeometryMetadata.build(structure,{('O','S'):3.47})
    local = local_geometry(structure,{('O','S'):3.47},metadata=metadata)
    pairs, targets, touching, _ = consolidation_targets(local,metadata,.02)
    assert len(pairs) == 1 and len(touching) == 0
    assert targets[0] == pytest.approx(3.49)
    assert len(consolidation_targets(local,metadata,.02,noncontact_slack=False)[0]) == 0


@pytest.mark.parametrize('override_mode', ['steric','number','reference'])
def test_statistical_neighbor_force_skips_only_actual_limit_overrides(override_mode):
    structure = Structure([Atom('S','S','CYS',i) for i in range(2)],np.array([[0.,0,0],[2.,0,0]]))
    analysis = SimpleNamespace(limits={('S','S'):1.5},
        neighbor_samples={'S-S':{'percentiles':{'50':20.0}}},
        distance_limit_overrides=[{'pair':'S-S','mode':override_mode}])
    metadata = GeometryMetadata.build(structure,analysis.limits)
    force = np.zeros_like(structure.xyz)
    characteristic_forces(force,structure,analysis,metadata)
    assert bool(force.any()) == (override_mode=='reference')
