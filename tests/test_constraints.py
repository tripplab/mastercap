import numpy as np
from mastercap.constraints import validate_geometry, acceptance, profile_metrics
from mastercap.model import Atom, Structure


def particle(x, residues):
    return Structure([Atom("CA", "C", "GLY", r) for r in residues],
                     np.column_stack([x, np.zeros((len(x), 2))]))


def test_neighbor_per_atom_is_not_global_connectivity():
    s = particle([-6, -4, 4, 6], [0, 0, 1, 1])
    result = validate_geometry(s, np.zeros(3), np.array([0, 10]), {("C", "C"): 1})
    assert result["global_components"] == 2
    assert not result["mandatory_pass"]


def test_internal_connectivity_is_independent_of_global():
    s = particle([0, 2, 4], [0, 1, 0])
    result = validate_geometry(s, np.array([2, 0, 0]), np.array([0, 5]), {("C", "C"): 1})
    assert result["global_components"] == 1
    assert result["disconnected_residues"] == [0]


def test_tangency_does_not_connect():
    s = particle([0, 3.4], [0, 1])
    result = validate_geometry(s, np.array([1.7, 0, 0]), np.array([0, 5]), {("C", "C"): 1})
    assert result["global_components"] == 2


def test_pdb_rounding_can_break_overlap():
    s = particle([0, 3.3996], [0, 1])
    assert validate_geometry(s, np.array([1.7, 0, 0]), [0, 5], {("C", "C"): 1})["mandatory_pass"]
    s.xyz = np.round(s.xyz, 3)
    assert not validate_geometry(s, np.array([1.7, 0, 0]), [0, 5], {("C", "C"): 1})["mandatory_pass"]


def test_rigid_internal_exception_and_center_tolerance():
    s = particle([0, 1.4], [0, 0])
    report = validate_geometry(s, np.zeros(3), [0, 2], {("C", "C"): 1.5}, mode="residue")
    assert report["mandatory_pass"]
    assert report["preserved_internal_distance_exceptions"] == 1
    report = validate_geometry(s, np.array([3, 0, 0]), [0, 5], {("C", "C"): 1.5}, mode="residue")
    assert not report["mandatory_pass"]


def test_no_soft_tolerance_can_override_mandatory_failure():
    assert acceptance({"mandatory_pass": False}, {}, 100, 100) == "failed"


def test_one_atom_does_not_satisfy_overlap_requirement():
    result = validate_geometry(particle([0], [0]), np.zeros(3), [0, 5], {("C", "C"): 1})
    assert result["isolated_atoms"] == 1
    assert not result["mandatory_pass"]
