"""Regressions for full-capsid timeout and contact-selection failures."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mastercap.analysis import analyze, target_profiles
from mastercap.cli import main
from mastercap.constraints import GeometryMetadata, local_geometry, validate_geometry
from mastercap.generate import connect_edges, search, select_inventory
from mastercap.model import Atom, Structure
import mastercap.generate as generation
import mastercap.cli as cli


def prepared(structure):
    analysis = analyze(structure, sigma=0)
    edges, profiles = target_profiles(analysis)
    rng = np.random.default_rng(42)
    inventory, _ = select_inventory(analysis, edges, profiles, rng)
    return inventory, analysis, edges, profiles, rng


def test_initialization_does_not_spend_repair_budget(small_structure, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(generation, "time", SimpleNamespace(monotonic=lambda: clock.now))
    initialize = generation.initialize

    def slow_initialize(*args, **kwargs):
        clock.now += 1000
        return initialize(*args, **kwargs)

    monkeypatch.setattr(generation, "initialize", slow_initialize)
    structure, analysis, edges, profiles, rng = prepared(small_structure)
    output, report = search(structure, analysis, edges, profiles, analysis.center, rng,
                            max_iterations=100, time_limit=1)
    assert output is not None
    assert report["iterations"] > 0
    assert report["preparation_seconds"] == 1000
    assert report["elapsed_seconds"] == 0
    assert report["termination"] == "constraints_satisfied"


def test_time_limit_has_its_own_termination_reason(small_structure, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(generation, "time", SimpleNamespace(monotonic=lambda: clock.now))
    local = generation.local_geometry

    def costly_geometry(*args, **kwargs):
        clock.now += 2
        return local(*args, **kwargs)

    monkeypatch.setattr(generation, "local_geometry", costly_geometry)
    structure, analysis, edges, profiles, rng = prepared(small_structure)
    output, report = search(structure, analysis, edges, profiles, analysis.center, rng,
                            max_iterations=1000, time_limit=1, margin=4)
    assert output is None
    assert report["iterations"] == 0
    assert report["elapsed_seconds"] == 2
    assert report["termination"] == "time_limit"


def test_iteration_limit_has_its_own_termination_reason(small_structure):
    structure, analysis, edges, profiles, rng = prepared(small_structure)
    output, report = search(structure, analysis, edges, profiles, analysis.center, rng,
                            max_iterations=2, time_limit=100, margin=4)
    assert output is None
    assert report["iterations"] == 2
    assert report["termination"] == "iteration_limit"


def test_one_neighbor_graph_and_one_metadata_build_per_search_iteration(small_structure):
    structure, analysis, edges, profiles, rng = prepared(small_structure)
    with patch.object(generation, "local_geometry", wraps=local_geometry) as geometry_spy, \
         patch.object(GeometryMetadata, "build", wraps=GeometryMetadata.build) as metadata_spy:
        _, report = search(structure, analysis, edges, profiles, analysis.center, rng,
                           max_iterations=2, time_limit=100, margin=4)
    assert geometry_spy.call_count == report["iterations"]+1
    assert metadata_spy.call_count == 1


def test_cached_checks_equal_fresh_independent_validation():
    structure = Structure([Atom(e, e, "ALA", r) for e, r in zip(["C", "O", "N", "S"], [0, 0, 1, 1])],
                          np.array([[0, 0, 0], [1.4, 0, 0], [3, 0, 0], [4.6, 0, 0]]))
    limits = {(a, b): 1.2 for a in ("C", "N", "O", "S") for b in ("C", "N", "O", "S") if a <= b}
    metadata = GeometryMetadata.build(structure, limits)
    local = local_geometry(structure, limits, metadata=metadata)
    fresh = validate_geometry(structure, np.array([2, 0, 0]), [0, 6], limits)
    cached = validate_geometry(structure, np.array([2, 0, 0]), [0, 6], limits, local=local, metadata=metadata)
    assert cached == fresh
    structure.xyz[-1] = [20, 0, 0]
    # Static metadata stays reusable; moving coordinates must change the graph.
    assert local_geometry(structure, limits, metadata=metadata)["components"] > local["components"]


def test_contacts_do_not_follow_arbitrary_component_ids():
    xyz = np.repeat([[0.0, 0, 0], [150.0, 0, 0], [50.0, 0, 0], [100.0, 0, 0]], 10, axis=0)
    xyz[:, 1] = np.tile(np.arange(10)*0.01, 4)
    labels = np.repeat(np.arange(4), 10)
    contacts = connect_edges(xyz, labels)
    assert len(contacts) == 3
    assert np.max(np.linalg.norm(xyz[contacts[:, 0]]-xyz[contacts[:, 1]], axis=1)) <= 50.001
    relabeled = np.array([17, 4, 99, 1])[labels]
    assert np.array_equal(contacts, connect_edges(xyz, relabeled))


def test_contact_uses_component_boundary_not_first_atom():
    xyz = np.array([[-100.0, 0, 0], [0, 0, 0], [100, 0, 0], [1, 0, 0]])
    assert np.array_equal(connect_edges(xyz, [0, 0, 1, 1]), [[1, 3]])


def test_dense_component_fallback_finds_external_boundary():
    xyz = np.column_stack([np.r_[np.arange(30)*0.01, 50+np.arange(30)*0.01], np.zeros((60, 2))])
    assert np.array_equal(connect_edges(xyz, np.repeat([0, 1], 30)), [[29, 30]])


def test_incompatible_nearest_pair_uses_other_elements():
    structure = Structure([Atom(e, e, "ALA", 0) for e in ["C", "O", "N"]],
                          np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    limits = {("C", "O"): 5, ("C", "N"): 1, ("N", "O"): 1}
    metadata = GeometryMetadata.build(structure, limits)
    assert np.array_equal(connect_edges(structure.xyz, [0, 1, 1], metadata=metadata), [[0, 2]])
    assert len(connect_edges(structure.xyz, [0, 1, 1], metadata=metadata, margin=4)) == 0


def test_internal_contacts_use_the_same_compatibility_rules():
    structure = Structure([Atom(e, e, "ALA", r) for e, r in zip(["S", "C", "O", "N"], [0, 1, 1, 1])],
                          np.array([[100.0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    limits = {("C", "O"): 5, ("C", "N"): 1, ("N", "O"): 1}
    metadata = GeometryMetadata.build(structure, limits)
    group = np.array([1, 2, 3])
    contacts = connect_edges(structure.xyz[group], [0, 1, 1], metadata=metadata, atom_indices=group)
    assert np.array_equal(group[contacts], [[1, 3]])


def test_cli_reports_prep_separately_from_search(tmp_path, small_pdb, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    timer = SimpleNamespace(monotonic=lambda: clock.now)
    monkeypatch.setattr(cli, "time", timer)
    monkeypatch.setattr(generation, "time", timer)
    read = cli.read_pdb
    initialize = generation.initialize

    def slow_read(*args, **kwargs):
        clock.now += 1000
        return read(*args, **kwargs)

    def slow_initialize(*args, **kwargs):
        clock.now += 500
        return initialize(*args, **kwargs)

    monkeypatch.setattr(cli, "read_pdb", slow_read)
    monkeypatch.setattr(generation, "initialize", slow_initialize)
    prefix = tmp_path/"timed"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--seed", "42", "--no-smoothing", "--allow-approximate", "--time-limit", "1"]) == 2
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["timings_seconds"]["analysis"] == 1000
    assert report["timings_seconds"]["initialization"] == 500
    assert report["timings_seconds"]["preparation"] == 1500
    assert report["timings_seconds"]["search"] == 0
    assert report["search"]["elapsed_seconds"] == 0
    assert report["geometry"]["mandatory_pass"]


def test_fallback_shares_only_spent_repair_time(tmp_path, small_pdb, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    timer = SimpleNamespace(monotonic=lambda: clock.now)
    monkeypatch.setattr(cli, "time", timer)
    monkeypatch.setattr(generation, "time", timer)
    real_search = cli.search
    budgets = []

    def fail_rigid_then_search(*args, **kwargs):
        budgets.append(kwargs["time_limit"])
        if kwargs["mode"] == "residue":
            clock.now += 1002
            return None, {"preparation_seconds": 1000, "elapsed_seconds": 2, "termination": "iteration_limit"}
        return real_search(*args, **kwargs)

    monkeypatch.setattr(cli, "search", fail_rigid_then_search)
    prefix = tmp_path/"fallback"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--seed", "42", "--no-smoothing", "--allow-approximate", "--mode", "residue",
                 "--allow-independent-fallback", "--time-limit", "10"]) == 2
    assert budgets == [10, 8]
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["timings_seconds"]["search"] == 2
    assert report["effective_mode"] == "independent"


def test_exhausted_repair_budget_prevents_fresh_fallback(tmp_path, small_pdb, monkeypatch):
    attempts = []
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(cli, "time", SimpleNamespace(monotonic=lambda: clock.now))

    def exhausted(*args, **kwargs):
        attempts.append(kwargs["mode"])
        clock.now += 10
        return None, {"preparation_seconds": 0, "elapsed_seconds": 10,
                      "termination": "time_limit", "iterations": 28}

    monkeypatch.setattr(cli, "search", exhausted)
    prefix = tmp_path/"exhausted"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--seed", "42", "--mode", "residue", "--allow-independent-fallback",
                 "--time-limit", "10"]) == 1
    assert attempts == ["residue"]
    report = json.loads((tmp_path/"exhausted.diagnostic.json").read_text())
    assert report["search"]["termination"] == "time_limit"
    assert not prefix.with_suffix(".pdb").exists()
