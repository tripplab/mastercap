"""Scientific effort accounting must agree with committed trial geometry."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mastercap.analysis import shell_volumes
from mastercap.generate import search
from mastercap.model import Atom, Structure
from mastercap.refinement import crossing_metrics, radial_bins


def trace_case(**options):
    structure = Structure([Atom(e, e, "ALA", i) for i, e in enumerate(("C", "C", "N", "N"))],
                          np.array([[-2.4, 0, 0], [-0.8, 0, 0], [0.8, 0, 0], [2.4, 0, 0]]))
    edges = np.array([0.0, 2.0, 4.0])
    profiles = {e: np.array([1.0, 1.0])/shell_volumes(edges) for e in ("C", "N")}
    analysis = SimpleNamespace(limits={(a, b): 0.1 for a, b in (("C", "C"), ("C", "N"), ("N", "N"))},
                               original=profiles)
    settings = dict(neighbor_forces="off", consolidate_iterations=0, refine_strength=1e-10,
                    refine_updates=3, refine_patience=0, time_limit=30, refine_min_step=0,
                    refine_selection="mismatch")
    settings.update(options)
    with patch("mastercap.generate.initialize", return_value=[]):
        return search(structure, analysis, edges, profiles, np.zeros(3), np.random.default_rng(42), **settings)


def test_per_element_counts_match_push_totals_and_profile_changes():
    _, report = trace_case()
    for push in report["push_history"]:
        per = push["per_element"]
        assert sum(v["selected"] for v in per.values()) == push["selected_atoms"]
        assert sum(v["accepted"] for v in per.values()) == push["atoms_accepted"]
        assert sum(v["undone"] for v in per.values()) == push["atoms_undone"]
        assert sum(v["undone_too_close"] for v in per.values()) == push["undone_by_reason"]["too_close"]
        assert sum(v["undone_contact_broken"] for v in per.values()) == push["undone_by_reason"]["contact_broken"]
        for e, values in per.items():
            assert values["profile_error_change"] == (push["profile_after"]["element_errors"][e]
                                                        -push["profile_before"]["element_errors"][e])


def test_no_bin_crossing_when_moves_do_not_reach_an_edge():
    edges = np.array([0, 2, 4])
    result = crossing_metrics(np.array([1.5, 2.5]), np.array([1.501, 2.499]),
                              {"C": np.array([0, 1])}, edges,
                              {"C": np.array([1, 1])/shell_volumes(edges)})
    assert result["C"]["bin_crossings"] == 0
    assert np.array_equal(radial_bins(np.array([0, 2, 4]), edges), [0, 1, 1])


def test_incomplete_consolidation_has_no_completed_slack_measurement():
    _, report = trace_case(consolidate_iterations=10, refine_slack=2, max_iterations=0)
    assert report["consolidation_completed"] is False
    assert report["slack_after_consolidation"] is None
    assert report["consolidation_iterations_total"] == 0
    assert report["consolidation_history"][0]["reason"] == "search_stopped"


def test_tiny_moves_classify_patience_as_below_rounding():
    _, report = trace_case(refine_patience=2)
    assert report["refinement_stop_limit"] == "refine_patience"
    assert report["stall_cause"] == "moves_below_rounding"
