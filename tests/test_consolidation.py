from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mastercap.analysis import shell_volumes
from mastercap.generate import search
from mastercap.model import Atom, Structure


def consolidation_case(lower=1.997, separation=2.0, **settings):
    structure = Structure([Atom("C", "C", "GLY", i) for i in range(2)],
                          np.array([[-separation/2, 0, 0], [separation/2, 0, 0]]))
    edges = np.array([0.0, 2.0, 4.0])
    profiles = {"C": np.array([2.0, 0])/shell_volumes(edges)}
    analysis = SimpleNamespace(limits={("C", "C"): lower}, original=profiles)
    observations = []
    options = dict(max_iterations=1, time_limit=30, refine_patience=0,
                   refine_selection="mismatch",
                   progress=lambda i, g, m: observations.append(g), progress_interval=1)
    options.update(settings)
    with patch("mastercap.generate.initialize", return_value=[]):
        output, report = search(structure, analysis, edges, profiles, np.zeros(3), np.random.default_rng(42), **options)
    return output, report, observations


def test_checked_consolidation_reduces_near_counts_without_using_push_budget():
    _, report, observations = consolidation_case()
    assert all(g["mandatory_pass"] for g in observations)
    assert report["consolidation_completed"]
    assert report["profile_updates"] == 0
    episode = report["consolidation_history"][0]
    assert episode["after"]["distance_within_half_slack"] < episode["before"]["distance_within_half_slack"]
    assert episode["steps"] == 1


def test_ready_structure_completes_one_checked_step_with_target_reason():
    _, report, observations = consolidation_case(lower=1.0)
    assert report["consolidation_history"][0]["steps"] == 1
    assert report["consolidation_history"][0]["reason"] == "target"
    assert report["consolidation_iterations_total"] == 1
    assert all(g["mandatory_pass"] for g in observations)


def test_periodic_episodes_do_not_consume_push_budget():
    _, report, observations = consolidation_case(lower=1, max_iterations=100,
                                                 refine_updates=5, reconsolidate_below=0, reconsolidate_every=2)
    assert report["profile_updates"] == 5
    assert [e["trigger"] for e in report["consolidation_history"]] == ["initial", "periodic", "periodic"]
    assert report["consolidation_iterations_total"] == sum(e["steps"] for e in report["consolidation_history"])
    assert all(g["mandatory_pass"] for g in observations)


def test_low_acceptance_triggers_checked_slack_restoration():
    _, report, observations = consolidation_case(lower=3.197, separation=3.2, max_iterations=100,
                                                 refine_strength=1e6, refine_updates=5,
                                                 reconsolidate_below=0.7, reconsolidate_every=0)
    assert any(p["acceptance_fraction"] < 0.7 for p in report["push_history"])
    assert any(e["trigger"] == "low_acceptance" for e in report["consolidation_history"])
    assert report["profile_updates"] == 5
    assert all(g["mandatory_pass"] for g in observations)


def test_disabled_triggers_leave_only_initial_consolidation():
    _, report, _ = consolidation_case(lower=1, max_iterations=100, refine_updates=12,
                                     reconsolidate_below=0, reconsolidate_every=0)
    assert len(report["consolidation_history"]) == 1
    assert report["consolidation_history"][0]["trigger"] == "initial"
    assert report["profile_updates"] == 12
