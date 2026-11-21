import numpy as np
from mastercap.analysis import analyze, target_profiles
from mastercap.generate import search, select_inventory


def test_refinement_histories_are_consistent(small_structure):
    analysis = analyze(small_structure, sigma=0)
    edges, profiles = target_profiles(analysis)
    rng = np.random.default_rng(42)
    structure, _ = select_inventory(analysis, edges, profiles, rng)
    messages = []
    _, report = search(structure, analysis, edges, profiles, analysis.center, rng,
                       max_iterations=30, refine_iterations=20, stage_progress=messages.append)
    assert report["initial_metrics"]["elements"]
    assert len(report["push_history"]) == report["profile_updates"] > 0
    assert set(item["iteration"] for item in report["best_history"]) <= set(report["feasible_iterations"])
    assert sum("New best feasible candidate" in message for message in messages) == len(report["best_history"])
    for item in report["best_history"]:
        assert len(item["ranking_key"]) == 4
        assert "C" in item["element_errors"]
    for push in report["push_history"]:
        assert "damage" in push
        assert push["damage"]["iteration"] == push["iteration"]+1
        assert 0 <= push["mean_requested_radial_move_angstrom"] <= push["max_requested_radial_move_angstrom"]
