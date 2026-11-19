import numpy as np
from mastercap.analysis import analyze, target_profiles, shell_volumes, distance_statistics
from mastercap.model import Atom, Structure


def test_smoothing_preserves_integrated_element_counts(small_structure):
    analysis = analyze(small_structure, width=0.2, sigma=0.4)
    volumes = shell_volumes(analysis.edges)
    for e in analysis.original:
        assert np.isclose(np.sum(analysis.original[e]*volumes), np.sum(analysis.smoothed[e]*volumes))


def test_resize_preserves_density_not_count(small_structure):
    analysis = analyze(small_structure, width=0.2)
    edges, profiles = target_profiles(analysis, analysis.inner*2, analysis.outer*2)
    assert np.array_equal(profiles["C"], analysis.smoothed["C"])
    assert np.isclose(np.sum(profiles["C"]*shell_volumes(edges)), 8*len(small_structure.atoms))


def test_pair_statistics_and_grouped_fallback():
    s = Structure([Atom("CA", "C", "ALA", 0), Atom("N", "N", "ALA", 0), Atom("O", "O", "ALA", 0)],
                  np.array([[0, 0, 0], [1.4, 0, 0], [0, 2, 0]]))
    limits, report = distance_statistics(s)
    assert all(v > 0 for v in limits.values())
    assert all(item["grouped_fallback"] for item in report.values())
    assert report["C-N"]["count"] == 2


def test_percentiles_exclude_and_report_atoms(small_structure):
    small_structure.xyz[3, 0] = 5.0
    analysis = analyze(small_structure, radius_method="percentile", percentiles=(0, 80))
    assert analysis.excluded == 1
    assert len(analysis.structure.atoms) == 3


def test_mixed_element_profiles_keep_separate_counts():
    elements = ["C", "N", "O", "S", "H", "C"]
    structure = Structure([Atom(e, e, "ALA", i//3) for i, e in enumerate(elements)],
                          np.array([[1,0,0], [0,2,0], [0,0,3], [-4,0,0], [0,-5,0], [0,0,-6.0]]))
    report = analyze(structure, width=0.4, sigma=0.8)
    for e in set(elements):
        assert np.isclose(np.sum(report.smoothed[e]*shell_volumes(report.edges)), elements.count(e))
