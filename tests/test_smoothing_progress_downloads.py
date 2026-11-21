import gzip
import io
import json
import ssl
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from mastercap import __version__
from mastercap.analysis import analyze, shell_volumes, target_profiles
from mastercap.cli import analysis_options, main, parser
from mastercap.generate import profile_quantiles, search, select_inventory
from mastercap.model import Atom, MastercapError, Structure
from mastercap.pdbio import download
from mastercap.reports import profile_png


def test_no_smoothing_overrides_sigma_and_preserves_raw_profile(small_structure):
    args = parser().parse_args(["analyze", "--input", "reference.vdb", "--output-prefix", "out",
                                "--no-smoothing", "--smoothing-sigma", "8", "--bin-width", "0.2"])
    options = analysis_options(args)
    assert options["sigma"] == 0
    result = analyze(small_structure, **options)
    assert result.summary()["smoothing"] == {"enabled": False, "sigma_angstrom": 0.0}
    _, target = target_profiles(result)
    for element in result.original:
        assert np.array_equal(result.original[element], result.smoothed[element])
        assert np.array_equal(target[element], result.original[element])


def test_larger_sigma_suppresses_narrow_radial_variation(small_structure):
    narrow = analyze(small_structure, width=0.1, sigma=0.1)
    broad = analyze(small_structure, width=0.1, sigma=0.5)
    assert np.max(broad.smoothed["C"]) < np.max(narrow.smoothed["C"])
    for result in (narrow, broad):
        assert np.isclose(np.sum(result.smoothed["C"]*shell_volumes(result.edges)), 4)


@pytest.mark.parametrize("enabled", [False, True])
def test_plot_legend_respects_smoothing_and_keeps_generated_green(enabled, monkeypatch):
    from matplotlib.figure import Figure
    seen = []
    save = Figure.savefig

    def inspect(figure, *args, **kwargs):
        for axis in figure.axes:
            seen.append({line.get_label(): line.get_color() for line in axis.lines})
        return save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    data = {"C": np.array([1.0, 2.0])}
    image = profile_png(np.array([0.0, 1.0, 2.0]), data, data, data, smoothing_enabled=enabled)
    assert image.startswith(b"\x89PNG")
    for labels in seen:
        assert ("Smoothed target" in labels) == enabled
        assert labels["Generated"] == "C2"
        assert labels["Original (mapped)"] == "C0"


@pytest.mark.parametrize("operation", ["analyze", "generate", "validate"])
def test_cli_passes_effective_smoothing_to_plot(tmp_path, small_pdb, monkeypatch, operation):
    import mastercap.cli as cli
    seen = []
    real_plot = cli.profile_png

    def inspect(*args, **kwargs):
        seen.append(kwargs["smoothing_enabled"])
        return real_plot(*args, **kwargs)

    monkeypatch.setattr(cli, "profile_png", inspect)
    prefix = tmp_path/operation
    if operation == "validate":
        source = tmp_path/"generated"
        assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(source),
                     "--seed", "42", "--no-smoothing", "--allow-approximate", "--refine-iterations", "0"]) == 2
        assert main(["validate", "--input", str(small_pdb), "--pdb", str(source.with_suffix(".pdb")),
                     "--report", str(source.with_suffix(".json")), "--output-prefix", str(prefix), "--plot"]) == 2
    else:
        args = [operation, "--input", str(small_pdb), "--output-prefix", str(prefix), "--plot",
                "--no-smoothing", "--smoothing-sigma", "9"]
        if operation == "generate":
            args += ["--seed", "42", "--allow-approximate", "--refine-iterations", "0"]
        assert main(args) == (0 if operation == "analyze" else 2)
    assert seen == [False]


def test_validation_plot_honors_no_smoothing_in_legacy_report(tmp_path, small_pdb, monkeypatch):
    import mastercap.cli as cli
    prefix = tmp_path/"legacy"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--seed", "42", "--no-smoothing", "--allow-approximate", "--refine-iterations", "0"]) == 2
    report_path = prefix.with_suffix(".json")
    report = json.loads(report_path.read_text())
    report["mastercap_version"] = "0.1.2"
    report["analysis"].pop("smoothing")
    for field in ("progress_interval", "refine_iterations", "no_check_certificate", "selection_objective",
                  "refine_updates", "refine_patience", "refine_strength", "refine_max_step",
                  "refine_fraction", "refine_recovery_limit", "neighbor_forces", "refine_slack",
                  "consolidate_iterations", "refine_acceptance", "rollback_violation_fraction",
                  "rollback_component_limit", "refine_strength_control", "refine_min_step",
                  "consolidate_target", "reconsolidate_below", "reconsolidate_every",
                  "reconsolidate_iterations", "refine_selection"):
        report["configuration"].pop(field)
    report_path.write_text(json.dumps(report))
    settings = []
    real_plot = cli.profile_png

    def inspect(*args, **kwargs):
        settings.append(kwargs["smoothing_enabled"])
        return real_plot(*args, **kwargs)

    monkeypatch.setattr(cli, "profile_png", inspect)
    assert main(["validate", "--input", str(small_pdb), "--pdb", str(prefix.with_suffix(".pdb")),
                 "--report", str(report_path), "--output-prefix", str(tmp_path/"validated"), "--plot"]) == 2
    assert settings == [False]


@pytest.mark.parametrize("interval,expected", [(5, [0, 5, 10, 13]), (3, [0, 3, 6, 9, 12, 13])])
def test_progress_interval_and_final_sample(small_structure, interval, expected):
    analysis = analyze(small_structure, sigma=0)
    edges, profiles = target_profiles(analysis)
    rng = np.random.default_rng(42)
    structure, _ = select_inventory(analysis, edges, profiles, rng)
    updates = []
    _, report = search(structure, analysis, edges, profiles, analysis.center, rng,
                       max_iterations=13, time_limit=20, progress_interval=interval,
                       progress=lambda iteration, *_: updates.append(iteration))
    assert updates == expected
    assert report["iterations"] == 13


def test_stage_messages_precede_warning_and_show_refinement(tmp_path, small_pdb, capsys):
    prefix = tmp_path/"stages"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix), "--seed", "42",
                 "--no-smoothing", "--allow-approximate", "--progress-interval", "2", "--refine-iterations", "3",
                 "--consolidate-iterations", "0"]) == 2
    output = capsys.readouterr().err
    assert output.splitlines()[0] == f"INFO: mastercap {__version__}"
    assert output.index("Reading reference coordinates") < output.index("WARNING: Reference analysis")
    for text in ("Estimating element-pair", "Selecting integer residue inventory", "Initializing", "refining per-element",
                 "Rereading written PDB", "Serializing JSON"):
        assert text.lower() in output.lower()
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["search"]["profile_updates"] > 0
    assert report["search"]["termination"] == "refinement_complete"
    assert report["geometry"]["mandatory_pass"]


@pytest.mark.parametrize("option,value", [("--progress-interval", "0"), ("--refine-iterations", "-1"),
    ("--refine-min-step", "-1"), ("--refine-min-step", "nan"), ("--consolidate-target", "1.1"),
    ("--reconsolidate-below", "-1"), ("--reconsolidate-below", "1.1"),
    ("--reconsolidate-every", "-1"), ("--reconsolidate-iterations", "-1"),
    ("--refine-strength-control", "invalid"), ("--refine-selection", "invalid")])
def test_invalid_progress_and_refinement_fail_before_input(option, value, capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["generate", "--input", "missing.pdb", "--output-prefix", "out", option, value])
    assert stopped.value.code == 2
    assert "Reading reference" not in capsys.readouterr().err


def test_stratified_quantiles_preserve_sparse_target_bins():
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    target = np.array([0.0, 2.0, 0.0, 1.0])/shell_volumes(edges)
    quantiles = profile_quantiles(target, edges, 3)
    assert np.array_equal(np.histogram(quantiles, edges)[0], [0, 2, 0, 1])


def test_uncalibrated_refinement_improves_element_profile_and_keeps_geometry():
    elements = ["C", "C", "S", "S", "N"]
    # Different axes permit radial improvement without atoms crossing each other.
    source = Structure([Atom(e, e, "ALA", i) for i, e in enumerate(elements)],
                       np.array([[-2.8, 0, 0], [2.8, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 0]]))
    edges = np.array([0.0, 0.1, 2.0, 3.0])
    volume = shell_volumes(edges)
    profiles = {"C": np.array([0.0, 2, 0])/volume, "S": np.array([0.0, 0, 2])/volume,
                "N": np.array([1.0, 0, 0])/volume}
    limits = {(a, b): 0.2 for a in sorted(set(elements)) for b in sorted(set(elements)) if a <= b}
    analysis = SimpleNamespace(limits=limits, original=profiles)
    # Isolate radial refinement from random initialization and neighbor targets.
    with patch("mastercap.generate.initialize", return_value=[]), patch("mastercap.generate.characteristic_forces", return_value={}):
        _, initial = search(source.copy(), analysis, edges, profiles, np.zeros(3), np.random.default_rng(42),
                            refine_iterations=0)
        result, refined = search(source.copy(), analysis, edges, profiles, np.zeros(3), np.random.default_rng(42),
                                  refine_updates=150, refine_patience=0, selection_objective="worst-element",
                                  refine_strength_control="acceptance", refine_selection="mismatch")
    assert result is not None
    assert refined["geometry"]["mandatory_pass"]
    assert refined["status"] == "uncalibrated"
    assert refined["profile_updates"] > 0
    assert refined["profile_error_objective"] == "worst-element"
    assert all("S" in item["element_errors"] for item in refined["best_history"])
    assert refined["metrics"]["elements"]["S"]["profile_error"] < initial["metrics"]["elements"]["S"]["profile_error"]


def test_certificate_check_is_default_and_skip_is_explicit(tmp_path, small_pdb, monkeypatch):
    calls = []

    def fake(request, timeout, **kwargs):
        calls.append(kwargs)
        return io.BytesIO(gzip.compress(small_pdb.read_bytes()))

    monkeypatch.setattr("mastercap.pdbio.urlopen", fake)
    _, verified = download("1cwp", tmp_path/"verified")
    _, skipped = download("1cwp", tmp_path/"skipped", no_check_certificate=True)
    assert "context" not in calls[0]
    assert calls[1]["context"].verify_mode == ssl.CERT_NONE
    assert calls[1]["context"].check_hostname is False
    assert verified["tls_certificate_verified"] is True
    assert skipped["tls_certificate_verified"] is False
    # A later default request must not silently trust the unverified cache.
    _, refreshed = download("1cwp", tmp_path/"skipped")
    assert "context" not in calls[2]
    assert refreshed["tls_certificate_verified"] is True
    assert refreshed["cache_reused"] is False


def test_certificate_failure_is_not_automatically_bypassed(tmp_path, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise ssl.SSLCertVerificationError("certificate test failure")

    monkeypatch.setattr("mastercap.pdbio.urlopen", fail)
    with pytest.raises(MastercapError, match="certificate test failure"):
        download("1cwp", tmp_path)
    assert len(calls) == 1 and "context" not in calls[0]


def test_legacy_cache_does_not_claim_verified_tls(tmp_path, small_pdb):
    (tmp_path/"1cwp_full.vdb.gz").write_bytes(gzip.compress(small_pdb.read_bytes()))
    _, metadata = download("1cwp", tmp_path)
    assert metadata["cache_reused"]
    assert metadata["tls_certificate_verified"] is None


def test_cli_forwards_certificate_option_for_viperdb(tmp_path, small_pdb, monkeypatch):
    import mastercap.cli as cli
    seen = []

    def fake(identifier, cache, refresh, **kwargs):
        seen.append(kwargs["no_check_certificate"])
        return small_pdb, {"tls_certificate_verified": False}

    monkeypatch.setattr(cli, "download", fake)
    prefix = tmp_path/"download"
    assert main(["analyze", "--id", "1cwp", "--no-check-certificate", "--output-prefix", str(prefix)]) == 0
    assert seen == [True]
    assert json.loads(prefix.with_suffix(".json").read_text())["acquisition"]["tls_certificate_verified"] is False
