import json
from pathlib import Path
import numpy as np
from mastercap.cli import main
from mastercap.analysis import analyze, target_profiles
from mastercap.generate import select_inventory, search, residue_groups
from mastercap.pdbio import read_pdb
from mastercap.reports import publish


def test_analyze_workflow_and_overwrite_protection(tmp_path, small_pdb):
    prefix = tmp_path/"analysis"
    args = ["analyze", "--input", str(small_pdb), "--output-prefix", str(prefix), "--csv", "--no-smoothing"]
    assert main(args) == 0
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["analysis"]["atom_count"] == 4
    original = prefix.with_suffix(".json").read_bytes()
    assert main(args) == 1
    assert prefix.with_suffix(".json").read_bytes() == original


def test_inventory_scaled_by_volume(small_structure):
    analysis = analyze(small_structure, sigma=0)
    edges, profiles = target_profiles(analysis, analysis.inner*2, analysis.outer*2)
    inventory, trace = select_inventory(analysis, edges, profiles, np.random.default_rng(4))
    assert len(inventory.atoms) == 32
    assert len(trace) == 16


def test_failed_generation_writes_diagnostic_only(tmp_path, small_pdb):
    prefix = tmp_path/"failure"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--overlap-margin", "4", "--seed", "1", "--max-iterations", "0", "--allow-approximate"]) == 1
    assert not prefix.with_suffix(".pdb").exists()
    assert Path(str(prefix)+".diagnostic.json").exists()


def test_generation_and_independent_validation(tmp_path, small_pdb):
    prefix = tmp_path/"particle"
    args = ["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
            "--no-smoothing", "--seed", "42", "--max-iterations", "600", "--time-limit", "30",
            "--allow-approximate"]
    assert main(args) == 2
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["geometry"]["mandatory_pass"]
    assert report["status"] == "uncalibrated"
    assert main(["validate", "--input", str(small_pdb), "--pdb", str(prefix.with_suffix(".pdb")),
                 "--report", str(prefix.with_suffix(".json")), "--output-prefix", str(tmp_path/"validation")]) == 2
    second = tmp_path/"second"
    args[args.index(str(prefix))] = str(second)
    assert main(args) == 2
    assert prefix.with_suffix(".pdb").read_bytes() == second.with_suffix(".pdb").read_bytes()


def test_rigid_mode_preserves_internal_distances(tmp_path, small_pdb):
    prefix = tmp_path/"rigid"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--mode", "residue", "--no-smoothing", "--seed", "42", "--allow-approximate"]) == 2
    generated = read_pdb(prefix.with_suffix(".pdb"), generated=True)
    for group in residue_groups(generated):
        assert abs(np.linalg.norm(generated.xyz[group[0]]-generated.xyz[group[1]])-1.6) < 0.002


def test_regular_generation(tmp_path, small_pdb):
    prefix = tmp_path/"regular"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--distribution", "regular", "--no-smoothing", "--seed", "42",
                 "--allow-approximate", "--max-iterations", "600"]) == 2
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["geometry"]["mandatory_pass"]


def test_explicit_soft_tolerances_can_produce_valid_output(tmp_path, small_pdb):
    prefix = tmp_path/"valid"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--no-smoothing", "--seed", "42", "--profile-tolerance", "2",
                 "--composition-tolerance", "0"]) == 0
    assert json.loads(prefix.with_suffix(".json").read_text())["status"] == "valid"


def test_uncalibrated_output_needs_explicit_permission(tmp_path, small_pdb):
    prefix = tmp_path/"uncalibrated"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--no-smoothing", "--seed", "42"]) == 1
    assert not prefix.with_suffix(".pdb").exists()
    assert Path(str(prefix)+".diagnostic.json").exists()


def test_soft_failure_can_be_saved_without_relaxing_geometry(tmp_path, small_pdb):
    prefix = tmp_path/"approximation"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--smoothing-sigma", "0.3", "--bin-width", "0.2", "--seed", "42", "--profile-tolerance", "0",
                 "--composition-tolerance", "0", "--allow-approximate", "--max-iterations", "50"]) == 2
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["status"] == "approximate"
    assert report["geometry"]["mandatory_pass"]


def test_missing_reference_writes_generation_diagnostic(tmp_path):
    prefix = tmp_path/"failed_download"
    assert main(["generate", "--input", str(tmp_path/"missing.pdb"), "--output-prefix", str(prefix)]) == 1
    assert Path(str(prefix)+".diagnostic.json").exists()


def test_png_and_csv_are_real_outputs(tmp_path, small_pdb, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path/"mpl-cache"))
    prefix = tmp_path/"profiles"
    assert main(["analyze", "--input", str(small_pdb), "--output-prefix", str(prefix), "--csv", "--plot"]) == 0
    assert Path(str(prefix)+"_profiles.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "original_atoms_per_A3" in Path(str(prefix)+"_profiles.csv").read_text()


def test_report_cannot_validate_another_pdb(tmp_path, small_pdb):
    prefix = tmp_path/"particle"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--seed", "42", "--no-smoothing", "--allow-approximate"]) == 2
    path = prefix.with_suffix(".pdb")
    path.write_text(path.read_text()+"REMARK MODIFIED\n")
    assert main(["validate", "--input", str(small_pdb), "--pdb", str(path), "--report",
                 str(prefix.with_suffix(".json")), "--output-prefix", str(tmp_path/"check")]) == 1


def test_publish_rolls_back_on_promotion_failure(tmp_path, monkeypatch):
    import os
    import pytest
    first, second = tmp_path/"first", tmp_path/"second"
    first.write_text("first original")
    second.write_text("second original")
    original_replace = os.replace
    attempts = 0
    def fail_second(src, dst):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError("simulated disk error")
        return original_replace(src, dst)
    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(OSError):
        publish({first: "new first", second: "new second"}, overwrite=True)
    assert first.read_text() == "first original"
    assert second.read_text() == "second original"


def test_publish_protects_input(tmp_path):
    import pytest
    from mastercap.model import MastercapError
    path = tmp_path/"input.pdb"
    path.write_text("original")
    with pytest.raises(MastercapError):
        publish({path: "replacement"}, overwrite=True, protected=[path])
    assert path.read_text() == "original"


def test_connected_spherical_shell_search():
    from mastercap.model import Atom, Structure
    n = 60
    k = np.arange(n)
    z = 1-2*(k+0.5)/n
    phi = k*np.pi*(3-np.sqrt(5))
    directions = np.column_stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z])
    centers = directions*(6+0.3*np.sin(k))[:, None]
    tangent = np.cross(directions, [0, 0, 1.0])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    coords = np.array([p for i in range(n) for p in (centers[i]-0.7*tangent[i], centers[i]+0.7*tangent[i])])
    source = Structure([Atom("CA" if i%2 == 0 else "C", "C", "GLY", i//2) for i in range(2*n)], coords)
    analysis = analyze(source, sigma=0.4)
    edges, profiles = target_profiles(analysis)
    rng = np.random.default_rng(42)
    inventory, _ = select_inventory(analysis, edges, profiles, rng)
    particle, report = search(inventory, analysis, edges, profiles, analysis.center, rng,
                               max_iterations=500, time_limit=20)
    assert particle is not None
    assert len(particle.atoms) == 120
    assert report["geometry"]["mandatory_pass"]


def test_download_uses_url_and_validated_cache(tmp_path, small_pdb, monkeypatch):
    import gzip
    import io
    from mastercap.pdbio import download
    calls = []
    def mock_urlopen(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(gzip.compress(small_pdb.read_bytes()))
    monkeypatch.setattr("mastercap.pdbio.urlopen", mock_urlopen)
    path, metadata = download("1CWP", tmp_path/"cache")
    download("1cwp", tmp_path/"cache")
    assert calls == ["https://viperdb.org/resources/OLIGOMERS/1cwp_full.vdb.gz"]
    assert len(read_pdb(path).atoms) == 4
    assert metadata["url"] == calls[0]
    download("1cwp", tmp_path/"cache", refresh=True)
    assert len(calls) == 2
