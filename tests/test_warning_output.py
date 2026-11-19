import json

import pytest

from mastercap.cli import main


@pytest.fixture
def inferred_reference(small_pdb):
    lines = small_pdb.read_text().splitlines()
    atom = next(line for line in lines if line.startswith("ATOM"))
    lines = [line[:76]+"    " if line.startswith("ATOM") else line for line in lines]
    lines.append("HETATM"+atom[6:])
    small_pdb.write_text("\n".join(lines)+"\n")
    return small_pdb


def assert_summary_only(captured, warnings, quiet):
    terminal = captured.out+captured.err
    assert "Inferred " not in terminal
    assert "Ignored " not in terminal
    assert "Suspicious angular coverage" not in terminal
    assert "Input is assumed" not in terminal
    summaries = [line for line in terminal.splitlines() if line.startswith("WARNING: Reference analysis:")]
    assert len(summaries) == (0 if quiet else 1)
    if not quiet:
        assert f"{len(warnings)} warnings (4 element inferences)" in summaries[0]
        assert "details only in JSON" in summaries[0]
    assert sum(w.startswith("Inferred ") for w in warnings) == 4
    assert any("HETATM" in w for w in warnings)


@pytest.mark.parametrize("verbosity", [[], ["--verbose"], ["--quiet"]], ids=["default", "verbose", "quiet"])
@pytest.mark.parametrize("operation", ["analyze", "generate", "diagnostic"])
def test_warning_details_stay_in_json(tmp_path, inferred_reference, capsys, verbosity, operation):
    prefix = tmp_path/operation
    command = "generate" if operation == "diagnostic" else operation
    args = [command, "--input", str(inferred_reference), "--output-prefix", str(prefix), "--no-smoothing", *verbosity]
    if command == "generate":
        args += ["--seed", "42", "--allow-approximate"]
    if operation == "diagnostic":
        args += ["--overlap-margin", "4", "--max-iterations", "0"]
    expected_exit = {"analyze": 0, "generate": 2, "diagnostic": 1}[operation]
    assert main(args) == expected_exit
    captured = capsys.readouterr()
    report_path = tmp_path/(operation+(".diagnostic.json" if operation == "diagnostic" else ".json"))
    report = json.loads(report_path.read_text())
    assert_summary_only(captured, report["analysis"]["warnings"], "--quiet" in verbosity)


@pytest.mark.parametrize("verbosity", [[], ["--verbose"], ["--quiet"]], ids=["default", "verbose", "quiet"])
def test_validation_keeps_warning_details_in_its_json(tmp_path, inferred_reference, capsys, verbosity):
    prefix = tmp_path/"particle"
    assert main(["generate", "--input", str(inferred_reference), "--output-prefix", str(prefix),
                 "--seed", "42", "--no-smoothing", "--allow-approximate"]) == 2
    capsys.readouterr()
    validation = tmp_path/"validation"
    assert main(["validate", "--input", str(inferred_reference), "--pdb", str(prefix.with_suffix(".pdb")),
                 "--report", str(prefix.with_suffix(".json")), "--output-prefix", str(validation), *verbosity]) == 2
    captured = capsys.readouterr()
    report = json.loads(validation.with_suffix(".json").read_text())
    assert_summary_only(captured, report["warnings"], "--quiet" in verbosity)
