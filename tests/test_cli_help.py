"""Help must be complete, useful without I/O, and consistent with real parsers."""
import argparse
from pathlib import Path
import re
import shlex
import tomllib

import pytest

from mastercap import __version__
from mastercap.cli import COMMAND_EXAMPLES, main, parser
import mastercap.cli as cli


def normalized(text):
    return re.sub(r"\s+", " ", text)


def test_root_help_lists_all_commands_options_descriptions_and_defaults(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/mastercap-help-cache")

    def must_not_run(*args, **kwargs):
        pytest.fail("Help must not read references, download data, or start a workflow")

    monkeypatch.setattr(cli, "run", must_not_run)
    with pytest.raises(SystemExit) as stopped:
        main(["-h"])
    assert stopped.value.code == 0
    captured = capsys.readouterr()
    assert not captured.err
    assert captured.out.startswith(f"mastercap {__version__}\n\n")
    output = normalized(captured.out)
    reference = parser()
    for command in reference.command_parsers:
        assert f"usage: {command.prog}" in output
        for action in command._actions:
            if action.help == argparse.SUPPRESS:
                continue
            assert action.help, f"Missing explanation for {action.option_strings}"
            for flag in action.option_strings:
                assert flag in captured.out
    for content in ("default: 300.0", "default: 1000", "default: independent",
                    "default: random", "default: reference", "default: extrema",
                    "default: [0.5, 99.5]", "default: False", "default: inferred from the reference",
                    "default: generate a seed", "default: unset (uncalibrated)",
                    "/tmp/mastercap-help-cache/mastercap", "Examples:", "Exit codes:",
                    "mandatory geometry/connectivity", "preparation", "PREFIX.diagnostic.json"):
        assert content in output
    assert "default: None" not in output


@pytest.mark.parametrize("command", ["analyze", "generate", "validate"])
def test_command_help_is_focused_and_complete(command, capsys):
    with pytest.raises(SystemExit) as stopped:
        main([command, "--help"])
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert output.startswith(f"mastercap {__version__}\n\n")
    assert f"usage: mastercap {command}" in output
    assert "Examples:" in output
    assert "Exit codes:" in output
    assert "--input" in output and "--id" in output
    assert "default:" in output
    for other in {"analyze", "generate", "validate"}-{command}:
        assert f"usage: mastercap {other}" not in output
    if command == "generate":
        assert "--profile-tolerance" in output and "--inner-radius" in output
        assert "--time-limit" in output and "--allow-approximate" in output
    elif command == "validate":
        assert "--report" in output and "--pdb" in output
        assert "--profile-tolerance" not in output


@pytest.mark.parametrize("command", ["analyze", "generate", "validate"])
def test_documented_help_examples_parse_without_execution(command):
    for line in COMMAND_EXAMPLES[command].splitlines():
        if line.strip().startswith("mastercap "):
            args = parser().parse_args(shlex.split(line.strip())[1:])
            assert args.command == command


def test_help_displays_changed_parser_defaults_instead_of_static_copies():
    reference = parser()
    generator = next(command for command in reference.command_parsers if command.prog.endswith(" generate"))
    generator.set_defaults(seed=73, max_iterations=42)
    output = normalized(reference.format_help())
    assert "default: 42" in output
    assert "default: 73" in output


def test_version_matches_project_metadata_and_cli(capsys):
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root/"pyproject.toml").read_text())
    assert __version__ == metadata["project"]["version"] == "0.1.7"
    assert f"**Current version: {__version__}**" in (root/"README.md").read_text()
    with pytest.raises(SystemExit) as stopped:
        main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.strip() == __version__
