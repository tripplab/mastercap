"""Command-line workflows. Help is part of the public interface."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import resource
import secrets
import shlex
import sys
import tempfile
import time
import numpy as np
import scipy

from . import __version__
from .analysis import (analyze, target_profiles, distance_statistics, parse_limit_override,
                       resolve_limit_overrides)
from .constraints import (acceptance, profile_metrics, validate_geometry,
                          connectivity_audit, require_connectivity_audit,
                          pair_distance_audit, require_pair_distance_audit)
from .tolerances import profile_tolerance_spec, check_coverage
from .calibration import calibrate, profile_settings, settings_from_generation, load_tolerances
from .generate import select_inventory, search, residue_transforms
from .model import Atom, MastercapError
from .pdbio import download, read_pdb, pdb_text
from .reports import encode_json, preflight, publish, profile_csv, profile_png, calibration_csv, calibration_png

LOG = logging.getLogger("mastercap")


class PairLimitAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        try:
            value = parse_limit_override(value)
            current = list(getattr(namespace, self.dest) or [])
            if any(spec.split("=")[0] == value.split("=")[0] for spec in current):
                raise MastercapError(f"Pair {value.split('=')[0]} specified twice")
        except MastercapError as exc:
            raise argparse.ArgumentError(self, str(exc)) from exc
        setattr(namespace, self.dest, current+[value])


def final_connectivity_check(structure, margin, geometry, report):
    audit = connectivity_audit(structure, margin)
    report["connectivity_audit"] = audit
    require_connectivity_audit(audit, geometry)
    LOG.info("Success! Final connectivity audit: 0 isolated atoms, 1 component, 0 disconnected residues (%d atoms, %d edges)",
             audit["atom_count"], audit["edge_count"])



def final_pair_distance_check(structure, limits, report):
    audit = pair_distance_audit(structure, limits)
    report['pair_distance_audit'] = audit
    require_pair_distance_audit(audit)
    sulfur = audit['pairs'].get('S-S', {})
    if sulfur.get('minimum_distance_angstrom') is not None:
        names = ' / '.join(f"{a['residue_name']} {a['residue_number']} {a['atom_name']}" for a in sulfur['closest_atoms'])
        LOG.info('Closest S-S: %.3f A (%s)',sulfur['minimum_distance_angstrom'],names)


class ProfileToleranceAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        try:normalized=profile_tolerance_spec(value)
        except MastercapError as exc:raise argparse.ArgumentError(self,str(exc)) from exc
        setattr(namespace,self.dest,normalized)
        namespace.profile_tolerance_spec=value


def configure_tolerances(args,analysis,edges,settings=None,saved=None):
    if args.tolerances_from:
        settings=settings or profile_settings(analysis,edges,args.bin_width,args.radius_method,args.radius_percentiles)
        value,composition,digest=load_tolerances(args.tolerances_from,settings,analysis.original,LOG.warning)
        args.profile_tolerance=value;args.composition_tolerance=composition
        args.profile_tolerance_spec=','.join(f'{e}={v}' for e,v in value.items())
        args.calibration_file=str(args.tolerances_from);args.calibration_sha256=digest
        args.tolerance_source='calibration'
    else:
        explicit=args.profile_tolerance is not None or args.composition_tolerance is not None
        if saved is not None:
            if args.profile_tolerance is None:
                args.profile_tolerance=saved.get('profile_tolerance')
                args.profile_tolerance_spec=saved.get('profile_tolerance_spec',str(args.profile_tolerance) if args.profile_tolerance is not None else None)
            if args.composition_tolerance is None:args.composition_tolerance=saved.get('composition_tolerance')
            if not explicit:
                for key in ('calibration_file','calibration_sha256'):
                    if key in saved:setattr(args,key,saved[key])
        args.profile_tolerance=check_coverage(args.profile_tolerance,analysis.original,LOG.warning)
        args.tolerance_source='explicit_with_saved_defaults' if saved is not None and explicit else 'saved' if saved is not None else 'explicit'


class DetailedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show concrete defaults and preserve readable examples and explanations."""

    inferred_defaults = {
        "inner_radius": "inferred from the reference",
        "outer_radius": "inferred from the reference",
        "seed": "generate a seed and record it in JSON",
        "profile_tolerance": "unset (uncalibrated)",
        "composition_tolerance": "unset (uncalibrated)",
        "refine_iterations": "no additional iteration cap",
        "pair_limit_override": "S-S=steric",
    }

    def _get_help_string(self, action):
        if action.default is None:
            explanation = self.inferred_defaults.get(action.dest)
            if explanation:
                return f"{action.help} (default: {explanation})"
            return action.help
        return super()._get_help_string(action)


class CompleteHelpParser(argparse.ArgumentParser):
    """Top-level help includes real child parsers, so options cannot drift."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_parsers = []

    def format_help(self):
        text = super().format_help()
        for command in self.command_parsers:
            text += "\n"+"="*72+"\n"+command.format_help()
        return f"mastercap {__version__}\n\n"+text


COMMAND_EXAMPLES = {
    "analyze": """Examples:
  mastercap analyze --input ./1cwp_full.vdb.gz --output-prefix results/reference --csv --plot
  mastercap analyze --id 1cwp --output-prefix results/1cwp
  mastercap analyze --input reference.pdb --radius-method percentile --radius-percentiles 0.5 99.5 --output-prefix results/trimmed

Input and output:
  Supply an already expanded, complete protein capsid. No symmetry expansion,
  missing-atom reconstruction or hydrogen addition is performed. Occupancy is
  ignored; HETATM and nonprotein records are excluded; alternate positions fail.
  --input and --id are mutually exclusive. Downloads use VIPERdb with TLS checks
  by default; --no-check-certificate explicitly disables them for downloads.
  Pair limits default to S-S=steric: an element-derived nonbonded floor replaces sulfur's
  symmetry spacing statistic. Repeat --pair-limit-override PAIR=VALUE for a
  positive A value, steric, pooled or reference; S-S=reference restores old behavior.
  Writes PREFIX.json and, when requested, PREFIX_profiles.csv / .png; no PDB.
  Element-inference and other reference warnings stay in JSON, with one terminal
  summary (suppressed by --quiet).""",
    "generate": """Examples:
  mastercap generate --input ./1cwp_full.vdb.gz --output-prefix results/particle --seed 42 --allow-approximate --csv --plot
  mastercap generate --id 1cwp --output-prefix results/particle --seed 42 --allow-approximate
  mastercap generate --input reference.pdb --output-prefix results/particle --profile-tolerance C=0.02,N=0.04,O=0.04,S=0.15 --composition-tolerance 0
  mastercap generate --input reference.pdb --output-prefix results/particle --tolerances-from results/calib_calibration.json
  mastercap generate --input reference.pdb --inner-radius 100 --outer-radius 130 --mode residue --distribution regular --output-prefix results/resized --seed 42 --allow-approximate

Workflow and defaults:
  Automatically ANALYZE -> GENERATE -> WRITE/REREAD PDB -> VALIDATE.
  By default, keep reference radii and center, use independent atoms and random
  initialization, and smooth radial profiles with the provisional 1 A sigma.
  Radii bound atom centers, not the outer surfaces of their vdW spheres.
  The 300 s repair budget excludes preparation; authorized fallback shares the
  remaining repair time. Individual repair operations can overrun the deadline.
  Progress is printed every 5 iterations and at termination; --progress-interval
  changes this frequency. Stage messages precede input reading and analysis.
  After first feasible geometry, allow up to --refine-updates profile pushes
  (default 50). Rolled-back pushes count too. Repair steps are not pushes.
  --refine-iterations is an optional hard iteration cap, off by default.
  Either budget set to 0 restores first-feasible stopping. Time/max-iteration
  limits still apply. --selection-objective total ranks atom-weighted error
  first; worst-element restores the previous worst-element-first ranking.
  --refine-selection crossing (default) targets surplus-to-deficit adjacent-bin
  crossings, then drifts atoms toward useful edges. mismatch keeps rank matching.
  At most 0.2 of each element is selected per crossing/drift push (round down).
  Radial strength is fixed at 0.005 by default; --refine-strength-control gain
  follows primary profile gain, and acceptance restores the earlier feedback.
  --refine-min-step 0.002 A avoids sub-rounding requests; the max is 0.05 A.
  Checked consolidation builds 0.02 A slack for up to 10 updates. It stops when
  near-limit counts are <=0.001 times atom count, progress is <5%, or budget ends.
  Restore slack below 70% acceptance or every 10 pushes, for up to 3 checked
  updates, with at least 2 pushes between episodes. Final limits are unchanged.
  --refine-acceptance per-atom (default) checks a rounded trial and undoes only
  conflicting moves, or whole residues in rigid mode, preserving every old
  contact. Only acceptance strength control grows above 85% acceptance and shrinks
  below 50%; gain control shrinks after 2 non-improving pushes and grows on gain
  with >85% acceptance. Fixed control cannot hit a strength floor.
  rollback permits repairs, retaining recovered geometry only if its ranking
  improves; disaster defaults are a violating-pair count above 0.05 times the
  atom count, or 5000 components.
  Fourfold strength reduction without 25% damage reduction stops with
  refinement_blocked. Patience/strength floor stops use refinement_stalled.
  --neighbor-forces off (default) disables statistical neighbor contributions.
  scaled follows strength and atom selection, with
  an accumulated per-atom cap of half --refine-max-step; off disables these
  contributions; fixed uses coefficient 0.002 on all atoms without that cap.
  Both active modes guard targets inside the distance/contact window.
  S-S=steric derives a ratio from measured non-overridden pairs containing S;
  pooled retains the old global formula but all new overrides must clear the
  covalent floor (0.9 times the covalent-radius sum). Saved validation uses
  recorded numeric limits, never the current meaning of a mode name.
  S-S=steric is the default distance override; measured neighbor statistics
  remain unchanged and every applied limit is recorded. S-S=reference restores
  the old sulfur spacing limit. Optional --sulfur-residue-placement profile
  copies reference S shells by anchoring S-containing residues at initialization
  and using a weak repair-only spring; off preserves the old random draws.
  --consolidate-noncontact-slack on builds distance-side slack even for pairs
  that cannot overlap; off permits an exact v0.1.6 consolidation comparison.
  Before publication, a fresh independent connectivity audit checks the reread
  PDB and reports specific atoms on any failure; --allow-approximate cannot bypass it.

Acceptance:
  Always require radial bounds, center error <= 1 A, applicable minimum distances,
  internal residue connectivity and global connectivity by positive vdW overlap.
  Unset profile/composition tolerances produce an uncalibrated result; saving it
  requires --allow-approximate. With tolerances, each element is compared to its own value (or the * fallback).
  Profile error = sum(abs(generated density - target density) * shell volume)
                  / target element count.
  Composition tolerance is relative, with a minimum allowance of one atom.
  These quality tolerances have no automatic defaults. calibrate suggests values
  from explicit reference resampling; --tolerances-from applies them only when
  reference, model, bin width, smoothing and radii match.
  --allow-approximate NEVER permits violation of mandatory geometry/connectivity.

Output:
  Writes PREFIX.pdb and PREFIX.json after final written/reread validation;
  optional profiles use PREFIX_profiles.csv / .png. A failure writes a separate
  PREFIX.diagnostic.json when possible, without publishing an invalid PDB.
  One PDB uses TER-separated segments; serials/chains can repeat in large files.
  JSON atom order/provenance is authoritative. This is a geometric model, not a
  chemically valid protein or force-field-ready structure.""",
    "validate": """Examples:
  mastercap validate --input ./1cwp_full.vdb.gz --pdb results/particle.pdb --report results/particle.json --output-prefix results/validation
  mastercap validate --id 1cwp --pdb results/particle.pdb --report results/particle.json --output-prefix results/validation --csv --plot

Validation:
  The matching generation JSON and original reference content are required.
  Recompute analysis, verify reference/PDB hashes and atom provenance, and check
  rounded-coordinate geometry, connectivity, composition and radial profiles.
  Model, radii, smoothing and mode come from --report. Quality tolerances use
  saved values unless overridden explicitly or via --tolerances-from.
  Plain and gzip references are interchangeable if decompressed content matches.
  Writes PREFIX.json and optional PREFIX_profiles.csv / .png; never generates a PDB.
  An uncalibrated status means mandatory checks passed but quality tolerances
  were not supplied; use explicit tolerances or --tolerances-from to reassess profile quality.
  Plot smoothing follows the effective setting saved in --report.""",
}

COMMAND_EXAMPLES['calibrate']="""Examples:
  mastercap calibrate --input ./1cwp_full.vdb --bin-width 4 --no-smoothing --replicates 50 --seed 42 --output-prefix results/calib_1cwp --csv --plot
  mastercap calibrate --input ./1cwp_full.vdb --bin-width 4 --no-smoothing --generated results/particle08.pdb --compare other_ccmv.pdb --output-prefix results/calib_1cwp --csv --plot

Counting samples finite bin populations. Orientation rotates every residue about
its OWN centroid without moving that centroid. Placement additionally shuffles
centroid radii. These are resampling baselines, not mandatory geometry checks.
Suggestions are never applied automatically. Use --tolerances-from explicitly;
reference, model, bins, smoothing and radii must match. Outputs use the separate
PREFIX_calibration.json/.csv/.png names. Calibration never generates a PDB."""

SMOOTHING_HELP = """
Smoothing and radial resolution:
  --smoothing-sigma is Gaussian width in A, not a number of bins. Higher sigma
  smooths more, broadens peaks and suppresses narrow fluctuations; counts are
  conserved by shell-volume normalization. Sigma 0 also disables smoothing.
  --no-smoothing overrides sigma and targets the original binned mapped profile.
  No 'Smoothed target' line is plotted when smoothing is disabled. Binning still
  applies: --bin-width 4 is coarser than 1 A, even without Gaussian smoothing.
  Generation seeks the target under geometric/connectivity constraints; neither
  setting guarantees an exact match or establishes scientifically valid tolerances.
  Final independent connectivity and inter-residue pair-distance audits run on
  the reread PDB. Sub-covalent pairs fail with atom identifiers (expected peptide
  links are excluded from that floor check). Compare errors_at_push at equal
  push counts; use --time-limit 1800 for benchmark runs. Time-limited results
  depend on machine speed even with a fixed seed.
"""

EXIT_HELP = """
Exit codes:
  0    Analysis completed, or generated/validated result passes supplied tolerances.
  1    Input, search, validation or output failure.
  2    Approximate/uncalibrated result saved or verified; also CLI usage errors.
  130  Interrupted.
"""


def finite_nonnegative(value):
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def finite_positive(value):
    number = finite_nonnegative(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def positive_fraction(value):
    number = finite_positive(value)
    if number > 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return number


def unit_fraction(value):
    number = finite_nonnegative(value)
    if number > 1:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return number


def parser():
    result = CompleteHelpParser(
        prog="mastercap", formatter_class=DetailedHelpFormatter,
        description="Reference-driven spherical geometric capsid particles. All distances are in angstroms.\n"
                    "This help includes ALL command options, defaults and examples below.\n"
                    "Use mastercap COMMAND -h for a shorter, command-specific reference.",
        epilog="Choose analyze, generate, validate or calibrate. generate includes analysis and final validation.\n"
               "Concrete defaults are shown beside options; inferred/unset defaults are explained.\n"
               "Run from an environment where mastercap and its dependencies are installed.\n"+EXIT_HELP)
    result.add_argument("--version", action="version", version=__version__, help="Print the installed mastercap version and exit")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("analyze", "generate", "validate", "calibrate"):
        description = {"analyze": "Analyze an already expanded protein capsid; no coordinates are generated.",
                       "generate": "Automatically ANALYZE reference -> GENERATE -> WRITE AND REREAD PDB -> VALIDATE. No separate analyze/validate commands are needed.",
                       "validate": "Independently reanalyze the reference and validate an existing PDB using its mandatory JSON report.",
                       "calibrate": "Measure reference profile error levels (counting, orientation, placement floors), optionally compare other structures, and suggest per-element tolerances. No particle is generated."}[name]
        command = sub.add_parser(name, help=description, description=description,
                                 formatter_class=DetailedHelpFormatter,
                                 epilog=COMMAND_EXAMPLES[name]+(SMOOTHING_HELP if name != "validate" else "")+"\n"+EXIT_HELP)
        result.command_parsers.append(command)
        if name!="calibrate":
            command.add_argument("--pair-limit-override", action=PairLimitAction, metavar="PAIR=VALUE", help="Repeatable positive A/steric/pooled/reference distance override; steric uses element-derived measured ratios; all new overrides obey the covalent floor; default S-S=steric. Use S-S=reference for old spacing. Validation applies saved numeric override values")
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--input", type=Path, help="Local expanded .pdb/.vdb, optionally gzip compressed; required alternative to --id")
        source.add_argument("--id", help="Four-character VIPERdb identifier, e.g. 1cwp; required alternative to --input")
        model_help = "Reference MODEL number to read" if name != "validate" else "Ignored here: the reference MODEL number saved in --report is used"
        command.add_argument("--model", type=positive_integer, default=1, help=model_help)
        command.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"mastercap", help="VIPERdb cache directory (honors XDG_CACHE_HOME)")
        command.add_argument("--refresh", action="store_true", help="Download a fresh VIPERdb cache entry instead of reusing the existing one; applies to --id")
        command.add_argument("--no-check-certificate", action="store_true", help="Explicitly skip TLS certificate verification for VIPERdb downloads only (--id); default verification is enabled; does not bypass HTTP/network-policy errors")
        command.add_argument("--output-prefix", type=Path, required=name!="calibrate", default=Path("results/calibration") if name=="calibrate" else None, help="Calibration output basename; _calibration suffix is appended and parent directories created" if name=="calibrate" else "Required output basename, e.g. results/particle; extensions are appended and parent directories created")
        command.add_argument("--csv", action="store_true", help="Write PREFIX_calibration.csv" if name=="calibrate" else "Write radial density profiles to PREFIX_profiles.csv")
        command.add_argument("--plot", action="store_true", help="Write PREFIX_calibration.png" if name=="calibrate" else "Write radial density comparison plots to PREFIX_profiles.png")
        command.add_argument("--overwrite", action="store_true", help="Explicitly permit replacing existing output files")
        verbosity = command.add_mutually_exclusive_group()
        verbosity.add_argument("--quiet", action="store_true", help="Suppress progress and warning summaries; errors are still printed")
        verbosity.add_argument("--verbose", action="store_true", help="Enable detailed runtime logging; per-record reference warnings remain only in JSON")
        if name != "validate":
            command.add_argument("--bin-width", type=finite_positive, default=1.0, help="Requested maximum radial shell-bin width in A; must be positive")
            command.add_argument("--smoothing-sigma", type=finite_nonnegative, default=1.0, help="Gaussian radial smoothing sigma in A; larger values smooth more and broaden peaks; 0 disables smoothing; overridden by --no-smoothing")
            command.add_argument("--no-smoothing", action="store_true", help="Disable Gaussian smoothing and use the original binned reference profile")
            command.add_argument("--radius-method", choices=("extrema", "percentile"), default="extrema", help="Infer reference bounds from minimum/maximum radii or the selected radial percentiles")
            command.add_argument("--radius-percentiles", nargs=2, type=float, default=[0.5, 99.5], metavar=("LOW", "HIGH"), help="Lower/upper radial percentiles, 0 <= LOW < HIGH <= 100; only used with --radius-method percentile; excluded atoms are reported")
            if name!="calibrate":
                command.add_argument("--distance-percentile", type=finite_nonnegative, default=1.0, help="Nearest-neighbor percentile in [0, 100] for element-pair minimum distances; provisional")
                command.add_argument("--min-pair-samples", type=positive_integer, default=100, help="Minimum observations for a pair-specific distance percentile; fewer use grouped normalized reference distances; provisional")
        if name in ("generate","calibrate"):
            command.add_argument("--inner-radius", type=finite_nonnegative, help="Inner atom-center radius in A; require 0 <= inner < outer")
            command.add_argument("--outer-radius", type=finite_positive, help="Outer atom-center radius in A; overrides do not change local density targets")
        if name == "generate":
            command.add_argument("--center", choices=("reference", "origin"), default="reference", help="Target center: reference coordinate mean or origin (0, 0, 0)")
            command.add_argument("--mode", choices=("independent", "residue"), default="independent", help="Move individual atoms (independent) or preserve observed residue geometry with rigid moves (residue)")
            command.add_argument("--sulfur-residue-placement", choices=("off", "profile", "reference"), default="off", help="Optionally anchor the first S per S-containing residue: profile uses target quantiles (regular: bin midpoints); reference uses its own reference atom radius without quantization, requires radii within target bounds. Spacing swaps; repair-only dead-zone spring; refinement preserves assigned bins")
            command.add_argument("--sulfur-anchor-tolerance", type=finite_nonnegative, default=1.5, help="Repair-only S radial dead-zone half-width in A, clipped inside its assigned bin; 0 holds exact radius. Refinement forbids leaving the assigned bin")
            command.add_argument("--no-sulfur-anchor-spring", action="store_true", help=argparse.SUPPRESS)
            command.add_argument("--distribution", choices=("random", "regular"), default="random", help="Random isotropic or regular spherical-layer initialization; repair may move initial positions")
            command.add_argument("--seed", type=int, help="Nonnegative random seed for reproducible generation")
            command.add_argument("--max-iterations", type=int, default=1000, help="Maximum repair steps per attempt; nonnegative, provisional")
            command.add_argument("--time-limit", type=finite_positive, default=300.0, help="Cooperative repair-search budget in seconds, shared with fallback; excludes input, analysis, inventory and initialization")
            command.add_argument("--progress-interval", type=positive_integer, default=5, help="Print search progress every this many iterations, plus initial/final progress; --quiet suppresses progress")
            command.add_argument("--refine-iterations", type=nonnegative_integer, default=None, help="Optional cap on all iterations after first feasible geometry, including repair; 0 stops at first feasible geometry")
            command.add_argument("--selection-objective", choices=("total", "worst-element"), default="total", help="Ranks feasible candidates: total = atom-weighted total profile error (default); worst-element = minimize the worst per-element error first (previous behaviour)")
            command.add_argument("--refine-updates", type=nonnegative_integer, default=50, help="Profile-push budget, counting rolled-back pushes too; 0 stops at first feasible geometry")
            command.add_argument("--refine-patience", type=nonnegative_integer, default=10, help="Stop after this many pushes without a significant relative improvement in the primary selection error (0.1 percent); 0 disables patience stopping")
            command.add_argument("--refine-strength", type=finite_positive, default=0.005, help="Radial strength, fixed by default; gain/acceptance controls adapt up to 4x and stall below 1/16; rollback retains damage feedback")
            command.add_argument("--refine-strength-control", choices=("fixed", "gain", "acceptance"), default="fixed", help="Per-atom strength: fixed keeps the starting value; gain follows primary profile improvement; acceptance restores the previous acceptance feedback. Rollback mode retains damage feedback")
            command.add_argument("--refine-min-step", type=finite_nonnegative, default=0.002, help="Minimum nonzero requested radial move in A when the target mismatch is at least this large; must not exceed --refine-max-step; 0 disables")
            command.add_argument("--refine-selection", choices=("crossing", "mismatch"), default="crossing", help="Crossing targets useful adjacent-bin crossings, then drifts surplus atoms toward edges; skips previous rejected atoms. Mismatch keeps v0.1.5 rank-matched selection")
            command.add_argument("--refine-max-step", type=finite_positive, default=0.05, help="Maximum absolute requested radial profile move per selected atom in A; excludes other forces and rigid projection")
            command.add_argument("--refine-fraction", type=positive_fraction, default=0.2, help="Crossing/drift atom budget per element, rounding down; mismatch rounds up and prefers largest radial mismatches; must be in (0, 1]")
            command.add_argument("--refine-recovery-limit", type=nonnegative_integer, default=10, help="Maximum repair updates after a damaging push before restoring its snapshot; 0 permits no repair updates")
            command.add_argument("--neighbor-forces", choices=("scaled", "off", "fixed"), default="off", help="Refinement nearest-neighbor forces: off disables them (default); scaled follows radial strength, selection and a half-step per-atom cap; fixed keeps unscaled 0.002/all-atom behavior. Targets are guarded in both active modes")
            command.add_argument("--refine-slack", type=finite_nonnegative, default=0.02, help="Target distance/contact slack in A during consolidation and neighbor targeting; final validation limits are unchanged")
            command.add_argument("--consolidate-iterations", type=nonnegative_integer, default=10, help="Maximum checked consolidation updates before refinement; 0 disables initial consolidation; restarts only as a safety net")
            command.add_argument("--consolidate-target", type=unit_fraction, default=0.001, help="Stop checked consolidation when each near-limit pair/edge count is at most this fraction of atom count, or improvement falls below 5 percent; at least one update")
            command.add_argument("--consolidate-noncontact-slack", choices=("on", "off"), default="on", help="Build full distance-side slack for pairs whose minimum separation precludes vdW contact; off restores v0.1.6 consolidation for comparison")
            command.add_argument("--reconsolidate-below", type=unit_fraction, default=0.7, help="Restore slack after a push with lower acceptance than this fraction; 0 disables; at least 2 pushes between episodes")
            command.add_argument("--reconsolidate-every", type=nonnegative_integer, default=10, help="Restore slack after this many pushes since the last episode; 0 disables periodic restoration")
            command.add_argument("--reconsolidate-iterations", type=nonnegative_integer, default=3, help="Maximum checked updates per slack-restoration episode; these consume time/iterations but not pushes; 0 disables episodes")
            command.add_argument("--refine-acceptance", choices=("per-atom", "rollback"), default="per-atom", help="Per-atom rejects conflicting trial moves before commit (whole residues in rigid mode); rollback permits repair and snapshot recovery")
            command.add_argument("--rollback-violation-fraction", type=unit_fraction, default=0.05, help="Immediate rollback disaster threshold as a fraction of atom count, in [0, 1]; smaller damage is allowed to repair")
            command.add_argument("--rollback-component-limit", type=positive_integer, default=5000, help="Immediate rollback when component count reaches this threshold")
            command.add_argument("--overlap-margin", type=finite_nonnegative, default=0.0, help="Extra required vdW overlap in A; must be nonnegative and tangency never qualifies")
            command.add_argument("--allow-approximate", action="store_true", help="Permit soft-tolerance deviations or uncalibrated results; mandatory constraints may NEVER be relaxed")
            command.add_argument("--allow-independent-fallback", action="store_true", help="Explicitly allow retrying a failed rigid-residue search in independent mode")
        if name in ('generate','validate'):
            command.set_defaults(profile_tolerance_spec=None)
            command.add_argument('--profile-tolerance',action=ProfileToleranceAction,help='Per-element normalized profile error: NUMBER, C=0.02,N=0.04,..., or C=0.02,*=0.05; each target element needs its own value or *; no automatic default')
            command.add_argument('--composition-tolerance',type=finite_nonnegative,help='Relative per-element count error limit, with a minimum one-atom allowance')
            command.add_argument('--tolerances-from',type=Path,help='Use suggested tolerances from a calibration JSON with matching reference/model/bin/smoothing/radii; incompatible with explicit tolerance options')
        if name=='calibrate':
            command.add_argument('--replicates',type=positive_integer,default=50,help='Independent replicates per counting, orientation and placement level')
            command.add_argument('--seed',type=nonnegative_integer,default=42,help='Seed for reproducible calibration resampling')
            command.add_argument('--compare',type=Path,action='append',default=[],help='Repeatable experimental comparison PDB; its own center, reference edges; report raw and count-normalized shape error')
            command.add_argument('--generated',type=Path,action='append',default=[],help='Repeatable generated PDB; check sibling report settings and use its saved center when available')
            command.add_argument('--floor-multiple',type=finite_positive,default=2.0,help='Multiply the chosen floor statistic to suggest tolerances; never applied automatically')
            command.add_argument('--floor-basis',choices=('orientation','counting'),default='orientation',help='Reference error level used for suggestions')
            command.add_argument('--floor-statistic',choices=('p95','mean'),default='p95',help='Chosen floor statistic; round suggestions upward to three significant figures')
        if name == "validate":
            command.add_argument("--pdb", type=Path, required=True, help="Required generated particle PDB to revalidate")
            command.add_argument("--report", type=Path, required=True, help="Required JSON report from the matching successful generation")
    return result


def reference(args):
    if args.input:
        LOG.info("Using local reference: %s", args.input)
        if args.no_check_certificate:
            LOG.info("--no-check-certificate has no effect for local input; no TLS download is performed")
        return args.input, {}
    LOG.info("Resolving expanded VIPERdb reference: %s", args.id)
    return download(args.id, args.cache_dir, args.refresh, no_check_certificate=args.no_check_certificate)


def analysis_options(args):
    if args.command=="calibrate":
        return {"width":args.bin_width,"sigma":0.0 if args.no_smoothing else args.smoothing_sigma,
                "radius_method":args.radius_method,"percentiles":args.radius_percentiles,"compute_distances":False}
    return {"width": args.bin_width, "sigma": 0.0 if args.no_smoothing else args.smoothing_sigma,
            "radius_method": args.radius_method, "percentiles": args.radius_percentiles,
            "distance_percentile": args.distance_percentile, "min_samples": args.min_pair_samples,
            "pair_limit_overrides": resolve_limit_overrides(args.pair_limit_override)}


def artifacts(prefix, csv=False, plot=False, pdb=False):
    names = {"json": Path(str(prefix)+".json")}
    if pdb:
        names["pdb"] = Path(str(prefix)+".pdb")
    if csv:
        names["csv"] = Path(str(prefix)+"_profiles.csv")
    if plot:
        names["plot"] = Path(str(prefix)+"_profiles.png")
    return names


def configuration_value(value):
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {k:configuration_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [configuration_value(v) for v in value]
    return value


def base_report(args, started):
    return {"schema_version": 1, "mastercap_version": __version__,
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
            "command": args.command, "status": "failed", "elapsed_seconds": time.monotonic()-started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "configuration": configuration_value(vars(args))}


def extra_profiles(files, names, edges, original, smoothed, generated=None, *, smoothing_enabled=True):
    if "csv" in names:
        files[names["csv"]] = profile_csv(edges, original, smoothed, generated)
    if "plot" in names:
        LOG.info("Rendering radial profile PNG")
        files[names["plot"]] = profile_png(edges, original, smoothed, generated, smoothing_enabled=smoothing_enabled)


def diagnostic(args, report, protected):
    protected=list(protected)+([args.tolerances_from] if getattr(args,'tolerances_from',None) else [])
    # Never replace the JSON belonging to a previously successful PDB.
    path = Path(str(args.output_prefix)+".diagnostic.json")
    publish({path: encode_json(report)}, args.overwrite, protected)
    LOG.error("No new PDB written. Diagnostic: %s", path)


def log_warning_summary(warnings):
    """Keep per-record reference diagnostics in JSON, even in verbose mode."""
    if warnings:
        inferred = sum(warning.startswith("Inferred ") for warning in warnings)
        LOG.warning("Reference analysis: %d warnings (%d element inferences); details only in JSON.",
                    len(warnings), inferred)


def generate_command(args, analysis, source, acquisition, started, names, preparation_timings=None):
    if args.seed is None:
        args.seed = secrets.randbits(63)
    if (args.seed < 0 or args.max_iterations < 0 or (args.refine_iterations is not None and args.refine_iterations < 0)
            or args.refine_updates < 0 or args.refine_patience < 0 or args.refine_recovery_limit < 0
            or args.consolidate_iterations < 0 or args.rollback_component_limit < 1
            or args.reconsolidate_every < 0 or args.reconsolidate_iterations < 0):
        raise MastercapError("Seed, iteration/update/patience/recovery limits must be nonnegative")
    if (not np.isfinite([args.refine_strength, args.refine_max_step, args.refine_fraction, args.refine_slack, args.rollback_violation_fraction, args.refine_min_step, args.consolidate_target, args.reconsolidate_below]).all()
            or args.refine_strength <= 0 or args.refine_strength > np.finfo(float).max/4
            or args.refine_max_step <= 0 or not 0 < args.refine_fraction <= 1 or args.refine_slack < 0
            or not 0 <= args.rollback_violation_fraction <= 1 or not 0 <= args.refine_min_step <= args.refine_max_step
            or not 0 <= args.consolidate_target <= 1 or not 0 <= args.reconsolidate_below <= 1):
        raise MastercapError("Refinement strength/max step must be finite and positive; fraction must be in (0, 1]")
    if args.neighbor_forces not in ("scaled", "off", "fixed"):
        raise MastercapError("Neighbor forces must be scaled, off or fixed")
    if args.refine_acceptance not in ("per-atom", "rollback"):
        raise MastercapError("Refinement acceptance must be per-atom or rollback")
    if args.refine_strength_control not in ("fixed", "gain", "acceptance"):
        raise MastercapError("Strength control must be fixed, gain or acceptance")
    if args.refine_selection not in ("crossing", "mismatch"):
        raise MastercapError("Refinement selection must be crossing or mismatch")
    rng = np.random.default_rng(args.seed)
    edges, profiles = target_profiles(analysis, args.inner_radius, args.outer_radius)
    center = analysis.center if args.center == "reference" else np.zeros(3)
    report = base_report(args, started)
    timings = dict(preparation_timings or {})
    timings.update({"inventory": 0.0, "initialization": 0.0, "search": 0.0, "export_validation": 0.0})
    report["timings_seconds"] = timings
    report.update({"analysis_options": analysis_options(args), "analysis": analysis.summary(),
                   "acquisition": acquisition, "target": {"center": center.tolist(), "edges": edges.tolist(),
                   "profiles": {e: p.tolist() for e, p in profiles.items()}},
                   "calibration": ("reference_calibration_supplied" if args.tolerances_from else
                                   "manual_tolerances" if args.profile_tolerance is not None or args.composition_tolerance is not None else "not_supplied"), "pdb_convention": "TER segments; serials and chains may repeat"})
    LOG.info("Selected reference: %d atoms; target radii %.3f..%.3f A; seed %d", len(analysis.structure.atoms), edges[0], edges[-1], args.seed)
    export_started = None
    try:
        LOG.info("Selecting integer residue inventory and element counts")
        inventory_started = time.monotonic()
        try:
            inventory, trace = select_inventory(analysis, edges, profiles, rng, time_limit=15.0)
        finally:
            timings["inventory"] = time.monotonic()-inventory_started
        report["residue_inventory"] = trace
        def progress(iteration, geometry, metrics):
            worst = max(m["profile_error"] or 0 for m in metrics["elements"].values())
            suffix = ""
            if geometry.get("final_search_state"):
                suffix = (" (final search state; not the exported candidate)" if geometry.get("export_candidate_available")
                          else " (final search state; no feasible candidate to export)")
            LOG.info("Iteration %d: components=%d isolated=%d distance violations=%d radial violations=%d "
                     "disconnected residues=%d center error=%.3f A profile error=%.4f max element error=%.4f%s", iteration,
                     geometry["global_components"], geometry["isolated_atoms"], geometry["distance_violations"],
                     geometry["outside_radial_bounds"], len(geometry["disconnected_residues"]),
                     geometry["center_error_angstrom"], metrics["total_profile_error"], worst, suffix)
        mode = args.mode
        def attempt(search_mode):
            attempt_started = time.monotonic()
            try:
                result, details = search(inventory.copy(), analysis, edges, profiles, center, rng,
                                         mode=search_mode, distribution=args.distribution,
                                         max_iterations=args.max_iterations,
                                         time_limit=args.time_limit-timings["search"],
                                         margin=args.overlap_margin, profile_tolerance=args.profile_tolerance,
                                         composition_tolerance=args.composition_tolerance, progress=progress,
                                         progress_interval=args.progress_interval, refine_iterations=args.refine_iterations,
                                         selection_objective=args.selection_objective, refine_updates=args.refine_updates,
                                         refine_patience=args.refine_patience, refine_strength=args.refine_strength,
                                         refine_max_step=args.refine_max_step, refine_fraction=args.refine_fraction,
                                         refine_recovery_limit=args.refine_recovery_limit,
                                         neighbor_forces=args.neighbor_forces,
                                         refine_slack=args.refine_slack, consolidate_iterations=args.consolidate_iterations,
                                         refine_acceptance=args.refine_acceptance,
                                         rollback_violation_fraction=args.rollback_violation_fraction,
                                         rollback_component_limit=args.rollback_component_limit,
                                         refine_strength_control=args.refine_strength_control,
                                         refine_min_step=args.refine_min_step,
                                         refine_selection=args.refine_selection,
                                         sulfur_residue_placement=args.sulfur_residue_placement,
                                         sulfur_anchor_tolerance=args.sulfur_anchor_tolerance,
                                         sulfur_anchor_spring=not args.no_sulfur_anchor_spring,
                                         consolidate_noncontact_slack=args.consolidate_noncontact_slack,
                                         consolidate_target=args.consolidate_target,
                                         reconsolidate_below=args.reconsolidate_below,
                                         reconsolidate_every=args.reconsolidate_every,
                                         reconsolidate_iterations=args.reconsolidate_iterations,
                                         stage_progress=LOG.info)
            except MastercapError:
                # Search raises only for invalid settings/templates during prep.
                timings["initialization"] += time.monotonic()-attempt_started
                raise
            timings["initialization"] += details["preparation_seconds"]
            timings["search"] += details["elapsed_seconds"]
            return result, details

        try:
            particle, search_report = attempt(mode)
        except MastercapError as exc:
            if not args.allow_independent_fallback or mode != "residue":
                raise
            particle, search_report = None, {"error": str(exc)}
        if particle is None and args.allow_independent_fallback and mode == "residue" and timings["search"] < args.time_limit:
            report["rigid_search"] = search_report
            mode = "independent"
            LOG.warning("Explicitly authorized fallback to independent mode")
            particle, search_report = attempt(mode)
        report["effective_mode"] = mode
        report["search"] = search_report
        timings["preparation"] = time.monotonic()-started-timings["search"]
        LOG.info("Preparation %.3f s; repair search %.3f s / %.3f s; termination=%s",
                 timings["preparation"], timings["search"], args.time_limit, search_report.get("termination", "preparation_error"))
        recoveries = [p["recovery_iterations"] for p in search_report.get("push_history", []) if p.get("recovery_iterations") is not None]
        LOG.info("Feasible iterations=%d; profile pushes=%d; acceptance=%s; committed pushes=%d; mean acceptance=%s; "
                 "rollbacks=%d; restored snapshots=%d; recoveries improved/worse=%d/%d; stop limit=%s; stall cause=%s; median repair iterations=%s",
                 len(search_report.get("feasible_iterations", [])), len(search_report.get("push_history", [])),
                 search_report.get("acceptance_mode"), search_report.get("committed_pushes", 0),
                 (f"{search_report['mean_acceptance_fraction']:.1%}" if search_report.get("mean_acceptance_fraction") is not None else "not applicable"),
                 search_report.get("rollback_count", 0),
                 len(search_report.get("restored_snapshot_iterations", [])),
                 search_report.get("recoveries_improved", 0), search_report.get("recoveries_worse", 0),
                 search_report.get("refinement_stop_limit") or search_report.get("termination"),
                 search_report.get("stall_cause") or "none",
                 f"{np.median(recoveries):g}" if search_report.get("repair_attempts", 0) else "0 attempted")
        if particle is None:
            raise MastercapError(f"Search ended ({search_report.get('termination', 'preparation_error')}) without a candidate satisfying every mandatory constraint")
        if "S" in profiles:
            LOG.info("Top S too-close rejection pairs: %s", ", ".join(
                f"{entry['pair']}={entry['count']}" for entry in search_report.get("sulfur_top_rejection_pairs", [])) or "none")
        LOG.info("Selected feasible candidate from iteration %d; preparing PDB export", search_report["candidate_iteration"])
        export_started = time.monotonic()
        text, mapping = pdb_text(particle)
        with tempfile.TemporaryDirectory(prefix="mastercap-validate-") as directory:
            temporary = Path(directory)/"particle.pdb"
            temporary.write_text(text)
            LOG.info("Rereading written PDB and independently validating geometry")
            reread = read_pdb(temporary, generated=True)
        if len(reread.atoms) != len(particle.atoms):
            raise MastercapError("PDB reread changed the atom count")
        # Use actual PDB groups to check serialization did not merge residues.
        geometry = validate_geometry(reread, center, edges, analysis.limits, args.overlap_margin, mode)
        final_connectivity_check(reread, args.overlap_margin, geometry, report)
        final_pair_distance_check(reread, analysis.limits, report)
        metrics = profile_metrics(reread, center, edges, profiles, analysis.original)
        status = acceptance(geometry, metrics, args.profile_tolerance, args.composition_tolerance)
        report.update({"geometry": geometry, "metrics": metrics, "status": status})
        if status == "failed":
            raise MastercapError("Written-and-reread PDB violates mandatory constraints")
        if status != "valid" and not args.allow_approximate:
            raise MastercapError(f"Candidate is {status}; provide tolerances or explicitly use --allow-approximate")
        LOG.info("Final normalized profile errors: %s", ", ".join(
            f"{element}={values['profile_error']:.4f}" if values["profile_error"] is not None else f"{element}=n/a"
            for element, values in metrics["elements"].items()))
        particle.xyz = reread.xyz
        LOG.info("Computing final neighbor statistics and residue provenance")
        _, generated_neighbors, _ = distance_statistics(particle, args.distance_percentile, args.min_pair_samples)
        neighbor_comparison = {}
        for pair, observed in generated_neighbors.items():
            baseline = analysis.neighbor_samples[pair]
            neighbor_comparison[pair] = {
                "reference": baseline, "generated": observed,
                "percentile_differences_angstrom": {
                    p: observed["percentiles"][p]-baseline["percentiles"][p]
                    for p in observed["percentiles"] if p in baseline["percentiles"]}}
        report.update({"atom_mapping": mapping, "residue_transforms": residue_transforms(particle, analysis.structure),
                       "neighbor_distance_comparison": neighbor_comparison,
                       "pdb_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "elapsed_seconds": time.monotonic()-started,
                       "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
        files = {names["pdb"]: text}
        extra_profiles(files, names, edges, analysis.original, profiles, metrics["generated_profiles"],
                       smoothing_enabled=analysis.smoothing_sigma > 0)
        timings["export_validation"] = time.monotonic()-export_started
        report["elapsed_seconds"] = time.monotonic()-started
        LOG.info("Serializing JSON report and publishing output files")
        files[names["json"]] = encode_json(report)
        publish(files, args.overwrite, [source]+([args.tolerances_from] if getattr(args,'tolerances_from',None) else []))
        LOG.info("%s: %s (%d atoms)", status, names["pdb"], len(particle.atoms))
        return 0 if status == "valid" else 2
    except (MastercapError, KeyboardInterrupt) as exc:
        if export_started is not None:
            timings["export_validation"] = time.monotonic()-export_started
        timings.setdefault("preparation", time.monotonic()-started-timings["search"])
        report.update({"status": "failed", "error": str(exc) or "Interrupted by user",
                       "elapsed_seconds": time.monotonic()-started})
        LOG.error("%s", report["error"])
        diagnostic(args, report, [source])
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


def validate_command(args, source, started, names):
    LOG.info("Loading generation report: %s", args.report)
    saved = json.loads(args.report.read_text())
    if saved.get("schema_version") != 1 or "pdb_sha256" not in saved:
        raise MastercapError("Report is not a supported successful generation report")
    LOG.info("Reading reference coordinates for validation")
    options = dict(saved["analysis_options"])
    if "distance_limit_overrides" not in saved["analysis"]:
        options["pair_limit_overrides"] = ["S-S=reference"]
        LOG.info("Legacy report has no distance-limit override record; using reference limits")
    else:
        options["saved_limit_overrides"] = saved['analysis']['distance_limit_overrides']
        for entry in options['saved_limit_overrides']:
            if entry['mode'] == 'steric' and 'ratio' not in entry:
                LOG.warning('Saved %s override %.3f A predates the covalent floor; structure may contain sub-contact %s pairs',
                            entry['pair'],entry['new_value'],entry['pair'])
    analysis = analyze(read_pdb(source, saved["analysis"]["reference"]["model"]), **options, progress=LOG.info)
    if analysis.structure.provenance["sha256"] != saved["analysis"]["reference"]["sha256"]:
        raise MastercapError("Reference SHA-256 does not match the generation report")
    log_warning_summary(analysis.warnings)
    actual_hash = hashlib.sha256(args.pdb.read_bytes()).hexdigest()
    if actual_hash != saved["pdb_sha256"]:
        raise MastercapError("Generated PDB SHA-256 does not match its report")
    LOG.info("Reading generated PDB and checking provenance: %s", args.pdb)
    particle = read_pdb(args.pdb, generated=True)
    config = saved["configuration"].copy()
    edges, profiles = target_profiles(analysis, config["inner_radius"], config["outer_radius"])
    configure_tolerances(args,analysis,edges,settings_from_generation(saved),config)
    config['profile_tolerance']=args.profile_tolerance;config['composition_tolerance']=args.composition_tolerance
    center = analysis.center if config["center"] == "reference" else np.zeros(3)
    mapping = saved["atom_mapping"]
    if len(mapping) != len(particle.atoms):
        raise MastercapError("Atom mapping length does not match PDB")
    # Validate identity and membership by file order, never repeated PDB serials.
    for i, (atom, item) in enumerate(zip(particle.atoms, mapping)):
        if item["id"] != i or item["synthetic_residue"] != atom.residue:
            raise MastercapError("PDB residue grouping does not match report")
        origin = analysis.structure.atoms[item["origin"]["reference_atom_index"]]
        if (atom.name, atom.element, atom.resname) != (origin.name, origin.element, origin.resname):
            raise MastercapError("PDB atom identity does not match reference provenance")
    LOG.info("Validating geometry, connectivity and radial profiles")
    geometry = validate_geometry(particle, center, edges, analysis.limits, config["overlap_margin"], saved["effective_mode"])
    audit_report = {}
    try:
        final_connectivity_check(particle, config["overlap_margin"], geometry, audit_report)
        final_pair_distance_check(particle, analysis.limits, audit_report)
    except MastercapError as exc:
        failed = base_report(args, started)
        failed.update(status='failed', error=str(exc), geometry=geometry, **audit_report,
                      validated_pdb=str(args.pdb), reference=analysis.structure.provenance,
                      warnings=analysis.warnings,
                      distance_limits={'-'.join(pair):limit for pair,limit in analysis.limits.items()},
                      distance_limit_overrides=analysis.distance_limit_overrides)
        path = Path(str(args.output_prefix)+'.diagnostic.json')
        publish({path:encode_json(failed)},args.overwrite,[source,args.pdb,args.report])
        LOG.error('Validation failed; audit diagnostic: %s',path)
        raise
    metrics = profile_metrics(particle, center, edges, profiles, analysis.original)
    status = acceptance(geometry, metrics, config["profile_tolerance"], config["composition_tolerance"])
    report = base_report(args, started)
    report.update({"status": status, "geometry": geometry, "metrics": metrics, **audit_report,
                   "validated_pdb": str(args.pdb), "reference": analysis.structure.provenance,
                   "warnings": analysis.warnings, "distance_limit_overrides": analysis.distance_limit_overrides,
                   "distance_limits": {"-".join(k): v for k, v in analysis.limits.items()}})
    files = {names["json"]: encode_json(report)}
    extra_profiles(files, names, edges, analysis.original, profiles, metrics["generated_profiles"],
                   smoothing_enabled=analysis.smoothing_sigma > 0)
    LOG.info("Publishing validation report: %s", names["json"])
    publish(files, args.overwrite, [source, args.pdb, args.report]+([args.tolerances_from] if args.tolerances_from else []))
    LOG.info("Validation status: %s", status)
    return 1 if status == "failed" else 0 if status == "valid" else 2


def run(args):
    started = time.monotonic()
    LOG.info("Starting %s; output prefix: %s", args.command, args.output_prefix)
    try:
        source, acquisition = reference(args)
    except (MastercapError, OSError, EOFError) as exc:
        if args.command == "generate":
            report = base_report(args, started)
            report["error"] = str(exc)
            diagnostic(args, report, [args.input] if args.input else [])
            return 1
        raise
    preparation_timings = {"reference": time.monotonic()-started}
    names = artifacts(args.output_prefix, args.csv, args.plot, args.command == "generate")
    if args.command=='calibrate':
        names={key:Path(str(args.output_prefix)+'_calibration.'+ext) for key,ext in [('json','json')]+([('csv','csv')] if args.csv else [])+([('plot','png')] if args.plot else [])}
        sibling=Path(str(args.output_prefix)+'.json')
        if sibling.exists():
            try:prior=json.loads(sibling.read_text())
            except ValueError:prior={}
            if isinstance(prior,dict) and (prior.get('command')=='generate' or 'pdb_sha256' in prior):
                raise MastercapError('Calibration prefix belongs to a generation report; choose a different prefix')
    protected = [source]
    if args.command=='calibrate':
        protected.extend(args.compare+args.generated)
        protected.extend(Path(str(p)[:-3] if p.suffix=='.gz' else str(p)).with_suffix('.json') for p in args.generated)
    if getattr(args,'tolerances_from',None):protected.append(args.tolerances_from)
    if args.command == "validate":
        protected.extend([args.pdb, args.report])
    preflight(names.values(), args.overwrite, protected)
    if args.command == "validate":
        return validate_command(args, source, started, names)
    try:
        analysis_started = time.monotonic()
        LOG.info("Reading reference coordinates (model %d)", args.model)
        structure = read_pdb(source, args.model)
        LOG.info("Loaded %d accepted protein atoms; starting reference analysis", len(structure.atoms))
        analysis = analyze(structure, **analysis_options(args), progress=LOG.info)
        preparation_timings["analysis"] = time.monotonic()-analysis_started
    except (MastercapError, OSError, EOFError) as exc:
        if args.command == "generate":
            report = base_report(args, started)
            report["error"] = str(exc)
            diagnostic(args, report, protected)
            return 1
        raise
    log_warning_summary(analysis.warnings)
    if args.command=='calibrate':
        edges,profiles=target_profiles(analysis,args.inner_radius,args.outer_radius)
        settings=profile_settings(analysis,edges,args.bin_width,args.radius_method,args.radius_percentiles)
        def level_progress(name,stats):
            LOG.info('Calibration %s: %s',name,', '.join(f"{e}={value['mean']:.6g}" for e,value in stats.items()))
        calibration=calibrate(analysis,edges,profiles,settings,args.replicates,args.seed,args.compare,args.generated,
                              args.floor_multiple,args.floor_basis,args.floor_statistic,level_progress)
        calibration['suggested']['tolerances_from_line']=f'--tolerances-from {shlex.quote(str(names["json"]))}'
        LOG.info('Suggested: %s',calibration['suggested']['paste_line'])
        LOG.info('Equivalent: %s',calibration['suggested']['tolerances_from_line'])
        for name,rows in [('compare',calibration['compare']),('generated',calibration['generated'])]:
            for row in rows:
                log_warning_summary(row['warnings'])
                LOG.info('Calibration %s %s: %s',name,row['file'],', '.join(f'{e}={v:.6g}' for e,v in row.get('errors',row.get('shape')).items()))
        report=base_report(args,started)
        report.update(status='calibration_complete',calibration=calibration,warnings=analysis.warnings,acquisition=acquisition)
        files={names['json']:encode_json(report)}
        if args.csv:files[names['csv']]=calibration_csv(calibration)
        if args.plot:files[names['plot']]=calibration_png(calibration)
        publish(files,args.overwrite,protected)
        LOG.info('Calibration report: %s',names['json'])
        return 0
    if args.command == "generate":
        try:
            edges,_=target_profiles(analysis,args.inner_radius,args.outer_radius)
            configure_tolerances(args,analysis,edges)
        except MastercapError as exc:
            report=base_report(args,started)
            report.update(error=str(exc),analysis=analysis.summary())
            diagnostic(args,report,protected)
            raise
        return generate_command(args, analysis, source, acquisition, started, names, preparation_timings)
    report = base_report(args, started)
    report.update({"status": "analyzed", "analysis_options": analysis_options(args),
                   "analysis": analysis.summary(), "acquisition": acquisition})
    files = {names["json"]: encode_json(report)}
    extra_profiles(files, names, analysis.edges, analysis.original, analysis.smoothed,
                   smoothing_enabled=analysis.smoothing_sigma > 0)
    LOG.info("Serializing and publishing analysis report: %s", names["json"])
    publish(files, args.overwrite, protected)
    LOG.info("Analyzed %d atoms; report: %s", len(analysis.structure.atoms), names["json"])
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if hasattr(args,'pair_limit_override'):
        args.pair_limit_override = resolve_limit_overrides(args.pair_limit_override)
    if getattr(args,'tolerances_from',None) and (args.profile_tolerance is not None or args.composition_tolerance is not None):
        parser().error('--tolerances-from cannot be combined with --profile-tolerance or --composition-tolerance')
    logging.basicConfig(level=logging.ERROR if args.quiet else logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s", force=True)
    LOG.info("mastercap %s", __version__)
    try:
        return run(args)
    except (MastercapError, OSError, ValueError, KeyError, IndexError, EOFError, TypeError) as exc:
        LOG.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOG.error("Interrupted; no incomplete PDB will be published")
        return 130
