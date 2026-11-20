"""Command-line workflows. Help is part of the public interface."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import resource
import secrets
import sys
import tempfile
import time
import numpy as np
import scipy

from . import __version__
from .analysis import analyze, target_profiles, distance_statistics
from .constraints import acceptance, profile_metrics, validate_geometry
from .generate import select_inventory, search, residue_transforms
from .model import Atom, MastercapError
from .pdbio import download, read_pdb, pdb_text
from .reports import encode_json, preflight, publish, profile_csv, profile_png

LOG = logging.getLogger("mastercap")


class DetailedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show concrete defaults and preserve readable examples and explanations."""

    inferred_defaults = {
        "inner_radius": "inferred from the reference",
        "outer_radius": "inferred from the reference",
        "seed": "generate a seed and record it in JSON",
        "profile_tolerance": "unset (uncalibrated)",
        "composition_tolerance": "unset (uncalibrated)",
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
        return text


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
  Writes PREFIX.json and, when requested, PREFIX_profiles.csv / .png; no PDB.
  Element-inference and other reference warnings stay in JSON, with one terminal
  summary (suppressed by --quiet).""",
    "generate": """Examples:
  mastercap generate --input ./1cwp_full.vdb.gz --output-prefix results/particle --seed 42 --allow-approximate --csv --plot
  mastercap generate --id 1cwp --output-prefix results/particle --seed 42 --allow-approximate
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
  After first feasible geometry, allow up to --refine-iterations additional
  steps (default 100) to improve profiles; 0 restores first-feasible stopping.
  Refinement shares the existing iteration/time budgets, preserves the best
  feasible candidate and balances relative errors across elements.

Acceptance:
  Always require radial bounds, center error <= 1 A, applicable minimum distances,
  internal residue connectivity and global connectivity by positive vdW overlap.
  Unset profile/composition tolerances produce an uncalibrated result; saving it
  requires --allow-approximate. With tolerances, each element must pass to be valid.
  Profile error = sum(abs(generated density - target density) * shell volume)
                  / target element count.
  Composition tolerance is relative, with a minimum allowance of one atom.
  These quality tolerances have no scientifically calibrated defaults.
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
  Model, radii, smoothing, mode and quality tolerances come from --report.
  Plain and gzip references are interchangeable if decompressed content matches.
  Writes PREFIX.json and optional PREFIX_profiles.csv / .png; never generates a PDB.
  An uncalibrated status means mandatory checks passed but quality tolerances
  were not supplied; standalone validation cannot retroactively calibrate them.
  Plot smoothing follows the effective setting saved in --report.""",
}

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


def parser():
    result = CompleteHelpParser(
        prog="mastercap", formatter_class=DetailedHelpFormatter,
        description="Reference-driven spherical geometric capsid particles. All distances are in angstroms.\n"
                    "This help includes ALL command options, defaults and examples below.\n"
                    "Use mastercap COMMAND -h for a shorter, command-specific reference.",
        epilog="Choose analyze, generate or validate. generate includes analysis and final validation.\n"
               "Concrete defaults are shown beside options; inferred/unset defaults are explained.\n"
               "Run from an environment where mastercap and its dependencies are installed.\n"+EXIT_HELP)
    result.add_argument("--version", action="version", version=__version__, help="Print the installed mastercap version and exit")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("analyze", "generate", "validate"):
        description = {"analyze": "Analyze an already expanded protein capsid; no coordinates are generated.",
                       "generate": "Automatically ANALYZE reference -> GENERATE -> WRITE AND REREAD PDB -> VALIDATE. No separate analyze/validate commands are needed.",
                       "validate": "Independently reanalyze the reference and validate an existing PDB using its mandatory JSON report."}[name]
        command = sub.add_parser(name, help=description, description=description,
                                 formatter_class=DetailedHelpFormatter,
                                 epilog=COMMAND_EXAMPLES[name]+(SMOOTHING_HELP if name != "validate" else "")+"\n"+EXIT_HELP)
        result.command_parsers.append(command)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--input", type=Path, help="Local expanded .pdb/.vdb, optionally gzip compressed; required alternative to --id")
        source.add_argument("--id", help="Four-character VIPERdb identifier, e.g. 1cwp; required alternative to --input")
        model_help = "Reference MODEL number to read" if name != "validate" else "Ignored here: the reference MODEL number saved in --report is used"
        command.add_argument("--model", type=positive_integer, default=1, help=model_help)
        command.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"mastercap", help="VIPERdb cache directory (honors XDG_CACHE_HOME)")
        command.add_argument("--refresh", action="store_true", help="Download a fresh VIPERdb cache entry instead of reusing the existing one; applies to --id")
        command.add_argument("--no-check-certificate", action="store_true", help="Explicitly skip TLS certificate verification for VIPERdb downloads only (--id); default verification is enabled; does not bypass HTTP/network-policy errors")
        command.add_argument("--output-prefix", type=Path, required=True, help="Required output basename, e.g. results/particle; extensions are appended and parent directories created")
        command.add_argument("--csv", action="store_true", help="Write radial density profiles to PREFIX_profiles.csv")
        command.add_argument("--plot", action="store_true", help="Write radial density comparison plots to PREFIX_profiles.png")
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
            command.add_argument("--distance-percentile", type=finite_nonnegative, default=1.0, help="Nearest-neighbor percentile in [0, 100] for element-pair minimum distances; provisional")
            command.add_argument("--min-pair-samples", type=positive_integer, default=100, help="Minimum observations for a pair-specific distance percentile; fewer use grouped normalized reference distances; provisional")
        if name == "generate":
            command.add_argument("--inner-radius", type=finite_nonnegative, help="Inner atom-center radius in A; require 0 <= inner < outer")
            command.add_argument("--outer-radius", type=finite_positive, help="Outer atom-center radius in A; overrides do not change local density targets")
            command.add_argument("--center", choices=("reference", "origin"), default="reference", help="Target center: reference coordinate mean or origin (0, 0, 0)")
            command.add_argument("--mode", choices=("independent", "residue"), default="independent", help="Move individual atoms (independent) or preserve observed residue geometry with rigid moves (residue)")
            command.add_argument("--distribution", choices=("random", "regular"), default="random", help="Random isotropic or regular spherical-layer initialization; repair may move initial positions")
            command.add_argument("--seed", type=int, help="Nonnegative random seed for reproducible generation")
            command.add_argument("--max-iterations", type=int, default=1000, help="Maximum repair steps per attempt; nonnegative, provisional")
            command.add_argument("--time-limit", type=finite_positive, default=300.0, help="Cooperative repair-search budget in seconds, shared with fallback; excludes input, analysis, inventory and initialization")
            command.add_argument("--progress-interval", type=positive_integer, default=5, help="Print search progress every this many iterations, plus initial/final progress; --quiet suppresses progress")
            command.add_argument("--refine-iterations", type=nonnegative_integer, default=100, help="Additional repair/refinement steps after first mandatory-feasible geometry, sharing time/max-iteration limits; 0 returns the first feasible result; no exact profile-match guarantee")
            command.add_argument("--overlap-margin", type=finite_nonnegative, default=0.0, help="Extra required vdW overlap in A; must be nonnegative and tangency never qualifies")
            command.add_argument("--profile-tolerance", type=finite_nonnegative, help="Maximum normalized integrated radial error for EACH element; no scientifically calibrated value")
            command.add_argument("--composition-tolerance", type=finite_nonnegative, help="Relative per-element atom-count error limit, with a minimum one-atom allowance")
            command.add_argument("--allow-approximate", action="store_true", help="Permit soft-tolerance deviations or uncalibrated results; mandatory constraints may NEVER be relaxed")
            command.add_argument("--allow-independent-fallback", action="store_true", help="Explicitly allow retrying a failed rigid-residue search in independent mode")
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
    return {"width": args.bin_width, "sigma": 0.0 if args.no_smoothing else args.smoothing_sigma,
            "radius_method": args.radius_method, "percentiles": args.radius_percentiles,
            "distance_percentile": args.distance_percentile, "min_samples": args.min_pair_samples}


def artifacts(prefix, csv=False, plot=False, pdb=False):
    names = {"json": Path(str(prefix)+".json")}
    if pdb:
        names["pdb"] = Path(str(prefix)+".pdb")
    if csv:
        names["csv"] = Path(str(prefix)+"_profiles.csv")
    if plot:
        names["plot"] = Path(str(prefix)+"_profiles.png")
    return names


def base_report(args, started):
    return {"schema_version": 1, "mastercap_version": __version__,
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
            "command": args.command, "status": "failed", "elapsed_seconds": time.monotonic()-started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}


def extra_profiles(files, names, edges, original, smoothed, generated=None, *, smoothing_enabled=True):
    if "csv" in names:
        files[names["csv"]] = profile_csv(edges, original, smoothed, generated)
    if "plot" in names:
        LOG.info("Rendering radial profile PNG")
        files[names["plot"]] = profile_png(edges, original, smoothed, generated, smoothing_enabled=smoothing_enabled)


def diagnostic(args, report, protected):
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
    if args.seed < 0 or args.max_iterations < 0 or args.refine_iterations < 0:
        raise MastercapError("Seed, max iterations and refinement steps must be nonnegative")
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
                   "calibration": "pending_1cwp", "pdb_convention": "TER segments; serials and chains may repeat"})
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
            LOG.info("Iteration %d: components=%d isolated=%d distance violations=%d radial violations=%d "
                     "disconnected residues=%d center error=%.3f A profile error=%.4f max element error=%.4f", iteration,
                     geometry["global_components"], geometry["isolated_atoms"], geometry["distance_violations"],
                     geometry["outside_radial_bounds"], len(geometry["disconnected_residues"]),
                     geometry["center_error_angstrom"], metrics["total_profile_error"], worst)
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
        if particle is None:
            raise MastercapError(f"Search ended ({search_report.get('termination', 'preparation_error')}) without a candidate satisfying every mandatory constraint")
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
        _, generated_neighbors = distance_statistics(particle, args.distance_percentile, args.min_pair_samples)
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
        publish(files, args.overwrite, [source])
        LOG.info("%s: %s (%d atoms)", status, names["pdb"], len(particle.atoms))
        return 0 if status == "valid" else 2
    except (MastercapError, KeyboardInterrupt) as exc:
        if export_started is not None:
            timings["export_validation"] = time.monotonic()-export_started
        timings.setdefault("preparation", time.monotonic()-started-timings["search"])
        report.update({"status": "failed", "error": str(exc) or "Interrupted by user",
                       "elapsed_seconds": time.monotonic()-started})
        diagnostic(args, report, [source])
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


def validate_command(args, source, started, names):
    LOG.info("Loading generation report: %s", args.report)
    saved = json.loads(args.report.read_text())
    if saved.get("schema_version") != 1 or "pdb_sha256" not in saved:
        raise MastercapError("Report is not a supported successful generation report")
    LOG.info("Reading reference coordinates for validation")
    analysis = analyze(read_pdb(source, saved["analysis"]["reference"]["model"]),
                       **saved["analysis_options"], progress=LOG.info)
    if analysis.structure.provenance["sha256"] != saved["analysis"]["reference"]["sha256"]:
        raise MastercapError("Reference SHA-256 does not match the generation report")
    log_warning_summary(analysis.warnings)
    actual_hash = hashlib.sha256(args.pdb.read_bytes()).hexdigest()
    if actual_hash != saved["pdb_sha256"]:
        raise MastercapError("Generated PDB SHA-256 does not match its report")
    LOG.info("Reading generated PDB and checking provenance: %s", args.pdb)
    particle = read_pdb(args.pdb, generated=True)
    config = saved["configuration"]
    edges, profiles = target_profiles(analysis, config["inner_radius"], config["outer_radius"])
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
    metrics = profile_metrics(particle, center, edges, profiles, analysis.original)
    status = acceptance(geometry, metrics, config["profile_tolerance"], config["composition_tolerance"])
    report = base_report(args, started)
    report.update({"status": status, "geometry": geometry, "metrics": metrics,
                   "validated_pdb": str(args.pdb), "reference": analysis.structure.provenance,
                   "warnings": analysis.warnings})
    files = {names["json"]: encode_json(report)}
    extra_profiles(files, names, edges, analysis.original, profiles, metrics["generated_profiles"],
                   smoothing_enabled=analysis.smoothing_sigma > 0)
    LOG.info("Publishing validation report: %s", names["json"])
    publish(files, args.overwrite, [source, args.pdb, args.report])
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
    protected = [source]
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
    if args.command == "generate":
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
    logging.basicConfig(level=logging.ERROR if args.quiet else logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s", force=True)
    try:
        return run(args)
    except (MastercapError, OSError, ValueError, KeyError, IndexError, EOFError, TypeError) as exc:
        LOG.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOG.error("Interrupted; no incomplete PDB will be published")
        return 130
