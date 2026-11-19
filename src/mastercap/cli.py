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


def parser():
    result = argparse.ArgumentParser(prog="mastercap", description="Reference-driven spherical geometric capsid particles. All distances are in angstroms.")
    result.add_argument("--version", action="version", version=__version__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("analyze", "generate", "validate"):
        description = {"analyze": "Analyze an already expanded protein capsid; no coordinates are generated.",
                       "generate": "Automatically ANALYZE reference -> GENERATE -> WRITE AND REREAD PDB -> VALIDATE. No separate analyze/validate commands are needed.",
                       "validate": "Independently reanalyze the reference and validate an existing PDB using its mandatory JSON report."}[name]
        command = sub.add_parser(name, help=description, description=description)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--input", type=Path, help="Local expanded .pdb/.vdb, optionally gzip compressed")
        source.add_argument("--id", help="Four-character VIPERdb identifier, e.g. 1cwp")
        command.add_argument("--model", type=positive_integer, default=1, help="Reference model number (default: 1); ignored by validate, which uses saved settings")
        command.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"mastercap", help="VIPERdb cache directory (honors XDG_CACHE_HOME)")
        command.add_argument("--refresh", action="store_true", help="Refresh the requested VIPERdb cache entry")
        command.add_argument("--output-prefix", type=Path, required=True, help="Output basename; extensions are appended")
        command.add_argument("--csv", action="store_true", help="Write radial density CSV")
        command.add_argument("--plot", action="store_true", help="Write radial density PNG")
        command.add_argument("--overwrite", action="store_true", help="Explicitly permit replacing existing output files")
        verbosity = command.add_mutually_exclusive_group()
        verbosity.add_argument("--quiet", action="store_true")
        verbosity.add_argument("--verbose", action="store_true")
        if name != "validate":
            command.add_argument("--bin-width", type=finite_positive, default=1.0, help="Requested maximum radial bin width in A (default: 1)")
            command.add_argument("--smoothing-sigma", type=finite_nonnegative, default=1.0, help="Gaussian width in A (provisional default: 1; not calibrated)")
            command.add_argument("--no-smoothing", action="store_true")
            command.add_argument("--radius-method", choices=("extrema", "percentile"), default="extrema")
            command.add_argument("--radius-percentiles", nargs=2, type=float, default=[0.5, 99.5], metavar=("LOW", "HIGH"))
            command.add_argument("--distance-percentile", type=finite_nonnegative, default=1.0, help="Lower distance percentile (default: 1)")
            command.add_argument("--min-pair-samples", type=positive_integer, default=100, help="Observations needed before using a pair-specific percentile (provisional: 100)")
        if name == "generate":
            command.add_argument("--inner-radius", type=finite_nonnegative)
            command.add_argument("--outer-radius", type=finite_positive)
            command.add_argument("--center", choices=("reference", "origin"), default="reference")
            command.add_argument("--mode", choices=("independent", "residue"), default="independent")
            command.add_argument("--distribution", choices=("random", "regular"), default="random")
            command.add_argument("--seed", type=int, help="Nonnegative seed; recorded random seed when omitted")
            command.add_argument("--max-iterations", type=int, default=1000, help="Maximum repair iterations (provisional: 1000)")
            command.add_argument("--time-limit", type=finite_positive, default=300.0, help="Cooperative repair-search budget in seconds, shared with fallback; excludes input, analysis, inventory and initialization (default: 300)")
            command.add_argument("--overlap-margin", type=finite_nonnegative, default=0.0, help="Required positive vdW overlap beyond this margin in A")
            command.add_argument("--profile-tolerance", type=finite_nonnegative, help="Maximum normalized integrated error for EACH element; no calibrated default")
            command.add_argument("--composition-tolerance", type=finite_nonnegative, help="Relative count tolerance per element, with one-atom rounding allowance; no calibrated default")
            command.add_argument("--allow-approximate", action="store_true", help="Permit soft-tolerance deviations or uncalibrated results; mandatory constraints may NEVER be relaxed")
            command.add_argument("--allow-independent-fallback", action="store_true", help="Explicitly allow retrying a failed rigid-residue search in independent mode")
        if name == "validate":
            command.add_argument("--pdb", type=Path, required=True, help="Generated particle PDB")
            command.add_argument("--report", type=Path, required=True, help="Matching mastercap JSON report")
    return result


def reference(args):
    if args.input:
        return args.input, {}
    return download(args.id, args.cache_dir, args.refresh)


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


def extra_profiles(files, names, edges, original, smoothed, generated=None):
    if "csv" in names:
        files[names["csv"]] = profile_csv(edges, original, smoothed, generated)
    if "plot" in names:
        files[names["plot"]] = profile_png(edges, original, smoothed, generated)


def diagnostic(args, report, protected):
    # Never replace the JSON belonging to a previously successful PDB.
    path = Path(str(args.output_prefix)+".diagnostic.json")
    publish({path: encode_json(report)}, args.overwrite, protected)
    LOG.error("No new PDB written. Diagnostic: %s", path)


def generate_command(args, analysis, source, acquisition, started, names, preparation_timings=None):
    if args.seed is None:
        args.seed = secrets.randbits(63)
    if args.seed < 0 or args.max_iterations < 0:
        raise MastercapError("Seed and max iterations must be nonnegative")
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
        inventory_started = time.monotonic()
        try:
            inventory, trace = select_inventory(analysis, edges, profiles, rng, time_limit=15.0)
        finally:
            timings["inventory"] = time.monotonic()-inventory_started
        report["residue_inventory"] = trace
        def progress(iteration, geometry, metrics):
            LOG.info("Iteration %d: components=%d distance violations=%d radial violations=%d profile error=%.4f", iteration,
                     geometry["global_components"], geometry["distance_violations"], geometry["outside_radial_bounds"], metrics["total_profile_error"])
        mode = args.mode
        def attempt(search_mode):
            attempt_started = time.monotonic()
            try:
                result, details = search(inventory.copy(), analysis, edges, profiles, center, rng,
                                         mode=search_mode, distribution=args.distribution,
                                         max_iterations=args.max_iterations,
                                         time_limit=args.time_limit-timings["search"],
                                         margin=args.overlap_margin, profile_tolerance=args.profile_tolerance,
                                         composition_tolerance=args.composition_tolerance, progress=progress)
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
        export_started = time.monotonic()
        text, mapping = pdb_text(particle)
        with tempfile.TemporaryDirectory(prefix="mastercap-validate-") as directory:
            temporary = Path(directory)/"particle.pdb"
            temporary.write_text(text)
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
        particle.xyz = reread.xyz
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
        extra_profiles(files, names, edges, analysis.original, profiles, metrics["generated_profiles"])
        timings["export_validation"] = time.monotonic()-export_started
        report["elapsed_seconds"] = time.monotonic()-started
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
    saved = json.loads(args.report.read_text())
    if saved.get("schema_version") != 1 or "pdb_sha256" not in saved:
        raise MastercapError("Report is not a supported successful generation report")
    analysis = analyze(read_pdb(source, saved["analysis"]["reference"]["model"]), **saved["analysis_options"])
    if analysis.structure.provenance["sha256"] != saved["analysis"]["reference"]["sha256"]:
        raise MastercapError("Reference SHA-256 does not match the generation report")
    actual_hash = hashlib.sha256(args.pdb.read_bytes()).hexdigest()
    if actual_hash != saved["pdb_sha256"]:
        raise MastercapError("Generated PDB SHA-256 does not match its report")
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
    geometry = validate_geometry(particle, center, edges, analysis.limits, config["overlap_margin"], saved["effective_mode"])
    metrics = profile_metrics(particle, center, edges, profiles, analysis.original)
    status = acceptance(geometry, metrics, config["profile_tolerance"], config["composition_tolerance"])
    report = base_report(args, started)
    report.update({"status": status, "geometry": geometry, "metrics": metrics,
                   "validated_pdb": str(args.pdb), "reference": analysis.structure.provenance})
    files = {names["json"]: encode_json(report)}
    extra_profiles(files, names, edges, analysis.original, profiles, metrics["generated_profiles"])
    publish(files, args.overwrite, [source, args.pdb, args.report])
    LOG.info("Validation status: %s", status)
    return 1 if status == "failed" else 0 if status == "valid" else 2


def run(args):
    started = time.monotonic()
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
        analysis = analyze(read_pdb(source, args.model), **analysis_options(args))
        preparation_timings["analysis"] = time.monotonic()-analysis_started
    except (MastercapError, OSError, EOFError) as exc:
        if args.command == "generate":
            report = base_report(args, started)
            report["error"] = str(exc)
            diagnostic(args, report, protected)
            return 1
        raise
    for warning in analysis.warnings:
        LOG.warning(warning)
    if args.command == "generate":
        return generate_command(args, analysis, source, acquisition, started, names, preparation_timings)
    report = base_report(args, started)
    report.update({"status": "analyzed", "analysis_options": analysis_options(args),
                   "analysis": analysis.summary(), "acquisition": acquisition})
    files = {names["json"]: encode_json(report)}
    extra_profiles(files, names, analysis.edges, analysis.original, analysis.smoothed)
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
