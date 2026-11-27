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
from types import SimpleNamespace
import numpy as np
import scipy

from . import __version__
from .diagnostics import particle_diagnostics, mesh_sizes
from .refinement import resolve_drift_step, resolve_landing_depth
from .constraints import GeometryMetadata, inter_residue_configuration
from .analysis import (analyze, build_target, distance_statistics, parse_limit_override,
                       resolve_limit_overrides, held_out_target, dither_partitions, resolve_geometry)
from .atomdata import store_atom_data, load_atom_data, verify_provenance
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


class WallResidueStepAction(argparse.Action):
    def __call__(self,parser,namespace,value,option_string=None):
        setattr(namespace,self.dest,value)
        namespace.wall_residue_step_explicit = True


def log_profile_errors(metrics):
    held = metrics.get('held_out')
    if held:
        values = ', '.join(f'{e}={value:.4f}' if value is not None else f'{e}=n/a'
                           for e, value in held['elements'].items())
        own = ', '.join(f"{e}={v['profile_error']:.4f}" if v['profile_error'] is not None else f'{e}=n/a'
                        for e,v in metrics['elements'].items())
        LOG.info('Final normalized profile errors: %s; held-out (half-bin offset): %s, total=%s',
                 own, values, f"{held['total_profile_error']:.6f}" if held['total_profile_error'] is not None else 'n/a')
        if held.get('used_in_ranking'):
            LOG.info('Half-bin held-out is a ranking partition here; use independent held-out for comparison')
    independent = metrics.get('held_out_independent')
    if independent:
        LOG.info('Independent held-out profile error (offset %.6g; never ranked): total=%s',
                 independent['offset_fraction'], independent['total_profile_error'])
    dithered = metrics.get('dithered')
    if dithered:
        LOG.info('Dithered mean profile error (%d partitions): total=%.6f, %s',
                 dithered['k'], dithered['total_profile_error'],
                 ', '.join(f'{e}={v:.6f}' for e,v in dithered['elements'].items()))


def final_profile_metrics(structure, center, target, k=1):
    partitions = dither_partitions(target,k)
    held = held_out_target(target)
    independent = held if k == 1 else held_out_target(target,1/(2*k))
    return profile_metrics(structure, center, target.edges, target.profiles, target.reference_on_edges,
                           held_out=held, held_out_independent=independent,
                           partitions=partitions if k > 1 else None)


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



def final_pair_distance_check(structure, limits, report, inter_limits=None):
    audit = pair_distance_audit(structure, limits, inter_residue_limits=inter_limits)
    report['pair_distance_audit'] = audit
    require_pair_distance_audit(audit)
    sulfur = audit['pairs'].get('S-S', {})
    if sulfur.get('minimum_distance_angstrom') is not None:
        names = ' / '.join(f"{a['residue_name']} {a['residue_number']} {a['atom_name']}" for a in sulfur['closest_atoms'])
        LOG.info('Closest S-S: %.3f A (%s)',sulfur['minimum_distance_angstrom'],names)


def final_metadata(structure, analysis):
    if not analysis.inter_residue:return None
    return GeometryMetadata.build(structure,analysis.limits,enforce_covalent_floor=True,
                                   inter_residue_limits=analysis.inter_residue,inter_residue_scale=analysis.inter_scale,
                                   overridden_pairs=[r['pair'] for r in analysis.distance_limit_overrides])


def add_diagnostics(report, structure, analysis, center, target, sizes, transforms=None):
    key=tuple(sizes)
    if key not in analysis.diagnostics_cache:
        LOG.info('Computing reference packing diagnostics (cached per analysis)')
        analysis.diagnostics_cache[key]=particle_diagnostics(analysis.structure,analysis.center,analysis.edges,None,
                                                            element_sizes=sizes,
                                                            transforms=[{'rigid_fit_rmsd_angstrom':0.}])
    LOG.info('Computing particle packing, wall and layering diagnostics')
    d=particle_diagnostics(structure,center,target.edges,target.profiles,target.step,sizes,
                           analysis.diagnostics_cache[key],transforms=transforms)
    report['diagnostics']=d
    edge = d['edge_concentration']
    ratio = edge['ratio_within']['0.005']
    LOG.info('Edge concentration: within 0.005 A of interior bin edges %.3f %% (uniform ~%.3f %%, ratio %s)',
             100*edge['fraction_within']['0.005'], 100*edge['expected_within']['0.005'],
             f'{ratio:.3f}' if ratio is not None else 'n/a')
    if ratio is not None and ratio > 3:
        LOG.warning('High interior-bin edge concentration: within 0.005 A ratio %.3f exceeds 3; inspect independent held-out profile error', ratio)
    ref=d['reference'];h=str(min(sizes));threshold=d['contacts']['gap_threshold_angstrom'][h]
    def number(value):return f'{value:.4g}' if value is not None else 'n/a'
    LOG.info('Diagnostics: overlap/atom %s (ref %s); near-gap<%.3g %s (ref %s); '
             'inter-residue <2.0 A %s/atom (ref %s); walls inner/outer within 0.01 A: %d/%d '
             '(expected within 0.01 A ~%s/%s); layering SD %s; residue drift median %s A',
             number(d['contacts']['overlap_per_atom']),number(ref['contacts']['overlap_per_atom']),threshold,
             number(d['contacts']['near_gap_per_atom'][h]),number(ref['contacts']['near_gap_per_atom'][h]),
             number(d['inter_residue']['close_contacts_per_atom']['2.0']),number(ref['inter_residue']['close_contacts_per_atom']['2.0']),
             d['walls']['inner']['atoms_within']['0.01'],d['walls']['outer']['atoms_within']['0.01'],
             number(d['walls']['inner']['expected_within_0p01']),number(d['walls']['outer']['expected_within_0p01']),
             number(d['layering']['relative_sd']),number(d['residue_drift']['median_angstrom']))
    zone = d['contact_zone']
    LOG.info('Contact zone: inner zone error %s, outermost 0.5 A x%s, slope ratio %s; '
             'outer zone error %s, outermost 0.5 A x%s, slope ratio %s',
             number(zone['inner']['zone_error']), number(zone['inner']['wall_excess_0p5']), number(zone['inner']['slope_ratio']),
             number(zone['outer']['zone_error']), number(zone['outer']['wall_excess_0p5']), number(zone['outer']['slope_ratio']))


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

    if args.tolerance_source == 'calibration':
        LOG.info('Loaded calibration tolerances: %s (SHA-256 %s); reference/model/profile model/bin/smoothing/radii/step edge width settings matched',
                 args.calibration_file, args.calibration_sha256)
    elif args.tolerance_source == 'saved':
        LOG.info('Using tolerances saved in generation report: %s; original calibration file is not reread', args.report)
    else:
        LOG.info('Tolerance source: %s', args.tolerance_source)
    tolerances = args.profile_tolerance
    LOG.info('Effective profile tolerances: %s', ', '.join(
        f"{e}={tolerances.get(e, tolerances.get('*')):.8g}" for e in analysis.original)
        if tolerances is not None else 'not configured (uncalibrated)')
    LOG.info('Effective composition tolerance: %s',
             f'{args.composition_tolerance:.8g} relative; allowance per element = max(1 atom, tolerance * target count)'
             if args.composition_tolerance is not None else 'not configured (uncalibrated)')


def log_acceptance(status, geometry, metrics, composition_tolerance):
    """Explain the effective checks on the actual exported/validated coordinates."""
    LOG.info('Mandatory geometry: %s; final connectivity and pair-distance audits completed',
             'PASS' if geometry['mandatory_pass'] else 'FAIL')
    for element, values in metrics['elements'].items():
        tolerance = values['tolerance']
        profile = 'not configured' if tolerance is None else (
            f"error={values['profile_error'] or 0:.8g} {'<=' if values['within_tolerance'] else '>'} "
            f"tolerance={tolerance:.8g} {'PASS' if values['within_tolerance'] else 'FAIL'}")
        composition = 'not configured'
        if composition_tolerance is not None:
            allowance = max(1.0, composition_tolerance * values['target_count'])
            difference = abs(values['count_difference'])
            passed = difference <= allowance
            composition = f"absolute count difference={difference:.8g} {'<=' if passed else '>'} allowance={allowance:.8g} {'PASS' if passed else 'FAIL'}"
        LOG.info('Acceptance %s: profile %s; composition %s', element, profile, composition)
    explanations = {
        'valid': 'mandatory checks passed and every element meets its profile and composition tolerances; Total error is diagnostic, not a separate acceptance threshold',
        'approximate': 'mandatory checks passed, but at least one element exceeds a supplied tolerance',
        'uncalibrated': 'mandatory checks passed, but profile and/or composition tolerances are not configured',
        'failed': 'mandatory geometry failed; soft tolerances cannot permit export',
    }
    LOG.info('Acceptance status: %s; %s', status, explanations[status])


class SpliceOptionAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.splice_options_explicit = True


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
        "smoothing_sigma": "1.0 (0 with --step-profile)",
        "step_edge_width": "4.0 with --step-profile",
        "floor_basis": "orientation (counting for step or resized layered targets)",
        "inter_residue_scale": "1.0 with --inter-residue-limits reference",
        "curvature_factor": "1 when either geometry factor is supplied",
        "thickness_factor": "1 when either geometry factor is supplied",
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
  To estimate profile error scales and suggest tolerances, use mastercap calibrate
  with the intended generation bin width, smoothing and radii; analyze alone does
  not establish acceptance tolerances.
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
  Reference-model radii bound atom centers, not vdW surfaces. In step mode the
  CLI radii are half-height bounds; support extends half the ramp width beyond them.
  The 300 s repair budget excludes preparation; authorized fallback shares the
  remaining repair time. Individual repair operations can overrun the deadline.
  Progress is printed every 5 iterations and at termination; --progress-interval
  changes this frequency. Stage messages precede input reading and analysis.
  After first feasible geometry, allow up to --refine-updates profile pushes
  (default 50). Rolled-back pushes count too. Repair steps are not pushes.
  Stop immediately if mandatory geometry and all supplied tolerances pass;
  push/time budgets are upper limits, not required work. Zero pushes can be valid.
  --refine-iterations is an optional hard iteration cap, off by default.
  Either budget set to 0 restores first-feasible stopping. Time/max-iteration
  limits still apply. --selection-objective total ranks atom-weighted error
  first; worst-element restores the previous worst-element-first ranking.
  With --refine-bin-dither N>1, those keys use the mean of the N partitions;
  acceptance and tolerances continue to use the original own-bin errors.
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
  Logs show tolerance provenance, effective values and each element's pass/fail.
  valid means these geometric/profile checks passed, not chemical validation.
  --allow-approximate permits saving soft failures; it does not force that status.
  --allow-approximate NEVER permits violation of mandatory geometry/connectivity.

Output:
  Writes PREFIX.pdb and PREFIX.json after final written/reread validation;
  optional profiles use PREFIX_profiles.csv / .png. A failure writes a separate
  PREFIX.diagnostic.json when possible, without publishing an invalid PDB.
  One PDB uses TER-separated segments; serials/chains can repeat in large files.
  PDB-record-order provenance (sidecar by default, or inline) is authoritative. This is a geometric model, not a
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
  Logs explain effective tolerances and per-element profile/count pass/fail.
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
PREFIX_calibration.json/.csv/.png names. Calibration never generates a PDB.

Reading the calibration PNG:
  Each element has its own log error scale: left is smaller error. Bars span
  replicate p05-p95; dots show means (not confidence intervals for the mean).
  Hollow diamonds show --compare shape errors; filled triangles show --generated
  errors. Without those options the corresponding rows have no sample markers.
  The dashed line is the suggested per-element tolerance (default 2 * orientation
  p95, or counting p95 for step/resized layered targets); an element's triangle at or left of
  it passes that profile threshold.
  Total is diagnostic and has no tolerance. Zero errors use a labelled display
  position because log(0) is undefined. No particle is assessed unless supplied
  with --generated; rerun calibrate with that option after generation."""

STEP_HELP = """
STEP PROFILE:
  --step-profile is a homogeneous control arm: shared half-height bounds fitted
  to reference atom moments <r> and <r^2>, preserving N and composition at fitted
  size. It removes layering and outer-tail character (see skewness diagnostics).
  Fixed linear edge ramps have width --step-edge-width (default 4 A; 0 is hard).
  Atom support extends width/2 beyond each half-height bound. Resizing with
  --inner-radius/--outer-radius keeps plateau densities; counts scale with step
  volume. Missing bounds use the fit. Smoothing and sulfur placement reference
  are incompatible; sulfur placement profile follows the homogeneous S target.
  Step calibration uses counting only; orientation/placement floors are undefined.
  Experimental --compare samples are reported but do not enlarge suggestions.
  Counting x2 thresholds are stricter than orientation thresholds: approximate
  particles are expected and need --allow-approximate; geometry stays mandatory.
"""
for _command, _example in {
    'analyze': 'mastercap analyze --input ./1cwp_full.vdb --bin-width 4 --step-profile --output-prefix results/step_fit --csv --plot',
    'calibrate': 'mastercap calibrate --input ./1cwp_full.vdb --bin-width 4 --step-profile --replicates 50 --seed 42 --floor-multiple 2 --output-prefix results/calib_step --csv --plot',
    'generate': 'mastercap generate --input ./1cwp_full.vdb --bin-width 4 --step-profile --mode independent --distribution regular --tolerances-from results/calib_step_calibration.json --allow-approximate --output-prefix results/step01 --csv --plot',
}.items():
    COMMAND_EXAMPLES[_command] += '\nStep example:\n  '+_example+'\n'


FEM_HELP = """
PROFILE ERROR MEASURES:
  Own-bin error remains the measure for tolerances and acceptance. Ranking and
  patience use own-bin error by default, or the mean across N offset partitions
  with --refine-bin-dither N (N=1 is off). Transport cycles through those partitions.
  Half-bin held-out error (offset 0.5) is part of ranking when N>1 is even;
  use independent held-out error at offset 1/(2N), which is NEVER ranked.
  Compare particles using independent held-out error together with the interior-edge
  concentration ratio. Low own-bin error plus a high ratio indicates edge
  bunching; a ratio above 3 within 0.005 A warns but never changes status.
  Random landing spreads crossing endpoints between 0.002 A and the landing
  depth inside the destination bin. Crossings may move max-step + landing-depth;
  these larger trials retain all geometry/anchor/contact checks and backtracking.
  Contact zones report atom-centre densities in 0.25 A shells within 5 A of
  each wall, exact-target error, 0.5 A excess and density slope ratios. The
  4 A ramp scale and wall excess also affect vdW-sphere meshes; this is not
  a mesher-equivalent voxel calculation. Diagnostics never change status.
  Wall-residue moves whole residues away from surplus walls, up to
  --refine-wall-residue-step (default 0.5 A), excluding other atom-wise forces.
  Whole-residue acceptance/backtracking preserves each translation's geometry.
  Opt-in example: --refine-bin-dither 4 --refine-wall-selection wall-residue

REPAIR RECOVERY:
  Before first feasible geometry only, --repair-residue-reset on (default)
  watches consecutive disconnections/clashes. At --repair-stuck-iterations 50,
  fit a residue template back onto it, or unjam persistent clashes with a
  seeded 0.5 A tangential translation. Preserve anchor radius and fit inside
  support; skip incompatible templates. At most --repair-reset-limit 5 resets
  per residue, and max(1,floor(0.001*residue count)) successful resets/iteration.
  No extra RNG draws or position changes occur before a reset fires. Off
  retains v0.1.14 repair. Events are capped at 200 with totals kept separately.
  Undo reports distinguish same-residue and other-residue too-close pairs,
  and separately report wall-origin undos. An atom may count in both sets.

FEM-ORIENTED DIAGNOSTICS:
  Reports compare particle and reference sphere overlaps, near gaps, non-bonded
  inter-residue contacts, wall densities, 1 A layering and residue deformation.
  Under sphere/intersect meshing, sqrt(3)*h is the potential gap-bridging scale:
  0.866 A at h=0.5 A and 1.732 A at h=1 A. These are geometric diagnostics,
  not an exact voxelization of a particular grid. They NEVER change status.
  Across an R x thickness series, compare the same overlap, near-gap, contact,
  wall and layering measurements to hold the approximation constant.
  Generation controls (validation reuses the saved settings):
  --wall-clamp band uses fixed seeded depths rather than piling atoms on a wall.
  --refine-drift-step auto uses fixed-speed edge transport and per-edge quotas;
  cumulative imbalance routes through intervening bins. Independent conflicting
  trials can take checked half-steps before rejection, without weakening guards.
  patience allows transit time (0 still disables patience). Use wall-clamp legacy
  and refine-drift-step legacy together with refine-landing legacy and
  repair-residue-reset off to reproduce
  v0.1.11 search behavior (existing geometry and push fields; new diagnostics added).
  Optional inter-residue-limits reference adds scaled non-bonded reference floors
  between residues; intra-residue bounds are unchanged and pair overrides win.
  Narrow contact windows warn; no positive C/N/O window fails before search.
  These packing controls do not change calibration settings or target profiles.
"""

GEOMETRY_HELP = """
GEOMETRY FACTORS:
  --curvature-factor F sets outer = arm reference outer / F. Missing F or G
  means 1; supported range is [0.25,4]. Factors are exclusive with explicit radii.
  Both arms use Delta_req = (G-1)*reference step-fit thickness, rounded to the
  nearest whole reference bin (Python round, ties to even). Thickness = arm
  reference thickness + Delta; inner = outer - thickness. G_eff records the
  rounded multiplier relative to step-fit thickness, not each arm's own thickness.
  Step changes its plateau; layered removes/duplicates a core band, preserving
  wall profiles and bin spacing. Core cuts use fixed 1 A scoring smoothing,
  independent of target smoothing. No acceptable cut, zero-bin requested change
  or invalid bounds fails with details. The step report states whether its paired
  layered splice is feasible. Ramp width stays fixed. Explicit layered radii
  keep legacy stretching and warn when thickness changes.
  Splice controls: --splice-wall-keep 6 --splice-window 2 --splice-max-mismatch 0.10.
  Raising mismatch is a scientific choice; inspect geometry_scaling.splice.
  Layered example (1cwp at 1 A bins: removes 7 reference bins):
    mastercap generate --input ./1cwp_full.vdb.gz --output-prefix results/thinner --thickness-factor 0.8 --allow-approximate
  Step example (same absolute rounded Delta):
    mastercap generate --input ./1cwp_full.vdb.gz --output-prefix results/thinner_step --step-profile --thickness-factor 0.8 --allow-approximate
  calibrate uses the same resolver. Resized layered targets define counting
  baselines only; self/orientation/placement are skipped. Use profile sulfur
  placement when resizing. Old v0.1.16 reports validate using legacy stretching.
"""

REPORT_HELP = """
REPORT FILES:
  generate defaults to --report-atom-data sidecar: PREFIX.json plus compressed
  PREFIX.atomdata.npz (format_version 1, arrays only, no pickle). Atom mapping,
  residue transforms and search initial transforms are outside JSON. Keep the
  sidecar with the report; its relative path and SHA-256 are recorded in JSON.
  inline retains the legacy JSON blocks. omit stores no per-atom data; validate
  then checks residue signatures only and residue RMSD is unavailable.
  validate reads legacy inline data first, otherwise --atom-data PATH or the
  saved sidecar beside --report. Missing or hash-mismatched sidecars fail.
  Validation reports never inline atom data. Outputs share overwrite/protection
  rules; terminal output reports JSON and compressed atom-data sizes.
"""

SMOOTHING_HELP = """
Smoothing and radial resolution:
  --smoothing-sigma is Gaussian width in A, not a number of bins. Higher sigma
  smooths more, broadens peaks and suppresses narrow fluctuations; counts are
  conserved by shell-volume normalization. Sigma 0 also disables smoothing.
  For reference targets, --no-smoothing overrides sigma and targets the original binned mapped profile.
  Step targets use analytic linear ramps and no Gaussian smoothing.
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
  0    Analysis/calibration completed, or generated/validated result passes supplied tolerances.
  1    Input, search, validation or output failure.
  2    Approximate/uncalibrated result saved or verified; also CLI usage errors.
  130  Interrupted.
"""


def drift_step_spec(value):
    if value in ('auto','legacy'):return value
    return finite_positive(value)


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


def repair_stuck_count(value):
    number = positive_integer(value)
    if number < 5:raise argparse.ArgumentTypeError('must be an integer >= 5')
    return number


def wall_residue_step_value(value):
    number = finite_positive(value)
    if not .02 <= number <= 1.:raise argparse.ArgumentTypeError('must be in [0.02, 1.0] A')
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
               "For tolerance-based acceptance, run calibrate separately, then generate --tolerances-from its JSON.\n"
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
                                 epilog=COMMAND_EXAMPLES[name]+(FEM_HELP+REPORT_HELP if name in ("generate","validate") else "")+(GEOMETRY_HELP if name in ("generate","calibrate") else "")+(STEP_HELP+SMOOTHING_HELP if name != "validate" else "")+"\n"+EXIT_HELP)
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
            command.add_argument('--step-profile', action='store_true', help='Moment-matched homogeneous control target with shared bounds and reference composition; default keeps reference-shaped profiles')
            command.add_argument('--step-edge-width', type=finite_nonnegative, default=None, help='Linear ramp width in A centred on each step bound; 0 is a hard step; requires --step-profile')
            command.add_argument("--bin-width", type=finite_positive, default=1.0, help="Requested maximum radial shell-bin width in A; must be positive")
            command.add_argument("--smoothing-sigma", type=finite_nonnegative, default=None, help="Gaussian radial smoothing sigma in A; larger values smooth more and broaden peaks; 0 disables smoothing; overridden by --no-smoothing")
            command.add_argument("--no-smoothing", action="store_true", help="Disable Gaussian smoothing; reference targets use the original binned profile; step targets are always unsmoothed")
            command.add_argument("--radius-method", choices=("extrema", "percentile"), default="extrema", help="Infer reference bounds from minimum/maximum radii or the selected radial percentiles")
            command.add_argument("--radius-percentiles", nargs=2, type=float, default=[0.5, 99.5], metavar=("LOW", "HIGH"), help="Lower/upper radial percentiles, 0 <= LOW < HIGH <= 100; only used with --radius-method percentile; excluded atoms are reported")
            if name!="calibrate":
                command.add_argument("--distance-percentile", type=finite_nonnegative, default=1.0, help="Nearest-neighbor percentile in [0, 100] for element-pair minimum distances; provisional")
                command.add_argument("--min-pair-samples", type=positive_integer, default=100, help="Minimum observations for a pair-specific distance percentile; fewer use grouped normalized reference distances; provisional")
        if name in ("analyze","generate","calibrate"):
            command.add_argument("--inner-radius", type=finite_nonnegative, help="Inner atom-center radius in A; with --step-profile: step half-height bound; atoms may extend --step-edge-width/2 beyond it")
            command.add_argument("--outer-radius", type=finite_positive, help="Outer atom-center radius in A; with --step-profile: step half-height bound; atoms may extend --step-edge-width/2 beyond it")
        if name in ('generate','calibrate'):
            command.add_argument('--curvature-factor',type=finite_positive,default=None,help='Curvature multiplier F in [0.25,4]: arm reference outer radius / F; exclusive with explicit radii; missing factor means 1')
            command.add_argument('--thickness-factor',type=finite_positive,default=None,help='Thickness multiplier G in [0.25,4]. Both arms change by the same absolute Delta = (G-1) x step-fit thickness, rounded to whole reference bins. Layered arm: wall-preserving core splice (refused without an acceptable cut). Step arm: longer/shorter plateau. Missing factor means 1; exclusive with explicit radii')
            command.add_argument('--splice-wall-keep',type=finite_positive,default=6.0,action=SpliceOptionAction,help='Reference core exclusion distance from fitted walls in A; only layered thickness splices')
            command.add_argument('--splice-window',type=finite_positive,default=2.0,action=SpliceOptionAction,help='Half-window in A for fixed-1-A-smoothed splice scoring; only layered thickness splices')
            command.add_argument('--splice-max-mismatch',type=positive_fraction,default=0.10,action=SpliceOptionAction,help='Maximum relative RMS seam mismatch in (0,1]; raise knowingly; only layered thickness splices')
        if name == "generate":
            command.add_argument('--report-atom-data',choices=('sidecar','inline','omit'),default='sidecar',help='Per-atom mapping and residue transforms: compressed NPZ beside JSON (default), old inline layout, or omit with weaker signature-only validation')
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
            command.add_argument("--selection-objective", choices=("total", "worst-element"), default="total", help="Ranks feasible candidates by total or worst per-element error first; uses own-bin errors by default, partition-mean errors with bin dithering; tolerances remain own-bin")
            command.add_argument("--refine-updates", type=nonnegative_integer, default=50, help="Maximum profile pushes, counting rolled-back pushes too; stops earlier when supplied tolerances pass; 0 stops at first feasible geometry")
            command.add_argument("--refine-patience", type=nonnegative_integer, default=10, help="Stop after this many pushes without a significant relative improvement in the primary selection error (0.1 percent); 0 disables patience stopping")
            command.add_argument("--refine-strength", type=finite_positive, default=0.005, help="Radial strength, fixed by default; gain/acceptance controls adapt up to 4x and stall below 1/16; rollback retains damage feedback")
            command.add_argument("--refine-strength-control", choices=("fixed", "gain", "acceptance"), default="fixed", help="Per-atom strength: fixed keeps the starting value; gain follows primary profile improvement; acceptance restores the previous acceptance feedback. Rollback mode retains damage feedback")
            command.add_argument("--refine-min-step", type=finite_nonnegative, default=0.002, help="Minimum nonzero requested radial move in A when the target mismatch is at least this large; must not exceed --refine-max-step; 0 disables")
            command.add_argument('--inter-residue-limits',choices=('off','reference'),default='off',help='Optional non-bonded nearest-reference inter-residue distance floors; off retains old bounds')
            command.add_argument('--inter-residue-scale',type=finite_positive,default=None,help='Scale reference inter-residue floors (default 1.0); requires --inter-residue-limits reference; pair overrides take precedence')
            command.add_argument('--wall-clamp',choices=('band','legacy'),default='band',help='Repair outside atoms into a fixed seeded edge-band depth; legacy retains the old wall clamp')
            command.add_argument('--refine-drift-step',type=drift_step_spec,default='auto',help='Cumulative edge transport in A: auto=max(min step,min(max step,0.02*bin width)); checked independent half-step fallback; legacy keeps old speed/local quotas/rejection')
            command.add_argument('--refine-transport-routing',choices=('cumulative','local'),default='cumulative',help='Cumulative imbalance carries material through intervening bins; local restricts transfers to adjacent surplus/deficit bins; legacy drift forces local routing')
            command.add_argument('--refine-transport-factor',type=finite_positive,default=2.0,help='Maximum drifters per edge/direction = ceil(factor * remaining quota); legacy drift retains old quotas')
            command.add_argument('--refine-landing',choices=('random','legacy'),default='random',help='Random spreads crossings inside destination bins using the search RNG; legacy preserves distance + 0.002 and consumes no landing draws')
            command.add_argument('--refine-landing-depth',default='auto',help='Landing depth in A: auto=min(0.25*bin width,0.25 A); numeric range [0.002,half configured bin width], capped per destination at half its actual width; crossing moves may reach max step + depth (about 0.30 A at 1 A bins)')
            command.add_argument('--refine-bin-dither',type=positive_integer,default=1,help='Fixed offset partitions cycled by push; N>1 ranks by their mean profile error; N=1 retains own-bin transport/ranking; tolerances remain own-bin')
            command.add_argument('--refine-wall-selection',choices=('edge-first','wall-first','wall-residue'),default='edge-first',help='Wall-first moves nearest-wall atoms; wall-residue rigidly translates their entire residues; both require crossing selection and auto/numeric drift')
            command.set_defaults(wall_residue_step_explicit=False)
            command.add_argument('--refine-wall-residue-step',type=wall_residue_step_value,action=WallResidueStepAction,default=.5,help='Largest rigid wall-residue translation in A, range [0.02,1.0]; no effect with other wall selectors')
            command.add_argument('--repair-residue-reset',choices=('on','off'),default='on',help='Recover stuck pre-feasible repair residues with template resets or tangential unjams; off retains v0.1.14 repair')
            command.add_argument('--repair-stuck-iterations',type=repair_stuck_count,default=50,help='Consecutive disconnected/clashing repair iterations before recovery; integer >= 5; inactive after first feasible geometry')
            command.add_argument('--repair-reset-limit',type=positive_integer,default=5,help='Maximum successful repair resets per residue; integer >= 1')
            command.add_argument("--refine-selection", choices=("crossing", "mismatch"), default="crossing", help="Crossing targets useful adjacent-bin crossings, then drifts surplus atoms toward edges; skips previous rejected atoms. Mismatch keeps v0.1.5 rank-matched selection")
            command.add_argument("--refine-max-step", type=finite_positive, default=0.05, help="Crossing candidate reach in A; random landing allows requested crossings up to this + landing depth; legacy/mismatch retain the original move cap; excludes other forces and rigid projection")
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
            command.add_argument('--mesh-element-sizes',type=mesh_sizes,default="0.5,1.0",help='Comma-separated positive mesh element sizes in A; diagnostic gap thresholds sqrt(3)*h only; never changes status')
            command.set_defaults(profile_tolerance_spec=None)
            command.add_argument('--profile-tolerance',action=ProfileToleranceAction,help='Per-element normalized profile error: NUMBER, C=0.02,N=0.04,..., or C=0.02,*=0.05; each target element needs its own value or *; no automatic default')
            command.add_argument('--composition-tolerance',type=finite_nonnegative,help='Relative per-element count error limit, with a minimum one-atom allowance')
            command.add_argument('--tolerances-from',type=Path,help='Use suggested tolerances from a calibration JSON with matching reference/model/profile model/bin/smoothing/radii/step edge width; incompatible with explicit tolerance options')
        if name=='calibrate':
            command.add_argument('--replicates',type=positive_integer,default=50,help='Independent replicates per level; step targets use counting only')
            command.add_argument('--seed',type=nonnegative_integer,default=42,help='Seed for reproducible calibration resampling')
            command.add_argument('--compare',type=Path,action='append',default=[],help='Repeatable experimental comparison PDB; its own center, reference edges; report raw and count-normalized shape error')
            command.add_argument('--generated',type=Path,action='append',default=[],help='Repeatable generated PDB; check sibling report settings and use its saved center when available')
            command.add_argument('--floor-multiple',type=finite_positive,default=2.0,help='Multiply the chosen floor statistic to suggest tolerances; never applied automatically')
            command.add_argument('--floor-basis',choices=('orientation','counting'),default=None,help='Reference error level used for suggestions; step targets require counting')
            command.add_argument('--floor-statistic',choices=('p95','mean'),default='p95',help='Chosen floor statistic; round suggestions upward to three significant figures')
        if name == "validate":
            command.add_argument("--pdb", type=Path, required=True, help="Required generated particle PDB to revalidate")
            command.add_argument("--report", type=Path, required=True, help="Required JSON report from the matching successful generation")
            command.add_argument('--atom-data',type=Path,help='Override a sidecar location; inline provenance takes priority; SHA-256 is checked against the generation report')
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
    target_options = {'step_profile': args.step_profile, 'step_edge_width': args.step_edge_width}
    if args.command=="calibrate":
        return {"width":args.bin_width,"sigma":0.0 if args.no_smoothing else args.smoothing_sigma,
                "radius_method":args.radius_method,"percentiles":args.radius_percentiles,"compute_distances":False, **target_options}
    return {"width": args.bin_width, "sigma": 0.0 if args.no_smoothing else args.smoothing_sigma,
            "radius_method": args.radius_method, "percentiles": args.radius_percentiles,
            "distance_percentile": args.distance_percentile, "min_samples": args.min_pair_samples,
            "pair_limit_overrides": resolve_limit_overrides(args.pair_limit_override),
            "inter_residue_limits":getattr(args,'inter_residue_limits','off'),
            "inter_residue_scale":getattr(args,'inter_residue_scale',1.0), **target_options}


def resolve_target_options(args):
    """Resolve contextual defaults and reject incompatible flags before input I/O."""
    if args.command == 'validate':
        return
    if args.command in ('generate','calibrate'):
        factors=(args.curvature_factor,args.thickness_factor)
        if any(value is not None for value in factors):
            if args.inner_radius is not None or args.outer_radius is not None:
                raise MastercapError('Geometry factors are mutually exclusive with --inner-radius and --outer-radius')
            if any(value is not None and not .25 <= value <= 4 for value in factors):
                raise MastercapError('Geometry factors outside the supported range [0.25, 4]')
    if args.command in ('generate','calibrate'):
        if (getattr(args,'splice_options_explicit',False) and
                (args.step_profile or args.thickness_factor in (None,1))):
            LOG.warning('Splice configuration has no effect without a layered thickness factor')
        if (not args.step_profile and getattr(args,'sulfur_residue_placement','off') == 'reference'
                and any(value not in (None,1) for value in (args.curvature_factor,args.thickness_factor))):
            raise MastercapError('--sulfur-residue-placement reference requires the reference geometry; use profile placement when resizing')
    if args.command == 'generate':
        if args.inter_residue_limits=='off' and args.inter_residue_scale is not None:
            raise MastercapError('--inter-residue-scale requires --inter-residue-limits reference')
        args.inter_residue_scale=1.0 if args.inter_residue_scale is None else args.inter_residue_scale
        resolve_drift_step(args.refine_drift_step,args.bin_width,args.refine_min_step,args.refine_max_step)
        if (args.refine_selection == 'crossing' and args.refine_landing == 'random') or args.refine_landing_depth != 'auto':
            resolve_landing_depth(args.refine_landing_depth,args.bin_width)
        if args.refine_wall_selection != 'edge-first' and (args.refine_selection != 'crossing' or args.refine_drift_step == 'legacy'):
            raise MastercapError('Wall selection requires crossing selection and an auto or numeric refine drift step')
        if args.wall_residue_step_explicit and args.refine_wall_selection != 'wall-residue':
            LOG.warning('--refine-wall-residue-step has no effect without --refine-wall-selection wall-residue')
    if not args.step_profile and args.step_edge_width is not None:
        raise MastercapError('--step-edge-width requires --step-profile')
    if args.step_profile and args.smoothing_sigma is not None and args.smoothing_sigma > 0:
        raise MastercapError('--smoothing-sigma does not apply to --step-profile targets; remove it or use --no-smoothing')
    if args.step_profile and getattr(args, 'sulfur_residue_placement', 'off') == 'reference':
        raise MastercapError('--sulfur-residue-placement reference is incompatible with --step-profile')
    args.step_edge_width = 4.0 if args.step_edge_width is None else args.step_edge_width
    args.smoothing_sigma = 0.0 if args.step_profile else (1.0 if args.smoothing_sigma is None else args.smoothing_sigma)
    if args.command == 'calibrate':
        args.floor_basis_explicit = args.floor_basis is not None
        if args.step_profile and args.floor_basis == 'orientation':
            raise MastercapError('orientation floors are not defined for step targets; use --floor-basis counting')
        if args.floor_basis is None:
            args.floor_basis = 'counting' if args.step_profile else 'orientation'


def target_settings(args, analysis, edges):
    return profile_settings(analysis, edges, args.bin_width, args.radius_method, args.radius_percentiles,
                            model='step' if args.step_profile else 'reference',
                            step_edge_width=args.step_edge_width if args.step_profile else None,
                            curvature_factor=getattr(args,'curvature_factor',None),thickness_factor=getattr(args,'thickness_factor',None),
                            splice_wall_keep=getattr(args,'splice_wall_keep',6.0),splice_window=getattr(args,'splice_window',2.0),
                            splice_max_mismatch=getattr(args,'splice_max_mismatch',0.10))


def configured_target(args, analysis):
    inner,outer,scaling=resolve_geometry(analysis,step=args.step_profile,edge_width=args.step_edge_width,
        curvature_factor=getattr(args,'curvature_factor',None),thickness_factor=getattr(args,'thickness_factor',None),
                            splice_wall_keep=getattr(args,'splice_wall_keep',6.0),splice_window=getattr(args,'splice_window',2.0),
                            splice_max_mismatch=getattr(args,'splice_max_mismatch',0.10))
    if scaling is not None:
        args.inner_radius,args.outer_radius=inner,outer
        args.curvature_factor=scaling['curvature_factor'];args.thickness_factor=scaling['thickness_factor']
        old,new=scaling['reference'],scaling['generated']
        LOG.info('Geometry factors: curvature x %g (outer %.6f -> %.6f A), requested thickness G %g (%.6f -> %.6f A); '
                 'inner %.6f A; mid-radius %.6f -> %.6f A; arm %s',
                 scaling['curvature_factor'],old['outer'],new['outer'],scaling['thickness_factor'],old['thickness'],new['thickness'],
                 new['inner'],old['mid_radius'],new['mid_radius'],scaling['arm'])
        LOG.info('Thickness: G %g requested -> Delta %+.4f A (%d bins of %.6f A), G_eff %.4f; '
                 'arm thickness %.4f -> %.4f A (ratio %.4f)',scaling['thickness_factor_requested'],
                 scaling['delta_angstrom'],scaling['delta_bins'],scaling['grid_width_angstrom'],
                 scaling['thickness_factor_effective'],old['thickness'],new['thickness'],scaling['arm_thickness_ratio'])
        paired=scaling.get('paired_layered_splice')
        if paired and not paired['feasible']:
            LOG.info('Paired layered splice is not feasible: %s',paired['reason'])
    from .splice import SplicePlan
    plan=SplicePlan.from_report(scaling['splice'],analysis) if scaling and 'splice' in scaling else None
    target = build_target(analysis, args.inner_radius, args.outer_radius, step=args.step_profile,
                          edge_width=args.step_edge_width, bin_width=args.bin_width, splice=plan)
    target.geometry_scaling=scaling
    if target.splice is not None:
        scaling['splice']=target.splice
        block=target.splice
        element_text=' '.join(f"{e} {v:.3f}" if v is not None else f'{e} unavailable'
                              for e,v in block['mismatch_by_element'].items())
        LOG.info('Layered splice: %s bins %d..%d (reference r %.3f..%.3f A), seam at generated r %.3f A, '
                 'mismatch total %.4f (threshold %.4f; candidates %d, median %s); %s',
                 block['mode'],block['cut_bin'],block['cut_bin']+block['n_bins'],*block['band_reference_radii'],
                 block['seam_generated_radius'],block['mismatch_total'],block['threshold'],block['candidates'],
                 block['candidate_median_mismatch'],element_text)
    if not args.step_profile and scaling is None and target.thickness_method == 'stretch':
        LOG.warning('Explicit radii stretch the layered profile (thickness ratio %.4f); layer spacing is scaled. '
                    'Use --thickness-factor for the wall-preserving splice.',
                    (target.edges[-1]-target.edges[0])/(analysis.outer-analysis.inner))
    if target.step is not None:
        block = target.step
        bounds, density = block['generated_bounds'], block['density']
        sensitivity = ', '.join(f"{name} t={value['thickness']:.4f}" if 'thickness' in value else f'{name} fit unavailable'
                                for name, value in block['sensitivity'].items())
        LOG.info('Step target: bounds %.6f..%.6f A (t %.6f A), support %.6f..%.6f A, edge %g A; '
                 'plateau %.6g atoms/A3 (%.3f x reference mean, %.3f x densest bin); sensitivity: %s; '
                 'skewness reference %.6f vs step %.6f',
                 bounds['inner'], bounds['outer'], bounds['outer']-bounds['inner'], *bounds['support'],
                 block['edge_width'], density['plateau_total'], density['ratio_to_mean'], density['ratio_to_max_bin'],
                 sensitivity, block['skewness']['reference'], block['skewness']['step'])
        if density['plateau_total'] > density['reference_max_bin']:
            LOG.warning('Step plateau density %g exceeds the densest reference bin %g; packing may fail',
                        density['plateau_total'], density['reference_max_bin'])
    return target


def rebuild_target(analysis, saved):
    config=saved['configuration'];options=saved['analysis_options']
    scaling=saved.get('geometry_scaling')
    if scaling and scaling.get('thickness_rule') == 'shared_absolute_delta':
        if config.get('curvature_factor') is None and config.get('thickness_factor') is None:
            raise MastercapError('Recomputed splice differs from geometry_scaling.splice: missing saved factors')
        args=SimpleNamespace(inner_radius=None,outer_radius=None,
            curvature_factor=config.get('curvature_factor'),thickness_factor=config.get('thickness_factor'),
            step_profile=options.get('step_profile',False),step_edge_width=options.get('step_edge_width',4.0),
            bin_width=options['width'],splice_wall_keep=config.get('splice_wall_keep',6.0),
            splice_window=config.get('splice_window',2.0),splice_max_mismatch=config.get('splice_max_mismatch',0.10))
        try:
            target=configured_target(args,analysis)
        except MastercapError as exc:
            raise MastercapError(f'Recomputed splice differs from geometry_scaling.splice: {exc}') from exc
        if not np.allclose([args.inner_radius,args.outer_radius],
                           [config['inner_radius'],config['outer_radius']],rtol=0,atol=1e-9):
            raise MastercapError('Recomputed splice differs from geometry_scaling.splice: configuration bounds')
        return target
    if scaling and not options.get('step_profile',False):
        LOG.warning('Legacy v0.1.16 report: layered thickness was stretched, not spliced')
    return build_target(analysis,config['inner_radius'],config['outer_radius'],
                        step=options.get('step_profile',False),edge_width=options.get('step_edge_width',4.0),
                        bin_width=options['width'])


def verify_geometry_scaling(target,saved):
    if not saved.get('geometry_scaling'):return
    try:
        recorded=saved['geometry_scaling']['generated']
        bounds=target.step['generated_bounds'] if target.step else {'inner':target.edges[0],'outer':target.edges[-1]}
        if not np.allclose([bounds[k] for k in ('inner','outer')],[recorded[k] for k in ('inner','outer')],rtol=0,atol=1e-9):
            raise ValueError('resolved bounds')
    except (KeyError,ValueError,TypeError) as exc:
        raise MastercapError(f'Recomputed geometry differs from geometry_scaling.generated: {exc}') from exc
    previous=saved['geometry_scaling']
    if previous.get('thickness_rule') == 'shared_absolute_delta':
        current=target.geometry_scaling or {}
        def differ(key):
            raise MastercapError(f'Recomputed splice differs from geometry_scaling.splice: {key}')
        if current.get('delta_bins') != previous.get('delta_bins'):differ('delta_bins')
        old,new=previous.get('splice'),current.get('splice')
        if (old is None) != (new is None):differ('presence')
        if old is not None:
            for key in ('mode','cut_bin','n_bins','window_bins','threshold'):
                if new.get(key) != old.get(key):differ(key)
            try:
                if not np.isclose(new['mismatch_total'],old['mismatch_total'],rtol=0,atol=1e-12):differ('mismatch_total')
            except (KeyError,TypeError,ValueError):differ('mismatch_total')
        recorded_edges=saved.get('target',{}).get('edges')
        if recorded_edges is not None:
            if np.shape(recorded_edges) != target.edges.shape or not np.allclose(recorded_edges,target.edges,rtol=0,atol=1e-9):differ('edges')


def verify_saved_step(target, saved):
    previous = saved.get('target', {}).get('step')
    if target.step is None and previous is None:
        return
    try:
        if target.step is None or previous is None:
            raise ValueError('profile model')
        current = target.step
        if (not np.allclose([current['generated_bounds'][k] for k in ('inner', 'outer')],
                            [previous['generated_bounds'][k] for k in ('inner', 'outer')], rtol=0, atol=1e-9)
                or not np.allclose(current['generated_bounds']['support'], previous['generated_bounds']['support'], rtol=0, atol=1e-9)):
            raise ValueError('bounds')
        counts, old_counts = current['counts']['target'], previous['counts']['target']
        if counts.keys() != old_counts.keys() or not all(np.isclose(v, old_counts[e], rtol=1e-9, atol=0) for e, v in counts.items()):
            raise ValueError('target counts')
        edges = saved['target']['edges']
        if len(edges) != len(target.edges) or not np.allclose(target.edges, edges, rtol=0, atol=1e-9):
            raise ValueError('edges')
    except (KeyError, TypeError, ValueError) as exc:
        raise MastercapError(f'Recomputed step target differs from the generation report: {exc}') from exc


def artifacts(prefix, csv=False, plot=False, pdb=False, atom_data=False):
    names = {"json": Path(str(prefix)+".json")}
    if pdb:
        names["pdb"] = Path(str(prefix)+".pdb")
    if atom_data:names['atom_data']=Path(str(prefix)+'.atomdata.npz')
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


def extra_profiles(files, names, edges, original, smoothed, generated=None, *, smoothing_enabled=True, target_label="Smoothed target"):
    if "csv" in names:
        files[names["csv"]] = profile_csv(edges, original, smoothed, generated)
    if "plot" in names:
        LOG.info("Rendering radial profile PNG")
        files[names["plot"]] = profile_png(edges, original, smoothed, generated, smoothing_enabled=smoothing_enabled, target_label=target_label)


def diagnostic(args, report, protected):
    protected=list(protected)+([args.tolerances_from] if getattr(args,'tolerances_from',None) else [])
    # Never replace the JSON belonging to a previously successful PDB.
    path = Path(str(args.output_prefix)+".diagnostic.json")
    files={}
    if 'initial_transforms' in report.get('search',{}) and getattr(args,'report_atom_data','inline') != 'inline':
        files.update(store_atom_data(report,args.report_atom_data,Path(str(args.output_prefix)+'.diagnostic.atomdata.npz')))
    files[path]=encode_json(report)
    publish(files, args.overwrite, protected)
    LOG.error("No new PDB written. Diagnostic: %s", path)


def log_warning_summary(warnings):
    """Keep per-record reference diagnostics in JSON, even in verbose mode."""
    if warnings:
        inferred = sum(warning.startswith("Inferred ") for warning in warnings)
        LOG.warning("Reference analysis: %d warnings (%d element inferences); details only in JSON.",
                    len(warnings), inferred)


def generate_command(args, analysis, source, acquisition, started, names, preparation_timings=None, target=None):
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
    target = target or configured_target(args, analysis)
    edges, profiles = target.edges, target.profiles
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
    if target.step is not None:
        report["target"]["step"] = target.step
    if target.thickness_method is not None:report['target']['thickness_method']=target.thickness_method
    if target.geometry_scaling is not None:report['geometry_scaling']=target.geometry_scaling
    LOG.info("Selected reference: %d atoms; target radii %.3f..%.3f A; seed %d", len(analysis.structure.atoms), edges[0], edges[-1], args.seed)
    export_started = None
    try:
        inter_limits, inter_report=inter_residue_configuration(analysis,args.overlap_margin,LOG.warning)
        report['inter_residue_limits']=inter_report
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
        attempt_structure = None
        def attempt(search_mode):
            nonlocal attempt_structure
            attempt_structure=inventory.copy()
            attempt_started = time.monotonic()
            try:
                result, details = search(attempt_structure, analysis, edges, profiles, center, rng,
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
                                         wall_clamp=args.wall_clamp,step_edge_width=target.step['edge_width'] if target.step else None,
                                         refine_drift_step=args.refine_drift_step,refine_transport_factor=args.refine_transport_factor,bin_width=args.bin_width,
                                         refine_transport_routing=args.refine_transport_routing,
                                         refine_landing=args.refine_landing,refine_landing_depth=args.refine_landing_depth,
                                         held_out=held_out_target(target),
                                         refine_bin_dither=args.refine_bin_dither,refine_wall_selection=args.refine_wall_selection,
                                         target_definition=target,
                                         repair_residue_reset=args.repair_residue_reset,
                                         repair_stuck_iterations=args.repair_stuck_iterations,
                                         repair_reset_limit=args.repair_reset_limit,
                                         refine_wall_residue_step=args.refine_wall_residue_step,
                                         target_model="step" if target.step else "reference",
                                         original_profiles=target.reference_on_edges, stage_progress=LOG.info)
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
        report["wall_clamp"] = search_report.get("wall_clamp")
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
            if attempt_structure is not None:
                diagnostic_started=time.monotonic()
                last=attempt_structure.copy();last.xyz=np.round(last.xyz,3)
                report['diagnostics_state']='last search coordinates; no PDB exported'
                report['diagnostic_geometry']=validate_geometry(last,center,edges,analysis.limits,args.overlap_margin,mode,
                                                               metadata=final_metadata(last,analysis))
                add_diagnostics(report,last,analysis,center,target,args.mesh_element_sizes,residue_transforms(last,analysis.structure))
                if target.step:
                    coarse=build_target(analysis,args.inner_radius,args.outer_radius,step=True,edge_width=args.step_edge_width,bin_width=4.)
                    report['diagnostic_metrics_rebinned_4A']=profile_metrics(last,center,coarse.edges,coarse.profiles,coarse.reference_on_edges)
                timings['failed_state_diagnostics']=time.monotonic()-diagnostic_started
            raise MastercapError(f"Search ended ({search_report.get('termination', 'preparation_error')}) without a candidate satisfying every mandatory constraint")
        if search_report.get('termination') == 'constraints_satisfied':
            if search_report.get('status') == 'valid':
                LOG.info('Search stopped because mandatory geometry and all supplied profile/composition tolerances passed; '
                         '%d of %d maximum profile pushes used. Push/time budgets are upper limits, not required work',
                         search_report.get('profile_updates', 0), args.refine_updates)
            else:
                LOG.info('Search stopped at first feasible geometry because a refinement budget was set to 0; quality status=%s',
                         search_report.get('status'))
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
        geometry = validate_geometry(reread, center, edges, analysis.limits, args.overlap_margin, mode,metadata=final_metadata(reread,analysis))
        final_connectivity_check(reread, args.overlap_margin, geometry, report)
        final_pair_distance_check(reread, analysis.limits, report,inter_limits)
        metrics = final_profile_metrics(reread, center, target, args.refine_bin_dither)
        status = acceptance(geometry, metrics, args.profile_tolerance, args.composition_tolerance)
        report.update({"geometry": geometry, "metrics": metrics, "status": status})
        log_acceptance(status, geometry, metrics, args.composition_tolerance)
        if status == "failed":
            raise MastercapError("Written-and-reread PDB violates mandatory constraints")
        if status != "valid" and not args.allow_approximate:
            raise MastercapError(f"Candidate is {status}; provide tolerances or explicitly use --allow-approximate")
        log_profile_errors(metrics)
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
        transforms = residue_transforms(particle, analysis.structure)
        add_diagnostics(report, reread, analysis, center, target, args.mesh_element_sizes, transforms)
        report.update({"atom_mapping": mapping, "residue_transforms": transforms,
                       "neighbor_distance_comparison": neighbor_comparison,
                       "pdb_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "elapsed_seconds": time.monotonic()-started,
                       "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
        files = {names["pdb"]: text}
        files.update(store_atom_data(report,args.report_atom_data,Path(str(args.output_prefix)+'.atomdata.npz')))
        extra_profiles(files, names, edges, target.reference_on_edges, profiles, metrics["generated_profiles"],
                       smoothing_enabled=analysis.smoothing_sigma > 0, target_label="Step target" if target.step else "Smoothed target")
        timings["export_validation"] = time.monotonic()-export_started
        report["elapsed_seconds"] = time.monotonic()-started
        LOG.info("Serializing JSON report and publishing output files")
        files[names["json"]] = encode_json(report)
        publish(files, args.overwrite, [source]+([args.tolerances_from] if getattr(args,'tolerances_from',None) else []))
        LOG.info('Report: %.3f MB JSON, %.3f MB atom data (%s)',len(files[names['json']])/1e6,
                 sum(len(value) for path,value in files.items() if Path(path).suffix=='.npz')/1e6,args.report_atom_data)
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
    try:
        target = rebuild_target(analysis,saved)
    except MastercapError as exc:
        if options.get('step_profile') or saved.get('target', {}).get('step'):
            raise MastercapError(f'Recomputed step target differs from the generation report: {exc}') from exc
        raise
    edges, profiles = target.edges, target.profiles
    verify_saved_step(target, saved)
    verify_geometry_scaling(target,saved)
    configure_tolerances(args,analysis,edges,settings_from_generation(saved),config)
    config['profile_tolerance']=args.profile_tolerance;config['composition_tolerance']=args.composition_tolerance
    center = analysis.center if config["center"] == "reference" else np.zeros(3)
    atom_data=load_atom_data(saved,args.report,getattr(args,'atom_data',None))
    if atom_data.mapping is None:
        LOG.warning('Report has no per-atom data; provenance checked by residue signature only')
    verify_provenance(atom_data,particle,analysis.structure)
    transforms=atom_data.transforms
    atom_protected=[atom_data.source['path']] if atom_data.source.get('path') else []
    inter_limits, inter_report=inter_residue_configuration(analysis,config["overlap_margin"],LOG.warning)
    LOG.info("Validating geometry, connectivity and radial profiles")
    geometry = validate_geometry(particle, center, edges, analysis.limits, config["overlap_margin"], saved["effective_mode"],metadata=final_metadata(particle,analysis))
    audit_report = {}
    try:
        final_connectivity_check(particle, config["overlap_margin"], geometry, audit_report)
        final_pair_distance_check(particle, analysis.limits, audit_report,inter_limits)
    except MastercapError as exc:
        failed = base_report(args, started)
        failed.update(status='failed', error=str(exc), geometry=geometry, **audit_report,
                      validated_pdb=str(args.pdb), reference=analysis.structure.provenance,
                      warnings=analysis.warnings,
                      distance_limits={'-'.join(pair):limit for pair,limit in analysis.limits.items()},
                      distance_limit_overrides=analysis.distance_limit_overrides)
        failed['atom_data']=atom_data.source
        add_diagnostics(failed,particle,analysis,center,target,args.mesh_element_sizes,transforms)
        path = Path(str(args.output_prefix)+'.diagnostic.json')
        publish({path:encode_json(failed)},args.overwrite,[source,args.pdb,args.report]+atom_protected)
        LOG.error('Validation failed; audit diagnostic: %s',path)
        raise
    metrics = final_profile_metrics(particle, center, target, config.get('refine_bin_dither',1))
    log_profile_errors(metrics)
    status = acceptance(geometry, metrics, config["profile_tolerance"], config["composition_tolerance"])
    log_acceptance(status, geometry, metrics, config['composition_tolerance'])
    report = base_report(args, started)
    report.update({"status": status, "geometry": geometry, "metrics": metrics, **audit_report,
                   "validated_pdb": str(args.pdb), "reference": analysis.structure.provenance,
                   "warnings": analysis.warnings, "distance_limit_overrides": analysis.distance_limit_overrides,
                   "distance_limits": {"-".join(k): v for k, v in analysis.limits.items()}})
    if target.step is not None:
        report['target'] = {'center': center.tolist(), 'edges': edges.tolist(),
                            'profiles': {e:p.tolist() for e,p in profiles.items()}, 'step': target.step}
    report["inter_residue_limits"]=inter_report
    report['atom_data']=atom_data.source
    if target.thickness_method is not None:report.setdefault('target',{})['thickness_method']=target.thickness_method
    if saved.get('geometry_scaling'):report['geometry_scaling']=saved['geometry_scaling']
    add_diagnostics(report, particle, analysis, center, target, args.mesh_element_sizes, transforms)
    files = {names["json"]: encode_json(report)}
    extra_profiles(files, names, edges, target.reference_on_edges, profiles, metrics["generated_profiles"],
                   smoothing_enabled=analysis.smoothing_sigma > 0, target_label="Step target" if target.step else "Smoothed target")
    LOG.info("Publishing validation report: %s", names["json"])
    publish(files, args.overwrite, [source, args.pdb, args.report]+atom_protected+([args.tolerances_from] if args.tolerances_from else []))
    LOG.info("Validation status: %s", status)
    return 1 if status == "failed" else 0 if status == "valid" else 2


def run(args):
    started = time.monotonic()
    resolve_target_options(args)
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
    names = artifacts(args.output_prefix, args.csv, args.plot, args.command == "generate",
                      args.command=='generate' and args.report_atom_data=='sidecar')
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
        if args.atom_data:protected.append(args.atom_data)
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
        target = configured_target(args, analysis)
        edges, profiles = target.edges, target.profiles
        settings = target_settings(args, analysis, edges)
        if settings['resized'] and not args.step_profile:
            if args.floor_basis_explicit and args.floor_basis != 'counting':
                raise MastercapError('orientation floors are not defined for resized layered targets; use --floor-basis counting')
            args.floor_basis='counting'
            LOG.info('Resized layered calibration: self/orientation/placement skipped; only counting sets tolerances')
        if target.step is not None:
            LOG.info('Step calibration: reference vs step is model mismatch, not a floor; '
                     'orientation/placement skipped; only counting sets tolerances')
        def level_progress(name,stats):
            LOG.info('Calibration %s: %s','reference vs step' if name=='self' and args.step_profile else name,stats['skipped'] if 'skipped' in stats else ', '.join(f"{e}={value['mean']:.6g}" for e,value in stats.items()))
        calibration=calibrate(analysis,edges,profiles,settings,args.replicates,args.seed,args.compare,args.generated,
                              args.floor_multiple,args.floor_basis,args.floor_statistic,level_progress)
        calibration['suggested']['tolerances_from_line']=f'--tolerances-from {shlex.quote(str(names["json"]))}'
        LOG.info('Suggestion rule: %g x %s %s; experimental comparisons %s', args.floor_multiple, args.floor_basis,
                 args.floor_statistic, 'excluded for step targets' if args.step_profile else 'may raise tolerances')
        LOG.info('Suggested: %s',calibration['suggested']['paste_line'])
        LOG.info('Equivalent: %s',calibration['suggested']['tolerances_from_line'])
        for name,rows in [('compare',calibration['compare']),('generated',calibration['generated'])]:
            for row in rows:
                log_warning_summary(row['warnings'])
                LOG.info('Calibration %s %s: %s',name,row['file'],', '.join(f'{e}={v:.6g}' for e,v in row.get('errors',row.get('shape')).items()))
        report=base_report(args,started)
        report.update(status='calibration_complete',calibration=calibration,warnings=analysis.warnings,acquisition=acquisition)
        if target.geometry_scaling is not None:report['geometry_scaling']=target.geometry_scaling
        if target.thickness_method is not None:report.setdefault('target',{})['thickness_method']=target.thickness_method
        if target.step is not None:report['target']={'step':target.step, 'edges':edges.tolist(), 'profiles':{e:p.tolist() for e,p in profiles.items()}}
        files={names['json']:encode_json(report)}
        if args.csv:files[names['csv']]=calibration_csv(calibration)
        if args.plot:
            files[names['plot']]=calibration_png(calibration)
            LOG.info('Calibration PNG: %s; log error scale (left is smaller); bars=p05-p95, dots=means, '
                     'diamonds=--compare shape errors, triangles=--generated errors, dashed line=suggested tolerance; Total has no threshold', names['plot'])
            if not args.generated:
                LOG.info('No generated structures supplied: PNG contains reference resampling baselines only '
                         '(plus any --compare samples); rerun calibrate with --generated particle.pdb to plot a particle')
        publish(files,args.overwrite,protected)
        LOG.info('Calibration report: %s',names['json'])
        return 0
    if args.command == "generate":
        try:
            target = configured_target(args, analysis)
            configure_tolerances(args,analysis,target.edges, target_settings(args, analysis, target.edges))
        except MastercapError as exc:
            report=base_report(args,started)
            report.update(error=str(exc),analysis=analysis.summary())
            diagnostic(args,report,protected)
            raise
        return generate_command(args, analysis, source, acquisition, started, names, preparation_timings, target)
    report = base_report(args, started)
    report.update({"status": "analyzed", "analysis_options": analysis_options(args),
                   "analysis": analysis.summary(), "acquisition": acquisition})
    files = {names["json"]: encode_json(report)}
    target = configured_target(args, analysis)
    if target.step is not None:report['step'] = target.step
    files[names['json']] = encode_json(report)
    extra_profiles(files, names, target.edges, target.reference_on_edges, target.profiles,
                   smoothing_enabled=analysis.smoothing_sigma > 0, target_label="Step target" if target.step else "Smoothed target")
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
