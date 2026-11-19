# mastercap

Reference-driven generation of spherical geometric capsid particles in Python.
Particles approximate the radial element density and residue composition of an
**already expanded, complete protein capsid**. They are for geometry and
visualization, not chemically valid protein design or molecular dynamics.

**Status:** initial implementation, tested on controlled synthetic structures.
CCMV 1CWP calibration, full-capsid performance, and VMD compatibility are pending.
The bounded heuristic search may fail to find a feasible particle.

## Installation

Linux, Python 3.11 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
mastercap --version
python -m pytest
```

NumPy, SciPy and Matplotlib are runtime dependencies. pytest is for development.
`python -m mastercap` is equivalent to `mastercap`.

To require prebuilt dependency wheels, use this installation command instead:

```bash
python -m pip install --only-binary=:all: -e '.[dev]'
```

### Updating an existing installation

From the repository root, save any local edits before switching branches. These
commands retrieve changes merged into GitHub's `main` branch, activate the existing
virtual environment, and refresh the installed package and dependencies:

```bash
git status --short
git switch main
git pull --ff-only origin main
. .venv/bin/activate
python -m pip install -e '.[dev]'
# Optional: rebuild Python bytecode for the updated source.
python -m compileall -q src/mastercap
mastercap --version
python -m pytest
```

The wheel-only installation alternative above also applies when updating.
mastercap is pure Python and needs no separate native compilation step. Editable
installation uses the source in this checkout; reinstalling refreshes dependency
requirements, package metadata, and the CLI entry point. Python normally compiles
bytecode automatically, so `compileall` is optional. Start a new mastercap process
after updating to load the new code.

## Workflows

All distances are in angstroms and densities in atoms/angstrom³. All interfaces
and reports are in English.

```bash
# Analyze local PDB/VDB content, including gzip-compressed files.
mastercap analyze --input reference.pdb --output-prefix results/reference --csv --plot

# Download a complete VIPERdb capsid into the cache.
mastercap analyze --id 1cwp --cache-dir .cache/mastercap --output-prefix results/1cwp

# ANALYZE -> GENERATE -> WRITE/REREAD -> VALIDATE in one operation.
mastercap generate --input reference.pdb --output-prefix results/particle \
  --seed 42 --allow-approximate --csv --plot

# Optional size and construction changes.
mastercap generate --input reference.pdb --output-prefix results/resized \
  --inner-radius 100 --outer-radius 130 --mode residue --distribution regular \
  --seed 42 --time-limit 600 --allow-approximate

# Independent validation requires the reference and matching JSON report.
mastercap validate --input reference.pdb --pdb results/particle.pdb \
  --report results/particle.json --output-prefix results/validation

mastercap generate -h
```

`generate` includes analysis and validation automatically. The other commands
support analysis alone and independent revalidation. Validation recomputes the
reference analysis, checks hashes and atom provenance, and uses saved settings.

Downloads use `https://viperdb.org/resources/OLIGOMERS/{id}_full.vdb.gz` with TLS
verification enabled. `--refresh` updates a cached file. gzip integrity and ATOM
presence are checked. Cloud network settings must allow `viperdb.org`.

## Input selection

* Only recognized protein residues in ATOM records of the selected model are used.
  `--model` defaults to 1. All protein chains are used.
* Occupancy is ignored; every accepted ATOM counts once.
* HETATM, water, nucleic acids and unrecognized residues are excluded with warnings.
  Modified protein residues encoded as HETATM are also excluded.
* Alternate locations are errors. Missing atoms/hydrogens are not reconstructed.
* Missing elements are inferred from protein atom/residue names, with each
  inference reported. Unknown elements and missing vdW radii are errors.
* The fixed Bondi table covers H, C, N, O, F, P, S, Cl, Se, Br and I:
  Bondi, *J. Phys. Chem.* 68 (1964), 441–451, DOI `10.1021/j100785a001`.
* Coincident reference atoms are rejected during distance analysis.

Protein recognition uses an explicit residue-name list, not a complete chemical
component database. No biological assembly expansion is performed. Angular
coverage warnings cannot prove that a capsid is complete.

Reference-analysis warning details, including every inferred element, are kept
only in JSON (`analysis.warnings` for analysis/generation and diagnostics;
`warnings` for standalone validation). The terminal shows one summary line with
the total warning and element-inference counts, including in `--verbose` mode.
`--quiet` suppresses this summary. Runtime errors and search progress retain
their existing behavior.

## Geometry and profiles

The reference center is the unweighted coordinate mean before radial trimming.
Radii bound **atomic centers**, not vdW surfaces. Defaults use radial extrema;
`--radius-method percentile --radius-percentiles LOW HIGH` selects an interval
and reports excluded atoms. Excluded atoms no longer contribute to profiles or
the selected inventory; trimming may leave partial residues.

`--bin-width` defaults to 1 A. Uniform bins have widths no greater than requested;
densities use exact spherical shell volumes. Gaussian smoothing is on by default
with a **provisional, uncalibrated** sigma of 1 A. Use `--smoothing-sigma` or
`--no-smoothing`. Renormalization conserves integrated element counts, confines
the smoothed profile to the interval, and keeps densities nonnegative.

Either radius may be overridden. Profiles map by relative position through the
thickness while retaining local number density; atom counts change with volume.
Different element profiles can produce changed global fractions after resizing.
Integer residue inventories approximate these objectives and report differences.
No manual density/composition multipliers are available.

`--center origin` moves the target center to zero; the default preserves the
reference center. The exported coordinate mean must be within **1 A** of it.

## Construction

`--mode independent` moves individual atoms while preserving residue membership,
observed composition, names and provenance. Residues may deform and do not imply
valid chemical bonds. `--mode residue` preserves each observed template rigidly.
Inventories approximate residue frequencies, permit repeated templates when
enlarged, and use integer optimization for element/variant count errors.

`--distribution random` initializes isotropic residue-center directions and
volume-weighted radial positions. `regular` uses Fibonacci directions and radial
layers. Repair may move these positions; a perfect final lattice is not promised.

The repair search uses spatial neighbors, repulsion, component contacts, radial
transport, and center correction. Contacts are selected at nearby component
boundaries, filtered for minimum-distance/overlap compatibility, and reduced to
a sparse minimum spanning forest of the proposals. Dense components without a
local outgoing proposal use external-component queries. There are no contacts
added solely by component-ID order. The same compatibility filter applies inside
residues. The contact forest proposes moves; it is not proof of connectivity.
Rigid updates use Kabsch projection. Mandatory
repairs precede soft profile improvement. The search is heuristic, not a physical
simulation, and does not guarantee convergence or final angular isotropy.

Every exported particle must satisfy after PDB rounding and rereading:

1. All centers within the requested radial bounds.
2. Center error at most 1 A.
3. Applicable minimum distances.
4. Internal connectivity of every residue.
5. Global connectivity and no isolated atoms.

An overlap requires `distance < vdW_radius_i + vdW_radius_j - margin`.
Tangency is insufficient. `--overlap-margin` defaults to zero.

Minimum distances use a nearest-neighbor percentile for each element pair
(`--distance-percentile`, provisional default 1). Both directions contribute for
different elements. Pairs with fewer than `--min-pair-samples` observations
(provisional default 100) use a pooled percentile of distances normalized by vdW
radius sums. Fallbacks are reported. Incompatible pairs cannot provide connections.

Rigid internal distances are preserved even below inferred limits, with
exceptions reported. Required internally disconnected rigid templates cause
failure. `--allow-independent-fallback` explicitly authorizes a retry with
independent atoms and the remaining repair-search time budget.

`--max-iterations` bounds repair steps per attempt. `--time-limit` (default 300 s)
is a **repair-search** budget: download, parsing, analysis, inventory selection,
and initialization are excluded. Inventory optimization has a separate 15-second
solver limit. Rigid and explicitly authorized independent fallback attempts share
the repair budget; a fallback does not receive a fresh full budget. The limit is
cooperative: an individual spatial query may finish after the deadline.

Reports contain `timings_seconds` for reference acquisition, analysis (including
parsing), inventory, initialization, preparation overall, repair search, and
export/validation. Preparation is an aggregate, not an additional phase to sum
with its subphases. `search.elapsed_seconds` measures only the reported attempt's
repair search; `search.preparation_seconds` measures its initialization/metadata
setup. `timings_seconds.search` includes both attempts when fallback occurs.
`search.termination` is `time_limit`, `iteration_limit`, or `constraints_satisfied`.
`iterations` counts completed repair updates; `candidate_iteration` identifies
the selected feasible candidate, which can precede termination. Search may end
at a limit while still returning a previously feasible approximate candidate.
v1 does not resume interrupted runs. Search failure is not proof of mathematical
infeasibility.

Static element/radius/residue metadata is cached per inventory. Each search
iteration computes one neighbor/overlap graph on PDB-rounded coordinates and
reuses it for checking and proposing repairs. Final export still independently
rereads the written PDB and rebuilds its geometry checks.

## Acceptance and reports

JSON is mandatory. CSV profiles (`--csv`) and PNG plots (`--plot`) are optional.
Reports include original, smoothed and generated profiles; element counts;
integrated profile errors; geometry and graph checks; nearest-distance percentile
comparisons; provenance; residue transformations/deformation RMSD; seed; software
versions; elapsed time; and Linux process peak RSS.

The primary per-element profile error is:
`sum(abs(generated_density - target_density) * shell_volume) / target_count`.
Comparisons against the mapped original unsmoothed profile are also included.

No calibrated profile/composition tolerances exist yet. Set `--profile-tolerance`
and `--composition-tolerance` for operational acceptance. The former applies to
each element; the latter allows relative count error with a one-atom rounding
allowance. Missing tolerances produce `uncalibrated`; exceeding supplied soft
tolerances produces `approximate`. Both require `--allow-approximate` to save.
**That flag never relaxes mandatory constraints.**

| Exit code | Meaning |
|---|---|
| 0 | Analysis completed, or generation/validation valid within supplied tolerances. |
| 1 | Input, search, validation or output failure. |
| 2 | Approximate/uncalibrated result saved or verified; also argparse usage errors. |
| 130 | Interrupted. |

Failures write a separate `.diagnostic.json` when possible, without publishing a
new PDB. Existing outputs require `--overwrite`. All output content is staged;
promotion errors trigger rollback. Individual files are atomic, but the group
is not crash-atomic. No incomplete PDB replaces a prior successful output.

## PDB convention and VMD

One file contains ATOM, TER and END, without MODEL or CONECT. Residues are never
split across segments. Serial/residue numbering restarts as needed and chain
characters may be reused. This repeated-serial convention is **not strict global
PDB identifier uniqueness**; JSON mapping by record order is authoritative.
Coordinates outside PDB 8.3 field capacity are rejected.

The internal reader round-trips more than 99,999 atoms. VMD must independently
verify complete atom count, residue identity and coordinates; this has not been
tested here. VMD's displayed bonds are not the program's vdW overlap graph.

An external acceptance script is supplied for an installed VMD:

```bash
vmd -dispdev text -e scripts/check_vmd.tcl -args results/particle.pdb
```

It checks one frame, atom count, names, elements, residue grouping and coordinates.
The script itself remains unverified until it can run in VMD.

## Reproducibility and remaining validation

A supplied seed is recorded; otherwise one is generated. Reproducibility requires
matching reference content, parameters, program and numerical library versions.
Cross-platform byte identity is not promised. Budget-limited MILP optimization
may also affect reproducibility; avoid solver timeouts for repeatable inventories.

Automated tests cover parsing, models/gzip, filtering, element inference, profiles,
volume scaling, distance fallback, global/internal connectivity, strict overlap,
PDB rounding, construction modes, regular initialization, reproducibility,
independent validation, diagnostics, large numbering, PNG/CSV and output rollback.

Before a full-capsid release we still need the expanded 1CWP file, calibration of
smoothing/tolerances, complete-capsid timing and memory measurements, convergence
assessment, and an installed VMD for compatibility testing. Small controlled
cases alone do not establish readiness at viral-capsid scale.

A repeatable synthetic timing probe is available:

```bash
python benchmarks/search_shell.py --residues 600 --iterations 30 --repeats 3
```

It reports timings and remaining violations on a carbon-only shell. This measures
iteration cost and is not a demonstration of full-capsid convergence.
