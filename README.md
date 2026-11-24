# mastercap

**Current version: 0.1.12**

Reference-driven generation of spherical geometric capsid particles in Python.
Particles derive radial element targets and residue composition from an
**already expanded, complete protein capsid**, with an optional homogeneous
step-profile control wall. They are for geometry and
visualization, not chemically valid protein design or molecular dynamics.

Version 0.1.12 adds packing/wall/layering diagnostics, a seeded wall-band clamp,
faster quota-limited edge transport at fine radial bins, and optional scaled
non-bonded inter-residue distance floors. The reference/step targets and
calibration definitions are unchanged. Use `--wall-clamp legacy
--refine-drift-step legacy` for the earlier search behavior.

**Status:** geometric prototype with synthetic regression tests, complete 1CWP
benchmarks and reference-resampling calibration. Independent experimental
validation and broader capsid coverage remain pending.
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

## Command-line help

`mastercap -h` (or `mastercap --help`) prints the complete reference for all four
commands: every option, its meaning, concrete or inferred/unset defaults, accepted
choices, units, examples, output files, acceptance rules and exit codes.

For a shorter reference limited to one command, use:

```bash
mastercap analyze -h
mastercap generate -h
mastercap validate -h
mastercap calibrate -h
mastercap --version
```

Help does not read coordinate files or download data. Quality tolerances remain
unset until explicitly supplied; the examples use `--allow-approximate` when
saving mandatory-feasible, uncalibrated particles.

## Workflows

All distances are in angstroms and densities in atoms/angstrom³. All interfaces
and reports are in English.

```bash
# Analyze local PDB/VDB content, including gzip-compressed files.
mastercap analyze --input reference.pdb --output-prefix results/reference --csv --plot

# Download a complete VIPERdb capsid into the cache.
mastercap analyze --id 1cwp --cache-dir .cache/mastercap --output-prefix results/1cwp

# Measure reference-resampling error scales and suggest tolerances (no PDB).
mastercap calibrate --input reference.pdb --output-prefix results/calib --csv --plot

# Generate using the suggestions; reference, model, bins, smoothing and radii must match.
mastercap generate --input reference.pdb --output-prefix results/calibrated_particle \
  --tolerances-from results/calib_calibration.json --seed 42 --csv --plot

# ANALYZE -> GENERATE -> WRITE/REREAD -> VALIDATE in one operation.
# Without tolerances, explicitly permit an uncalibrated result.
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

`generate` includes analysis and validation automatically. `analyze` measures the
reference, `calibrate` estimates profile error scales and suggests tolerances,
and `validate` independently rechecks a particle. Calibration is optional and
must be invoked separately to establish tolerances. Validation recomputes the
reference analysis, checks hashes and atom provenance, and uses saved settings.

Downloads use `https://viperdb.org/resources/OLIGOMERS/{id}_full.vdb.gz` with TLS
verification enabled. `--refresh` updates a cached file. gzip integrity and ATOM
presence are checked. Cloud network settings must allow `viperdb.org`.

The downloader uses Python's `urllib.request`, not wget. For an explicitly
requested download without certificate verification, use:

```bash
mastercap analyze --id 1cwp --no-check-certificate --refresh --output-prefix results/1cwp
```

`--no-check-certificate` skips TLS server-identity verification only for VIPERdb
downloads; the default continues to verify certificates. It has no effect on a
local `--input`, and does not fix HTTP 403 responses or network-policy denials.
An actual unchecked download prints a warning. Download provenance records the
verification mode and a hash-matched cache receipt; legacy caches have unknown
TLS provenance. A later default request re-downloads a known unchecked cache
using verified TLS. gzip/coordinate checks do not authenticate the remote server.

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
Reference-model radii bound **atomic centers**, not vdW surfaces. Step-mode CLI
radii are half-height bounds and atoms may extend half the ramp width beyond
them (see [step targets](#step-profile-control-targets)). Reference selection
defaults use radial extrema;
`--radius-method percentile --radius-percentiles LOW HIGH` selects an interval
and reports excluded atoms. Excluded atoms no longer contribute to profiles or
the selected inventory; trimming may leave partial residues.

The following smoothing/mapping rules describe the default **reference-shaped
target**; step targets use analytic ramps and no Gaussian smoothing.

`--bin-width` defaults to 1 A. Uniform bins have widths no greater than requested;
densities use exact spherical shell volumes. Gaussian smoothing is on by default
with a **provisional, uncalibrated** sigma of 1 A. Use `--smoothing-sigma` or
`--no-smoothing`. Renormalization conserves integrated element counts, confines
the smoothed profile to the interval, and keeps densities nonnegative.

`--smoothing-sigma` is the Gaussian standard deviation in **A**, not bins. A
higher value smooths more, broadens peaks, and suppresses narrow fluctuations.
The implementation converts it to bin units using the actual bin width.
`--smoothing-sigma 0` disables Gaussian filtering too.

`--no-smoothing` overrides any supplied sigma and uses an exact copy of the
original **binned** profile as the target. The PNG then shows the original mapped
profile and generated profile, omitting the redundant "Smoothed target" curve.
Standalone validation plots honor the effective setting saved during generation.
CSV keeps generic original/target columns, and the JSON's legacy `smoothed` array
remains for compatibility (equal to the original when disabled); the effective
state is explicit in `analysis.smoothing` and `analysis_options.sigma`.

Binning still applies without Gaussian smoothing. For example, `--bin-width 4`
uses coarser layers than 1 A and can hide narrow features or reduce the reported
error. Compare runs at the same bin width. With unchanged geometry, generation
seeks the original binned profile; changing radii maps that profile to the new
interval. Neither option guarantees an exact profile match under integer-count,
residue, distance and connectivity constraints.

```bash
mastercap generate --input ./1cwp_full.vdb --output-prefix results/particle03 \
  --mode independent --distribution regular --bin-width 4 --no-smoothing \
  --time-limit 600 --seed 42 --allow-approximate --csv --plot \
  --progress-interval 5 --refine-updates 50 --selection-objective total
```

For reference-shaped targets, either radius may be overridden. Profiles map by relative position through the
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

### Adaptive profile refinement

After first feasible geometry, the main refinement budget counts **profile
pushes**, not the repair iterations between them. Rolled-back pushes count too.
`--time-limit` and `--max-iterations` remain hard, cooperative search limits.
Setting either `--refine-updates 0` or `--refine-iterations 0` stops at the first
feasible geometry. The best feasible candidate is kept separately and final PDB
validation remains independent.

| Option | Default | Meaning |
|---|---|---|
| `--selection-objective` | `total` | Rank feasible candidates by atom-weighted total error first; `worst-element` restores the previous ranking. |
| `--refine-updates` | 50 | Maximum pushes, including attempts subsequently rolled back. |
| `--refine-iterations` | unset | Optional cap on all iterations since first feasible geometry, including repair and rollback; off by default. |
| `--refine-patience` | 10 | Pushes without significant improvement before stopping; 0 disables this check. |
| `--refine-strength` | 0.005 | Radial strength; fixed by default. |
| `--refine-strength-control` | `fixed` | Keep strength fixed, follow primary profile `gain`, or restore `acceptance` feedback. Applies to per-atom acceptance. |
| `--refine-min-step` | 0.002 A | Raise nonzero requests below this size when the desired distance is at least this large; 0 disables; must not exceed max step. |
| `--refine-selection` | `crossing` | Reserve useful adjacent-bin crossings then drift toward edges; `mismatch` retains v0.1.5 rank matching. |
| `--refine-max-step` | 0.05 A | Cap on the requested radial profile contribution per selected atom. |
| `--refine-fraction` | 0.2 | Per-element crossing/drift budget rounds down; mismatch rounds up. |
| `--refine-recovery-limit` | 10 | Repair updates allowed after damage; 0 restores immediately if the next state is invalid. |
| `--neighbor-forces` | `off` | Disable statistical neighbor forces; `scaled` follows push controls, `fixed` uses coefficient 0.002 on all atoms. |
| `--refine-slack` | 0.02 A | Target distance/contact margin during consolidation and neighbor targeting; final validation limits remain unchanged. |
| `--consolidate-iterations` | 10 | Checked slack-building updates before profile pushes; 0 disables the initial episode. |
| `--consolidate-target` | 0.001 | Each count of pairs/edges within half the requested slack must be at most this fraction of atom count. |
| `--consolidate-noncontact-slack` | `on` | Build full distance-side slack for pairs that cannot form a vdW contact; `off` restores v0.1.6 consolidation for comparison. |
| `--reconsolidate-below` | 0.7 | Restore slack after lower acceptance; 0 disables this trigger. |
| `--reconsolidate-every` | 10 | Restore slack after this many pushes since the last episode; 0 disables. |
| `--reconsolidate-iterations` | 3 | Maximum checked updates per restoration episode; 0 disables restoration. |
| `--refine-acceptance` | `per-atom` | Reject conflicting atom moves before committing; rigid mode rejects whole residues. `rollback` permits repair and snapshot recovery. |
| `--rollback-violation-fraction` | 0.05 | Immediate rollback above this distance-violating-pair count divided by atom count; range [0, 1]. |
| `--rollback-component-limit` | 5000 | Immediate rollback at this many components. |

Before profile pushes, consolidation tries to build `--refine-slack` on both
minimum distances and contact distances. Narrow feasible windows use one third
of their width instead. Consolidation builds only separation/contact forces,
caps each atom's accumulated request at half the slack, and checks a rounded
trial through the same per-atom/residue acceptance path. It consumes iterations
and time but no profile pushes. At least one step runs; it ends when both
near-limit counts meet `--consolidate-target`, their combined count improves by
less than 5% in the last step, or the episode budget ends. A safety interruption
logs a warning and permits at most three restarts. A completed episode does not
imply every pair gained the requested slack. A zero refinement budget stops at
first feasibility before any slack update.

During refinement, low acceptance or the periodic trigger starts another checked
episode, with at least two pushes between episodes. Disable both triggers with
`--reconsolidate-below 0 --reconsolidate-every 0`. The patience counter advances
only on pushes, not on consolidation updates.

The default crossing selection computes current and composition-normalized target
counts in the radial bins. For neighboring surplus/deficit bins, it reserves at
most the floor of the available surplus and deficit, preferring atoms closest to
the shared edge. Requested moves reach 0.002 A beyond that edge, capped at max
step. Remaining budget drifts surplus atoms toward useful edges, using strength
and the minimum/maximum step. Atoms rejected in the previous push are skipped.
Crossings and drift together use at most the floor of fraction times element
count; very small inventories may need a larger `--refine-fraction`.

`--refine-selection mismatch` ranks atoms against deterministic volume-aware
quantiles and selects the largest mismatches as in v0.1.5. Its requested move
is strength times mismatch, raised to the minimum step where eligible, then
capped at max step. With
`--neighbor-forces scaled`, nearest-neighbor contributions use coefficient
`0.002 * (current_strength / starting_strength)` and affect only atoms selected
by the radial push. Their accumulated vector per atom is capped at half
`--refine-max-step`, independently of the radial contribution. `off` skips these
contributions entirely; `fixed` keeps coefficient 0.002 and all-atom coverage
without this additional cap. Both active modes clip pair targets between
`lower + refine_slack` and `sum_of_vdw_radii - overlap_margin - refine_slack`, skipping
empty windows. Thus `fixed` preserves the old coefficient and coverage but also
includes the target safety fix. Target clipping alone does not guarantee that
the combined coordinate update is feasible. The overall 0.2 A update cap and
rigid projection still apply. Each push records scalar nearest-neighbor move
statistics under `push_history[*].neighbor_forces`.

The default acceptance path builds a separate radial/neighbor trial update,
applies the overall cap and rigid projection, and rounds it to PDB precision.
It rejects moves that violate a guarded minimum distance (lower + 0.003 A),
break any existing contact (radius sum - margin - 0.003 A), or leave the radial
bounds. It also checks new nearby pairs and mixed trial/restored positions.
Offenders are reverted for up to three passes, then with their direct neighbors;
a remaining conflict restores the complete trial as a safety fallback. In rigid
mode every rejection restores a complete residue. Preserving existing edges
preserves global and internal residue connectivity without recomputing components
inside the trial check. The next normal iteration and final PDB reread still
validate geometry independently. A center error above 1 A rejects the whole trial.

Per-atom strength is fixed by default. `gain` compares the primary candidate key
before and after each push: two consecutive non-improving pushes reduce strength
by 0.7; a positive gain with acceptance above 85% grows it by 1.25, up to four
times the starting value. `acceptance` restores v0.1.5 feedback, growing above
85% and shrinking by 0.7 below 50%. Only adaptive controls can hit the starting
strength/16 floor. Acceptance counts selected proposals,
including feasible proposals that round to no motion; rigid mode expands them
to complete residues. Physically moving proposals and retained moves are reported
separately. An entirely stationary trial still consumes a push, and patience
can stop it. Zero-accepted proposals also consume a push. Repair,
radial clipping and centering are not added to feasible per-atom trial updates.

Run the full-reference benchmark with the new defaults:

```bash
mastercap generate --input ./1cwp_full.vdb --output-prefix results/revision \
  --mode independent --distribution regular --bin-width 4 --no-smoothing \
  --time-limit 600 --seed 42 --allow-approximate --csv --plot --overwrite
```

The historical Step 1 off/fixed comparison used commit `df200e5`, before
consolidation and per-atom acceptance. It requires the complete 214,440-protein-atom
reference. Uncalibrated successful generation returns exit code 2. Benchmark
measurements and the reference checksum are retained in `benchmarks/1cwp_refinement.json`.

On the complete 1CWP reference (seed 42, regular initialization, 4 A bins,
no smoothing, 600 s budget), version 0.1.5 reached total error **0.060828**
with **57.6%** mean proposal acceptance and **zero rollbacks** in **399.7 s**.
All refinement states after consolidation were feasible. Search stopped after
32 pushes on patience: `termination=refinement_stalled` with
`refinement_stop_limit=refine_patience`, preserving the existing termination
mapping; it did not hit the strength floor. The independently reread PDB passed
mandatory checks. Consolidation exhausted its three restarts, so its target
slack was only partially achieved. Sulfur error remained 0.837; total-error
selection does not guarantee improvement for each element. These measurements
are a reproducible benchmark, not calibrated scientific acceptance tolerances.

The revision-02 A/B comparison changed only reporting and the neighbor default:
`off` reached 0.058776 total error with 62.3% mean acceptance, versus 0.060828 and
57.6% for `scaled`. Both left S at 0.837 and stopped below rounding resolution.
This evidence is why statistical neighbor forces are now off by default.
Final v0.1.6 comparisons use seeds 42, 7 and 2026. For the same-code compatibility
case, add `--refine-selection mismatch --refine-strength-control acceptance
--neighbor-forces scaled --reconsolidate-below 0 --reconsolidate-every 0`.

The complete 1CWP runs used the same 600 s budget and 4 A unsmoothed bins:

| Seed | Default total error | Compatibility total error | Default acceptance | Compatibility acceptance |
|---|---:|---:|---:|---:|
| 42 | 0.051604 | 0.060651 | 83.0% | 57.2% |
| 7 | 0.053973 | 0.063915 | 84.8% | 53.2% |
| 2026 | 0.049935 | 0.059728 | 86.8% | 54.0% |

All default runs finished 50 pushes within 600 s, with no rollbacks and passing
independent PDB validation. Their total-error spread is 0.00404; each run's
improvement below 0.0609 exceeds that spread. **The full scientific benchmark
gate does not pass:** seed 42 S error remains 0.837037. Its 52 crossing proposals
produced no sulfur bin crossings; only 33.4% of S proposals were accepted, with
minimum-distance rejections dominant. C improved for every seed, and S improved
for seeds 7 and 2026. Global acceptance alone therefore does not establish good
refinement for a rare element. Full per-push evidence and explicit pass/fail
checks are in the benchmark JSON. Runs used three workers on a four-core quota
with BLAS threads limited to one; reported search time includes that environment.

A separate matched-checkpoint probe raised `--refine-slack` to 0.05 A after the
same first-feasible seed-42 state. Its 50 pushes reached total error 0.049767,
but S still had zero bin crossings and error 0.837037. S acceptance rose only to
40.3%. This probe skips initial repair and is explicitly not another full CLI
benchmark; it does not establish that a larger global slack resolves the local
sulfur blockage. That revision-02 scientific gate remains open in the historical
results; the revision-03 measurements below test a different initialization.

Revision 03 benchmarks compare the measured S-S spacing (`S-S=reference`), the
v0.1.7 pooled steric limit, and opt-in sulfur-profile placement. The default
placement remains `off`. For an exact v0.1.6 comparison also pass
`--consolidate-noncontact-slack off`, since consolidation slack is a separate
changed default. In that v0.1.7 benchmark, both compatibility switches reproduce the seed-42
PDB byte for byte. The v0.1.8 inter-residue floor is an additional constraint. Independent validation also accepts the old v0.1.6 report
using its reference limits.

Profile placement deliberately copies reference S shells. Zero S profile error
therefore measures retention of that initialization, rather than independent
molecular validation. It can increase initial repair time and trade a small
increase in another element's error for lower total error. Runs that reach the
cooperative time limit may complete fewer than 50 pushes; a feasible candidate
still undergoes the mandatory final audit and independent validation. The JSON
retains per-element comparisons and stop reasons instead of treating aggregate
improvement as proof that every element improved.

The unsmoothed v0.1.7 comparisons use the same 1CWP reference, 4 A bins and
600 s search budget. Both columns use the v0.1.7 pooled S-S limit:

| Seed | Placement off: total / S error | Profile placement: total / S error | Profile pushes | Profile stop |
|---|---:|---:|---:|---|
| 42 | 0.051408 / 0.781481 | 0.048909 / 0 | 38 | time limit |
| 7 | 0.053721 / 0.811111 | 0.049767 / 0 | 25 | time limit |
| 2026 | 0.049888 / 0.851852 | 0.049599 / 0 | 50 | refinement complete |

All 540 anchored S atoms remain in their assigned bins in these three profile
runs. Each written PDB and standalone validation passes the independent audit
(zero isolated atoms, one component, zero disconnected residues).
**The strict no-C/N/O-regression expectation does not pass:** C is slightly
worse for seed 42, N and O for seed 7, and C and O for seed 2026. These are geometric,
uncalibrated outputs; lower total error does not establish chemical realism.

The additional seed-42 profile-placement run with default 1 A smoothing reached
total error 0.048623 and S error 0.001207, retaining all 540 anchors in their
assigned bins. It stopped at the time limit after 40 pushes. Its target differs
from the unsmoothed cases, so these errors are not a matched smoothing A/B test.
All eleven revision-03 runs passed both the written-PDB audit and standalone
validation with their saved distance overrides; outputs remain `uncalibrated`.

The nine unsmoothed v0.1.8 benchmark runs use a 1800 s budget and all complete
50 pushes. These comparisons share 4 A bins, regular initialization and the new
S-S steric limit. All six placed runs retain S error 0:

| Seed | Off total error | Profile total error | Reference total error | First feasible (off / profile / reference) |
|---|---:|---:|---:|---|
| 42 | 0.051408 | 0.051660 | 0.051371 | 62 / 455 / 201 |
| 7 | 0.053805 | 0.049599 | 0.047892 | 64 / 248 / 222 |
| 2026 | 0.049907 | 0.048517 | 0.050028 | 68 / 121 / 293 |

**The strict scientific expectations do not all pass.** Every placed run has at
least one C/N/O error higher than its seed-matched control at 50 pushes. Total
error improves in four of six comparisons, but worsens for seed-42 profile and
seed-2026 reference placement. For seed 42 profile, the C/N/O error changes are
+0.004814 / -0.010283 / +0.004078, and total error increases by 0.000252 despite
S reaching zero. Full per-element comparisons at pushes 10/20/30/40/50 are in
`revision04.paired_comparisons` in the benchmark JSON; initial errors and repair
history are retained as well.

The predeclared seed-42 spring controls use zero profile pushes and an extended
1800 s diagnostic budget to observe first feasibility:

| Spring control | First feasible iteration | Search seconds | S error |
|---|---:|---:|---:|
| Exact radius, tolerance 0 | 531 | 840.4 | 0 |
| Dead zone, tolerance 1.5 A | 455 | 718.5 | 0 |
| Disabled diagnostic control | 63 | 104.1 | 0.003704 |

The dead zone reduces iterations about 14% versus exact anchoring, but misses
the expected near-80 result. Both enabled controls exceed the old 600 s budget.
The disabled control lets one S leave its assigned bin. At iteration 100, a
coordinate probe finds one ASP split into N, CA and a six-atom fragment; its N
is 3.49 A from the one CYS S outside its dead zone. This suggests coupled local
packing, rather than proving that all S atoms are overconstrained. The full
seed-42 profile run repeats first feasibility at iteration 455 under a different
workload. Reference placement reports 45 distinct target radii at 1e-6 A and nine
shells separated by > 0.01 A, preserving the actual finite-precision reference.

The smoothed seed-42 profile case (sigma 1 A) also completes 50 pushes: total
error 0.053226, S error 0.001207, first feasibility at iteration 435. Its target
differs from the unsmoothed cases, so it is not a matched smoothing A/B test.
All ten main runs stop with `refinement_complete`; none hits the clock or
iteration limit. Search times range from 353.0 to 1139.8 s in this environment.
All thirteen PDBs, including the three spring controls, pass both independent
audits and standalone validation with matching saved numeric overrides. The
smallest inter-residue S-S distance across the main matrix is 2.789399 A, above
the 2.786177 A limit; no inter-residue covalent-floor violation is reported.
Outputs remain `uncalibrated` (exit 2), not scientifically calibrated structures.
Runs used three workers on a four-core quota with one BLAS thread per process;
brief diagnostic overlap reached four processes. Timings include that workload.

For revision-04 benchmarks, give the push budget time to finish:

```bash
mastercap generate --input ./1cwp_full.vdb.gz --output-prefix results/revision04 \
  --mode independent --distribution regular --bin-width 4 --no-smoothing \
  --sulfur-residue-placement profile --sulfur-anchor-tolerance 1.5 \
  --time-limit 1800 --seed 42 --allow-approximate --csv --plot --overwrite
```

Compare `search.errors_at_push` at equal 10/20/30/40/50 attempted pushes, plus the
last reached push. Each checkpoint reports the best feasible total and per-element
errors, not the potentially damaged last search state. Keep first-feasible
iteration, total iterations and elapsed seconds separate. A fixed seed controls
random draws but cannot reproduce how many pushes fit into a wall-clock budget;
`time_limit` termination emits an explicit warning. `benchmarks/search_shell.py`
also supports `--time-limit` (default 1800 s) and retains these checkpoints; it
remains a synthetic timing probe, not independent full-capsid validation.

With `--refine-acceptance rollback`, a full coordinate snapshot is saved before
each push, including rigid residues. Damage below the configured disaster limits
gets up to `--refine-recovery-limit` repair updates. Recovery is compared against
the snapshot's complete ranking key: an improved state is retained; an equal or
worse state restores the snapshot and halves strength. Immediate recovery with
improvement grows strength by 1.25. A disaster or exhausted repair allowance also
restores and halves. Restored snapshots are tracked separately and cannot count
again as new feasible/best candidates. If strength falls by at least fourfold
but damage falls by less than 25%, the controller stops halving and terminates
with `refinement_blocked` / `damage_independent_of_push`, logging the evidence.
The best candidate is always a separate rounded copy. Falling below
1/16 of starting strength stops with `refinement_stalled`; no push uses that
below-floor strength. This is a controller stop, not proof of infeasibility.

`total` ranks total error, then mean element error, then maximum element error,
then count difference. `worst-element` ranks maximum, mean, total, then count
difference. Both preserve per-element measurements, including S. A significant
improvement resets patience when the primary ranking error decreases by more than
0.1% relative to its last significant improvement (absolute floor 1e-12).
Smaller improvements can still replace the best candidate without resetting
patience. Sparse elements and coarse bins produce plateaus; increase patience or
set it to 0 when investigating those cases. Quality tolerances can stop search
earlier, and no setting guarantees exact profile matching.

Every exported particle must satisfy after PDB rounding and rereading:

1. All centers within the target support (requested radii for reference-shaped
   targets; half-height bounds extended by half the ramp width for step targets).
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

`--pair-limit-override PAIR=VALUE` is repeatable on analyze, generate and validate.
Pairs are normalized (N-C and C-N are the same); duplicates and unknown elements
are errors before input is read. VALUE is a positive finite distance in A or:

- `steric`: for each participating element, find its smallest measured limit /
  vdW-radius-sum ratio among non-overridden pairs without pooled fallback; use
  the smaller element ratio times this pair's vdW-radius sum. Missing source
  evidence for an element requires a numeric override.
- `pooled`: the v0.1.7 global pooled ratio times the vdW-radius sum.
- `reference`: retain the measured percentile.

**S-S=steric is the default.** On 1CWP it uses C-S's ratio 0.773938 and gives
**2.786177 A**, replacing the 14.068 A symmetry-spacing statistic. The override
record includes `ratio` and `source_pair` as well as original/applied values,
mode and reason. Other measured limits and neighbor samples remain unchanged;
statistical neighbor forces skip non-reference overridden pairs.

Every new override must be at least `0.9 * (covalent_radius_a + covalent_radius_b)`.
The fixed single-bond covalent radii use Pyykko/Atsumi (2009); S-S's floor is
1.854 A. Numeric overrides below the floor fail before input is read. Derived
steric/pooled/reference limits are checked after reference analysis, since their
values depend on that input. Thus `S-S=pooled` preserves the old formula but its
1CWP value of 1.371 A is rejected for new generation. There is no bypass flag.

Independent validation reapplies each saved record's **numeric `new_value`**,
never the current interpretation of its mode. A v0.1.7 steric record without
`ratio` produces a compatibility warning and retains its saved 1.371 A limit.
The final distance audit still rejects actual sub-covalent inter-residue pairs:
a legacy limit is readable, but does not exempt its coordinates from that check.
Failed standalone audits save `PREFIX.diagnostic.json` with measured distances
and named atoms; they preserve the original PDB and generation report.
Reports predating override records explicitly use reference limits. Analysis is
rebuilt from the reference; there is no separate saved-analysis loader.

### Optional sulfur residue placement

`--sulfur-residue-placement profile` anchors the first S atom in every
S-containing residue, selected by element rather than residue name. It retains
seeded rotations and assigns volume-quantile S radii in a seeded random order;
regular initialization uses bin midpoints, random uses the quantiles directly.
A bounded greedy pass swaps crowded S residues' directions with non-S residues'
directions, targeting separation of S-S limit + 2 A. Each conflict tries at most
50 seeded candidates; unresolved planned conflicts are counted and warned about.
The exact positive-root offset equation then places the rotated S atom at its
target radius. Remaining residues sample aggregate density minus the placed
residues' contribution. Target radii and bin counts are unchanged by direction swaps.

`--sulfur-residue-placement reference` instead uses each anchored atom's own
reference radius via its inventory provenance, without midpoint quantization or
radius shuffling. Those radii must lie within the requested target bounds;
use profile placement when resizing to incompatible bounds. Reference mode
preserves the actual coordinate-derived radii. On the uploaded 1CWP file there
are 45 distinct values rounded to 1e-6 A, forming nine shells with gaps > 0.01 A;
reports give both counts rather than pretending rounded coordinates have nine
exactly equal values. Both placement modes copy reference radial structure by design.

`--sulfur-anchor-tolerance` defaults to 1.5 A. Before first feasible geometry,
a weak 0.05 spring acts only outside `[target - tolerance, target + tolerance]`,
clipped inside the atom's assigned bin by 0.25 A per edge. Bins narrower than
1 A use a quarter-bin padding; empty intersections collapse to the nearest
safe point. Tolerance 0 restores exact-radius anchoring for comparison. The
spring uses rigid projection in residue mode and stops at first feasibility.
During consolidation and refinement, checked moves cannot take an anchored S
out of its assigned bin (`anchor_bin` rejection); crossing selection also filters
such proposals. No spring acts in trial profile pushes.

The default `off` and S-free paths preserve v0.1.6 initialization coordinates and
RNG draws. The stricter v0.1.8 distance constraints can change subsequent repair. A summary warns about multi-S residues, where only the first S is
anchored. `search.sulfur_placement` records spacing conflicts, target shells,
nonzero spring updates, bin rejections, profile errors and anchor-bin retention.
`repair_history` contains one scalar summary per progress interval: top five
disconnected residue names and violating element pairs, anchored participation,
and anchors outside their dead-zone intervals. The first-feasible repair log's
percentage uses disconnected-residue observations across those samples; it is
compared with the anchored share of the residue inventory. Private anchor arrays
are not serialized into JSON.

When contact distance minus minimum separation is nonpositive, consolidation
now builds full distance-side slack and never applies a contact pull. Positive
narrow windows still use one third of their width. Disable this optional change
with `--consolidate-noncontact-slack off` when isolating v0.1.6 behavior.

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
`search.termination` is `time_limit`, `iteration_limit`, `constraints_satisfied`,
`refinement_complete`, `refinement_stalled`, or `refinement_blocked`. Completion means a push/optional
iteration budget was used, not that scientifically calibrated criteria passed;
`refinement_stop_limit` identifies the budget or stall condition. Checks prioritize
supplied quality acceptance/first-feasible stopping, refinement budgets, blocking, stalling,
time limit, then iteration limit. `iterations` is the last evaluated search
iteration, including rollback-only iterations; `candidate_iteration` identifies
the selected feasible candidate, which can precede termination. Search may end
at a limit while still returning a previously feasible approximate candidate.
v1 does not resume interrupted runs. Search failure is not proof of mathematical
infeasibility.

INFO messages announce reference acquisition, reading, analysis, inventory,
initialization, repair/refinement, final PDB checks, plotting and JSON output.
Progress is printed at iteration 0, every **5** iterations by default, and at
termination. Use `--progress-interval N` (positive integer) to change the cadence;
`--quiet` suppresses stage/progress messages. Progress includes isolated atoms,
disconnected residues and worst-element error alongside the aggregate error.
The final selected candidate's iteration and per-element errors are also logged.
Patience stops classify `stall_cause` from the last patience window, in order:
`moves_below_rounding` when mean actual per-atom motion is below 0.001 A,
`low_acceptance` below 30% mean acceptance, otherwise `converged`.

Static element/radius/residue metadata is cached per inventory. Each search
iteration computes one neighbor/overlap graph on PDB-rounded coordinates and
reuses it for checking and proposing repairs. Final export still independently
rereads the written PDB and rebuilds its geometry checks.

## Step-profile control targets

`--step-profile` adds a homogeneous control arm for geometric/FEM nanoindentation
comparisons. It fits **shared half-height bounds** to the selected reference
atoms' count-weighted mean radius and mean squared radius. At fitted size it
preserves every element's atom count, hence composition and mass. All elements
share one radial shape and have constant composition through the wall. This
removes the reference layering and outer-tail character; the skewness difference
quantifies part of that loss. It does not make a mechanically equivalent capsid.

The plateau has constant volumetric density. Each edge is softened by a **linear
ramp centred on its bound**, width `--step-edge-width` (default 4 Å; 0 is a hard
step). The fit includes the ramps when matching moments. At half-height bounds
`a,b` and width `w`, atom centers may occupy **[a − w/2, b + w/2]**. Ramps cannot
overlap: `a − w/2 >= 0` and `b − a >= w`. A fit with no valid interior solution
fails explicitly. Shell densities are integrated analytically, including partial
ramp bins, so changing bin alignment does not change the total material.

```bash
# Inspect the fit before generating anything.
mastercap analyze --input ./1cwp_full.vdb --bin-width 4 --step-profile \
  --output-prefix results/step_fit --csv --plot

# Step targets use counting (not orientation) as the suggestion basis.
mastercap calibrate --input ./1cwp_full.vdb --bin-width 4 --step-profile \
  --replicates 50 --seed 42 --floor-multiple 2 \
  --output-prefix results/calib_step --csv --plot

mastercap generate --input ./1cwp_full.vdb --bin-width 4 --step-profile \
  --mode independent --distribution regular \
  --tolerances-from results/calib_step_calibration.json --allow-approximate \
  --output-prefix results/step01 --csv --plot
```

**Resizing:** `--inner-radius` and `--outer-radius` override the step's half-height
bounds, with omitted bounds taken from the fit. The ramp width stays fixed.
Plateau densities stay at their fitted values; target element counts scale with
the analytic step volume. Integer residue inventory selection then approximates
those real-valued counts. This does not preserve the original N or moments in a
resized particle. A half-radius pilot keeps `w=4 Å`, rather than halving `w`.
Calibrate that new geometry separately; its support and counts differ.

The JSON step block reports fitted and generated bounds, support, exact moment
residuals, reference/target counts, plateau density versus reference mean and
maximum unsmoothed bin density, and sensitivity fits. Hard-edge, mass-weighted,
and depth/per-area fits are **diagnostics only**; they never choose the target.
Per-element fits likewise diagnose differences rather than setting element-specific
bounds. Unrepresentable diagnostic fits record an error. A plateau denser than
the densest reference bin produces a packing warning. At fitted size the reference
is re-binned on the step support and the report counts atoms outside it. Resized
steps have no reference curve or `original_profile_error`; CSV original columns
are empty. Profile PNGs always show a **Step target** line.

`--smoothing-sigma > 0` is incompatible with step mode; `--no-smoothing` is
accepted but unnecessary. `--step-edge-width` requires `--step-profile`.
`--sulfur-residue-placement profile` follows the step S target;
`reference` is rejected because reference shells contradict homogeneity.
Connectivity, minimum distances and both final audits remain mandatory.

Step calibration labels self-error **reference vs step**. It is the deliberate
model mismatch, not a floor. Orientation and placement levels are marked skipped;
the PNG/CSV omit them. The PNG adds a reference-vs-step point and a counting
range; generated triangles still require `--generated`. Counting-floor multiples
are reported, but orientation multiples and placement-gap closure are undefined.
`--compare` remains measurable but **does not enlarge step tolerances**. The
counting ×2 rule is stricter than the reference model's orientation rule, so an
`approximate` result is expected and can be saved with `--allow-approximate`.
This never permits failed geometry. Calibration checks include profile model and
edge width; old reports without these fields mean reference-shaped targets.
Standalone validation rebuilds the saved step and checks bounds, counts and edges.

Crystallographic missing/disordered protein (including 1CWP N-terminal arms) is
absent from both target models: the inner fit describes visible protein only.
The fitted step outer bound lies inside the reference's geometric outer radius.
Interpret indentation contact against **b**, rather than the reference R_out,
while retaining the distinction between half-height bound and atomic support.

### First 1CWP step pilots

With 4 Å bins and 4 Å ramps, the fitted half-height bounds are
**98.5680–135.4006 Å** (thickness 36.8326 Å; support 96.5680–137.4006 Å).
Hard-edge, mass-weighted and per-area sensitivity thicknesses are respectively
37.0537, 36.8938 and 36.1891 Å. Plateau density is 0.0335732 atoms/Å³:
1.532 times the geometric reference mean and 0.780 times its densest bin.
Reference skewness 0.10093 becomes −0.21264; 9,780 reference atoms lie outside
the fitted ramp support.

Seed 42, independent/regular initialization, sulfur profile placement and
S-S=3.0 Å produced these results (multiples use the **mean** counting floor):

| Geometry | Atoms | Pushes used / maximum | C error / floor | N error / floor | O error / floor | S error / floor |
|---|---:|---:|---:|---:|---:|---:|
| Fitted size | 214,440 | 10 / 50 | 2.39× | 1.91× | 2.85× | 0.064× |
| Half bounds, same 4 Å ramps | 26,814 | 0 / 50 | 2.07× | 2.03× | 2.51× | 0.157× |

Both finished **valid** under their own `2 × counting p95` tolerances and passed
standalone validation and both audits (0 isolated atoms, 1 component, 0 internally
disconnected residues). A mean-floor multiple above 2 can still pass the p95
rule. Normal acceptance stopped them before the 50-push ceiling; this is not an
equal-push comparison. Observed repair-search times were 152.5 s and 10.7 s,
respectively, with first feasibility at iterations 57 and 49. Timings include
concurrent development workload and are not isolated speed measurements.
Detailed fits, floors, errors, runtimes and audit evidence are recorded under
`step_profile` in [the benchmark report](benchmarks/1cwp_refinement.json).

## FEM-oriented packing diagnostics and fine-bin transport

The intended FEM mesh marks a cell solid when it intersects an atom's vdW sphere.
At 0.5 Å resolution, atom-scale artifacts affect the solid geometry. Earlier user
pilots reported wall populations of 267/532 atoms within 0.01 Å, an outer 0.5 Å
shell density of 0.0111 versus a ramp target of 0.0021, and about 0.59 nonbonded
inter-residue contacts per atom below 2 Å versus zero for 1CWP. Their 1 Å profile
error barely changed over 50 pushes (0.0302 to 0.0299). These motivated this
revision; they are prior pilot observations, not results of the new defaults.

Every `generate` and `validate` report now contains `diagnostics`, including a
reference summary cached once on its `Analysis`. Validation can measure old
particles with their existing matching reports, without regeneration. Diagnostics
**never set quality status or change acceptance thresholds**:

- `contacts`: `2 * pair_count / N` for positive overlaps and for positive gaps
  below `sqrt(3)*h`, separately for each `--mesh-element-sizes` value (default
  `0.5,1.0` Å). Thresholds are recorded: 0.866 and 1.732 Å. This estimates
  *potential* mesh bridging, not exact voxel occupancy for a particular grid.
- `inter_residue`: close-contact rates below 2 and 2.5 Å and per-element-pair
  nearest-neighbor p01/p50. Whole residue pairs joined by an observed backbone
  C--N distance below 1.6 Å are excluded. The peptide-pair count divided by the
  residue count is also reported; inference uses geometry, not chain numbering.
- `residue_drift`: median/p95 observed rigid-fit RMSD, reusing generated residue
  transforms. Old reports missing transforms give null values, not a fabricated
  zero. Reference drift is zero by definition.
- `walls`: atom counts within 0.01, 0.1 and 0.5 Å; eight inward 0.5 Å density
  shells per wall; expected target densities; first-shell excess; and expected
  counts within 0.1 Å (also 0.01 Å for the terminal summary). Step expectations
  use the analytic ramp; reference targets use the bin density at each midpoint.
  Reference-only wall expectations are null, since no target is imposed on it.
- `layering`: 1 Å shell density divided by its 9-shell moving average, retaining
  the valid interior, with relative SD, extrema and shell count. Step targets
  also report plateau density/target relative SD, at least 6 Å from support walls.
  Too few shells or no density support gives null, not a zero layering claim.

Pairs are queried once at `2*max(vdW)+sqrt(3)*max(h)` and processed in blocks of
2 million to limit temporary arrays. The pair list itself still scales with the
number of local contacts. Exact nonbonded nearest statistics use separate
per-element trees. These diagnostics add analysis/export time, not repair time.
Across an R × thickness grid, compare these same measurements with the reference
and with each other to quantify how consistently the approximation is held.
They do not reconstruct the user's exact voxel mesher. Matching a 1 Å bin
average does not guarantee the continuous ramp inside that bin; the 0.5 Å wall
measurements expose this remaining discretization effect.

### Seeded wall-band clamp

`--wall-clamp band` is the new default. Fixed inward depths are drawn once per
atom and wall. A step ramp of width w uses `w*sqrt(U)`; hard steps and reference
targets use a uniform depth from 0.002 Å to the smaller of 2 Å and the wall bin
width. Only out-of-support atoms receive the 0.8 radial correction. The band mask
uses PDB-rounded positions, so raw positions that round outside cannot remain
stuck indefinitely. During repair, anchored sulfur keeps its existing spring
instead of receiving a conflicting random band target. Refinement continues to
reject out-of-bounds moves. `search.wall_clamp` and top-level `wall_clamp` record
mode, band widths and atom-iteration clamp events.

`--wall-clamp legacy` retains the original clipped targets at wall ±0.002 Å and
consumes no extra random draws. It is useful for a controlled comparison.

### Edge-directed transport at 1 Å bins

The previous drift was `strength * distance`, typically 0.0025 Å at 1 Å bins.
An atom 0.5 Å from an edge needed about 200 pushes, exhausting patience before
its bin count could change. `--refine-drift-step auto` now uses
`max(min_step, min(max_step, 0.02*bin_width))`: 0.02 Å at 1 Å bins, capped at
0.05 Å with the default maximum. A positive numeric value selects a fixed speed
no larger than `--refine-max-step`. The actual request is limited by the distance
to the edge plus the 0.002 Å crossing margin.

Per edge and direction, at most `ceil(--refine-transport-factor * remaining_quota)`
atoms drift (factor default 2). The per-element fraction cap still applies;
nearest-to-edge ordering keeps transit deterministic. The new transport uses the
**cumulative count imbalance** on each side of an edge, allowing material to
travel through balanced bins toward a distant deficit. Restricting transport to
immediately adjacent surplus/deficit pairs left most remaining error unreachable:
a direct implementation trial retained 1,835 surplus C atoms but only 4.5 atoms
of adjacent transfer quota. This routing addition was needed to pursue the
specified fine-bin acceptance targets. `--refine-transport-routing local`
selects adjacent-only routing with the new speed and guarded trials, making the
strategy explicit for comparisons; the default is `cumulative`. Legacy drift
always restores local routing. A routing choice can change finite-budget L1
results for sharply layered profiles, so compare the reported element errors. Accepted useful crossings
reset patience even when their improvement is below the usual 0.1% threshold.
For crossing selection, effective patience is at least twice the estimated
half-bin transit time: 50 pushes at 1 Å with auto drift. Explicit patience 0
still disables that stop. `refinement_settings.effective_patience` records the
resolved value. Mismatch selection retains its earlier patience behavior.

Push records include drift/crossing proposals, actual crossings, useful
crossings, requested/actual motion and mean initial distance to the chosen edge.
`transport_starved` distinguishes a no-crossing window with drift proposals from
rounding stalls. `--refine-drift-step legacy` restores the old speed, quotas and
patience, including when combined with `--wall-clamp legacy` for seed regression.
For independent crossing refinement, a conflicting trial can be halved toward
its snapshot in up to four checked rounds before rejection. Lower-distance,
contact, wall and sulfur-bin guards are unchanged; failed residual trials still
revert. Rigid mode does not interpolate coordinates. Push records distinguish
backtracked/accepted atoms, attempted guard reasons, final undone reasons and
actual motion. This bounded backtracking addition addresses the rejection rate
seen with faster drift; it is disabled by legacy drift. `actual_move` retains
its all-atom mean; the extra per-proposal and moving-atom means make denominators
explicit. Mandatory geometry and final audits remain active.

### Optional nonbonded inter-residue floors

`--inter-residue-limits off` is the default. To scan reference-derived floors:

```bash
mastercap generate --input ./1cwp_full.vdb --output-prefix results/packing_scan \
  --step-profile --bin-width 1 --mode independent --distribution regular \
  --sulfur-residue-placement profile --pair-limit-override S-S=3.0 \
  --inter-residue-limits reference --inter-residue-scale 0.6 \
  --wall-clamp band --refine-drift-step auto --refine-updates 50 \
  --seed 42 --time-limit 3600 --max-iterations 3000 --allow-approximate
```

The statistic is each atom's nearest partner of the relevant element in a
*different, nonbonded residue*, symmetrized over element pairs. Reference residue
pairs joined by backbone C--N <1.6 Å or S--S <2.3 Å are excluded. The requested
nearest-neighbor percentile and grouped fallback settings are reused; this is
not the percentile of all pairs within 4 Å. Counts, p1/p50/p99 and fallbacks are
recorded under `analysis.inter_residue_limits`.

For different residues, the lower bound is the maximum of the usual element
limit, the covalent floor, and `scale * reference_nonbonded_limit`. Within a
residue the earlier element limit remains. Pairs in `--pair-limit-override`
retain their selected limit and covalent floor, without the extra nonbonded
term. `--inter-residue-scale` defaults to 1 when enabled and is rejected without
`--inter-residue-limits reference`. Repair, component-contact selection, trial
acceptance, consolidation and final audits use the same effective bounds.
`inter_residue_limits` records mode, scale, effective matrix, contact windows and
reference statistics. Standalone validation rebuilds saved settings; missing old
settings mean off.

A window is `vdW_sum - overlap_margin - 0.003 - effective_lower`. Windows below
0.05 Å warn; if no C/N/O pair has a positive window, search fails immediately.
Reference-derived floors can make dense packing unreachable, so do not assume
that a larger scale is better or feasible. Scan off, 0.6, 0.7, 0.8, 0.9 and 1.0
in order and stop at the first failed geometry. The production suggestion is
the largest feasible scale whose observed repair-search time is within 3× off.
The controls do not change target profiles, so calibration compatibility keys
remain unchanged. This version does not improve rigid residue-mode packing.

### Revision 07 measured results (1CWP, seed 42)

The 1 Å step benchmark used independent atoms, regular distribution, sulfur
profile placement, S--S=3 Å, 50 pushes and no acceptance tolerances. Its total
error fell from 0.035806 at first feasible geometry to 0.002683 at push 50
(**92.5%**), with **85.1%** mean move acceptance. Layering relative SD fell from
0.02371 to 0.02150. The 4 Å step benchmark reached about 0.00019 total error,
below the supplied 0.0174 regression bound. Exported structures passed fresh
connectivity/pair-distance audits and standalone validation; without tolerances
their status remains `uncalibrated`.

The adjacent-only implementation did not meet the specified 1 Å targets:
16.5% error reduction and 45.8% mean acceptance. This prompted cumulative
routing and checked partial moves. These changes have a finite-budget tradeoff
for the unsmoothed layered target: total error was 0.10187 versus 0.08707 for
the direct local control. Carbon improved, while nitrogen and oxygen worsened.
Use the element errors and the explicit routing option when comparing methods;
these benchmarks do not establish a universal improvement for every target.

The step particle has 7/18 atoms within 0.01 Å of the inner/outer wall, but its
outermost 0.5 Å shell remains **2.41×** the continuous ramp target. Matching
1 Å bin averages does not control every sub-bin density or exact voxel geometry.
Packing diagnostics therefore remain necessary across the full radius/thickness
grid. The separately measured available v0.1.11 baseline has different
generation bins and stopping conditions and is labelled accordingly in the
evidence; it is not an equal-work comparison. The separately rebinned 1 Å and
4 Å partitions can have nonmatching edges, so a coarser-bin error need not be
smaller. Their disagreement can expose variation within the generation bins.

The ordered inter-residue scan reached feasible geometry and completed 50 pushes
for every scale. Search times exclude preparation and diagnostics:

| Scale | First feasible iteration | Search seconds | Total error, 1 Å | Nonbonded <2 Å / atom |
|---|---:|---:|---:|---:|
| off | 220 | 470.2 | 0.00268 | 0.5571 |
| 0.6 | 223 | 534.9 | 0.00282 | 0.4307 |
| 0.7 | 297 | 682.6 | 0.01297 | 0.0000 |
| 0.8 | 392 | 721.6 | 0.01501 | 0.0000 |
| 0.9 | 455 | 851.6 | 0.01822 | 0.0000 |
| 1.0 | 630 | 1533.7 | 0.02482 | 0.0000 |

**Suggested scale for this benchmark: `--inter-residue-limits reference
--inter-residue-scale 0.9`.** F=1.0 was feasible but exceeded the measured
3× off cutoff (1,410.5 s). This recommendation uses geometry and runtime only;
F=0.9 mean move acceptance was 27.9%, versus 85.1% with limits off, and its
profile error was higher. The default remains `off`. More capsids, seeds and
grid points are needed before treating this scale as generally applicable.

The half-bound step pilot completed 50 pushes with 26,814 atoms,
total 1 Å error 0.00835, and 2/2 atoms within 0.01 Å
of the inner/outer wall. It also passed standalone validation. Full diagnostics,
rebinned errors, trajectories and replay settings are recorded under `revision07`
in [benchmarks/1cwp_refinement.json](benchmarks/1cwp_refinement.json).

## Calibration

Calibrate one complete reference at the **same bin width, smoothing, model and
radii** that generation will use:

```bash
mastercap calibrate --input ./1cwp_full.vdb.gz --bin-width 4 --no-smoothing \
  --replicates 50 --seed 42 --output-prefix results/calib_1cwp --csv --plot

mastercap generate --input ./1cwp_full.vdb.gz --bin-width 4 --no-smoothing \
  --tolerances-from results/calib_1cwp_calibration.json \
  --output-prefix results/particle --seed 42

# Reassess an existing particle using the calibration, preserving its original report.
mastercap validate --input ./1cwp_full.vdb.gz --pdb results/particle.pdb \
  --report results/particle.json --tolerances-from results/calib_1cwp_calibration.json \
  --output-prefix results/validation
```

`calibrate` writes `PREFIX_calibration.json`, optionally `.csv` and `.png`.
It never writes a PDB. It refuses a prefix already belonging to a generation
report even with `--overwrite`; use a distinct prefix. Its profile analysis
skips distance limits entirely, so it has no `--pair-limit-override` option.

For the default reference-shaped model, the reported levels use the same volume-weighted integrated absolute error as
generation, divided by each element's target count. Total is atom-weighted:

- **Self:** the reference against its target; zero without smoothing, otherwise
  it shows smoothing bias.
- **Counting:** independent multinomial bin counts at each reference element's
  atom count. This measures typical finite-count fluctuations.
- **Orientation:** rotate every reference residue about its **own centroid**,
  leaving that centroid and its radius fixed. This measures sensitivity to free
  residue orientation, and is the default suggestion basis.
- **Placement:** additionally permute centroid radii among residues, retaining
  their original directions. This removes the element/radius correlation. It is
  a model baseline, not the exact generator initialization algorithm.

Each stochastic level has mean, population SD, p05/p50/p95 and extrema over
`--replicates` (default 50, seed 42). These are reference-resampling scales, not
proven lower bounds: fitting profiles deliberately can beat the orientation
mean. Levels retain reference atom counts and centroids even when target radii
are overridden; resized targets can therefore include an inventory/geometry
mismatch in these error scales.

Repeat `--compare other.pdb` for experimental whole-particle comparisons. Each
uses its own geometric center and the reference edges. JSON retains raw errors,
per-element count ratios, and shape errors normalized to the target's in-range
count; actual and in-range counts are reported. Missing target elements
or no in-range support cause a clear error. CSV comparison rows show **shape**
error; JSON also retains raw error. Chain labels with at least 1000 atoms each
provide a separate scaled profile-spread diagnostic, not an automatic tolerance
increase or an independent whole-particle experiment.

Repeat `--generated particle.pdb` to assess generated structures. A sibling
`.json`, when present, must match the reference hash, model, bins, smoothing and
radii, and binds the PDB hash; its saved center is used. Without a sibling, use
the analysis center. Reports and CSV express errors as multiples of counting and
orientation means and give `(placement - error)/(placement - orientation)`.
`gap_closed` retains raw values, including above 1 or negative; JSON additionally
provides `gap_closed_clipped` capped at 1. Zero denominators/nonpositive gaps are
null with a reason, never invented finite ratios. Log plots mark zero errors at
an explicitly labelled display position. No Total tolerance is applied.

For reference-shaped targets, suggestions default to **2 times the orientation p95**, rounded upward to three
significant figures. Choose `--floor-multiple`, `--floor-basis counting` or
`--floor-statistic mean` explicitly to change that rule. An explicit comparison's
largest shape error raises the suggestion when larger. Composition is suggested
as 0, retaining the existing one-atom allowance. Suggestions are logged as a
ready-to-paste command and stored with the rule used for each element.

Tolerances are **never applied automatically**. Generate and validate accept:

```bash
--profile-tolerance 0.03
--profile-tolerance C=0.02,N=0.04,O=0.04,S=0.15
--profile-tolerance 'C=0.02,N=0.04,*=0.05'
```

Every target element needs an explicit value or `*`; duplicate/unknown elements,
negative/nonfinite values and mixed scalar/map forms are errors. Absent reference
elements are warned about and ignored. The config records the normalized map and
original string; `metrics.elements` records each applied `tolerance` and
`within_tolerance`. Old saved float tolerances still work. Validation uses saved
tolerances unless explicitly overridden, and does not need the original
calibration file after generation has saved its values and SHA-256.

`--tolerances-from` cannot be combined with either explicit tolerance option.
A reference, model, bin, smoothing or radius mismatch is an error, not a bypass;
recalibrate. Geometry, connectivity and pair-distance audits remain mandatory
regardless of the tolerance values.

### Reading the calibration PNG

Each panel shows one element, or the atom-weighted Total, on its **own logarithmic
error axis**. Read left as smaller error; compare ratios within a panel rather
than horizontal positions across panels.

- Counting, Orientation and Placement bars span the replicate p05–p95 range;
  the dot is the mean. These ranges describe replicate variation, not confidence
  intervals for the mean or proven lower bounds on achievable error.
- Hollow diamonds are experimental `--compare` **shape** errors. Filled triangles
  are `--generated` particle errors; up to five receive filename labels.
- The dashed vertical line is the suggested per-element tolerance, normally
  `2 × orientation p95`. A triangle at or left of it passes that element's
  profile threshold. Geometry and composition must still pass separately.
- Total has no tolerance line or acceptance threshold. A zero error is displayed
  at a labelled positive position because a logarithmic axis cannot show zero.

A calibration run without `--generated` does **not** assess a subsequently
generated particle. Its Generated row has no markers. To see particle10 against
the same baselines, use a distinct prefix and matching profile settings:

```bash
mastercap calibrate --input ./1cwp_full.vdb --bin-width 4 --no-smoothing \
  --replicates 50 --seed 42 --generated results/particle10.pdb \
  --output-prefix results/calib_particle10 --csv --plot
```

For example, N error 0.1267 is about 1.93 times the reported orientation mean
0.06583, yet passes the suggested tolerance 0.141. Passing the suggestion means
meeting the selected rule, not matching the orientation mean. A loose S threshold
also does not imply that a generated S error must be large: an error of zero passes.

### First 1CWP calibration

Fifty replicates, seed 42, 4 A bins, no smoothing; means and suggested per-element
limits (`2 * orientation p95`):

| Element | Counting mean | Orientation mean | Placement mean | Suggested tolerance |
|---|---:|---:|---:|---:|
| C | 0.006668 | 0.035818 | 0.096589 | 0.0753 |
| N | 0.013554 | 0.065827 | 0.126262 | 0.141 |
| O | 0.013103 | 0.051661 | 0.078231 | 0.113 |
| S | 0.072148 | 0.369778 | 0.779037 | 0.864 |
| Total | 0.009210 | 0.044620 | 0.099487 | — |

Across the nine v0.1.8 particles at 50 pushes:

| Element | Error / orientation mean | Raw placement-to-orientation gap closed |
|---|---:|---:|
| C | 0.73-1.02 | 0.99-1.16 |
| N | 1.46-1.71 | 0.23-0.50 |
| O | 1.15-1.46 | 0.10-0.71 |
| S | 0-2.30 | -0.18-1.90 |

C already matches or beats that typical baseline; N has the largest remaining
gap (about 0.030-0.047 normalized error),
followed by O (0.008-0.024). Independent-atom fitting can still improve beyond a
random-residue baseline. Mean mode effects for C and O are below pooled within-mode
seed SD. Profile placement's mean N improvement is about **1.4 pooled seed SD**,
or 0.075 orientation-mean units, but it reverses for one seed. Three seeds do not
establish statistical significance or a universally superior placement mode.

With sigma 1 A, self-error is 0.000156 overall and 0.001207 for S. Suggested
limits are C=0.0767,N=0.140,O=0.113,S=0.843. All ten supplied benchmark particles
pass standalone validation when their matching suggestions are explicitly loaded.
That is validity under the selected `K=2, p95` rule, not independent experimental
confirmation. No second complete experimental CCMV structure was supplied;
A/B/C chain-label spread is reported separately. Full levels, generated ratios,
gap closure, per-seed effects and validation results are retained under
`calibration` in `benchmarks/1cwp_refinement.json`.

## Acceptance and reports

JSON is mandatory. CSV profiles (`--csv`) and PNG plots (`--plot`) are optional.
The independent `pair_distance_audit` also uses fresh element trees on the
reread PDB. For each pair it records the closest atoms in different residues,
minimum distance, counts below the applied limit and below the covalent floor,
and bounded examples with atom serial/name and residue identity. The floor
excludes expected peptide C(i)-N(i+1) links in the same chain and TER segment.
An actual floor violation is a hard, atom-specific failure before publication;
`--allow-approximate` cannot bypass it. Generation repair and checked moves
also enforce inter-residue covalent floors, without altering the measured
analysis limits. This matters for 1CWP C-O: its measured 1.2196 A bond-scale
limit is below the 1.242 A floor. An INFO line reports the final inter-residue
S-S minimum. Standalone validation repeats both audits.

After PDB writing and rereading, `connectivity_audit` builds a fresh contact
graph independently of the search's local graph/helper. It checks strict positive
vdW overlap, atom degree, global components and each residue's internal subgraph.
Single-atom residues are internally connected but still require a global contact.
Any isolated atom, multiple components, internally disconnected residue or
disagreement with `validate_geometry` is a hard failure before publication.
Errors identify file-order indices, PDB serials, atom/residue names, chains and
residue numbers. The audit records the largest component sizes and bounded atom/
residue examples. Generate and standalone validate both run it; approximation
cannot bypass it. A success INFO line reports N atoms and M edges.
Reports include original, smoothed and generated profiles; element counts;
integrated profile errors; geometry and graph checks; nearest-distance percentile
comparisons; provenance; residue transformations/deformation RMSD; seed; software
versions; elapsed time; and Linux process peak RSS.
Search metadata also records the first feasible iteration, refinement steps,
actual profile updates and the selected candidate's iteration. Candidate geometry
and metrics describe the selected feasible output, which may precede the final
search iteration.

`feasible_iterations` lists feasible checks excluding restored snapshot repeats;
`restored_snapshot_iterations` lists those repeats separately. `best_history` records each
replacement, the four ranking components and every element error. New-best INFO
messages are independent of `--progress-interval`. `initial_metrics` stores the
iteration-0 total and per-element metrics. `push_history` records push iteration,
strength, selected/nonzero-pushed counts, mean/max absolute requested radial move,
the next evaluated state's damage, recovery, strength changes and rollbacks.
`recovery_iterations` excludes the push itself: 0 means the next state was already
feasible. Unrecovered pushes have no completed recovery value. Recovery followed
by a worse-profile rollback still records its completed recovery length.
Each push contains `profile_before`/`profile_after`, `force_components` (mean/max
repair, bridges, radial clip, centering, radial push and neighbor contributions
before caps/projection), `applied_force_components`, and `actual_move` after
projection/checking. Per-atom records include proposed/accepted/undone counts,
reasons, passes and acceptance fraction. Reason counts can overlap, and undoing
a whole residue or neighbors can undo additional atoms. Rollback records keep
the first post-push `profile_after` and, when recovered, `profile_recovered`.
Histories contain scalar summaries,
never coordinate snapshots or atom-index arrays.

`slack_at_first_feasible` and `slack_after_consolidation` report minimum and 5th
percentile gaps to distance/contact limits and counts within 0.005 A of them.
They also count pairs/edges within half the requested slack. Initial
`slack_after_consolidation` is null unless the initial episode completed;
`slack_at_refinement_start` always describes the actual start of refinement,
including when consolidation was abandoned. `consolidation_completed` refers to
the initial episode. `consolidation_iterations_total` includes later restorations;
`consolidation_history` records each episode's trigger, start/end, steps, budget,
before/after gaps/counts and reason (`target`, `no_progress`, `iteration_limit`,
`interrupted`, `abandoned` or `search_stopped`).

Each push's `per_element` records selected/accepted/undone atoms, too-close and
broken-contact rejections, actual bin crossings (improved/worsened/neutral L1
count error) and `profile_error_change` (after minus before). Crossing attribution
updates bin counts in atom-index order so multiple moves cannot all claim the
same deficit. Masks and coordinate arrays remain private.
`undone_too_close_by_pair` attributes rejections to sorted element pairs, both
globally and per element. Counts deduplicate an atom/pair type across trial
passes; one atom can contribute to several pair types. They identify direct
offenders before whole-residue rejection expands the undo. The search report
sums these counts over pushes and lists the top three sulfur pairs, also logged
at the end. These are rejection occurrences across pushes, not unique run atoms.
`acceptance_mode`, `committed_pushes`, `mean_acceptance_fraction`,
`recoveries_improved`, `recoveries_worse`, `repair_attempts` and `blocked_evidence`
support the final console summary. A committed push has at least one retained
atom move, or an improved recovery in rollback mode; zero-move attempts still
consume the push budget. In push records, `atoms_proposed = atoms_accepted + atoms_undone`
counts proposals; `proposed_moving_atoms`, `atoms_moved` and `stationary_proposals`
distinguish physical motion from proposals below PDB rounding resolution.
The console says `0 attempted` when no post-push repair was attempted.

`refinement_updates` equals `profile_updates` and `len(push_history)`;
`refinement_steps` still counts iterations since first feasible geometry. Final
logs summarize feasible checks, pushes, rollbacks and median recovery. A broken
final progress sample is explicitly marked as not the exported candidate;
`last_search_geometry` describes that last evaluated state, while `geometry`
describes the selected candidate. A rollback may restore coordinates after that
evaluation without reusing its stale geometry graph.

The primary per-element profile error is:
`sum(abs(generated_density - target_density) * shell_volume) / target_count`.
Comparisons against the mapped original unsmoothed profile are also included.

No universal tolerances are applied by default. Set `--profile-tolerance` and
`--composition-tolerance`, or explicitly load a matching calibration with
`--tolerances-from`. Profile tolerance may be a scalar, per-element map or map
with `*` fallback. Composition allows relative count error with a one-atom
rounding allowance. Missing tolerances produce `uncalibrated`; exceeding supplied soft
tolerances produces `approximate`. Both require `--allow-approximate` to save.
**That flag never relaxes mandatory constraints.**

Terminal output identifies whether tolerances came from a checked calibration
file, explicit CLI values, or the saved generation report. It prints effective
per-element values and the composition allowance, then compares each final
element's error and absolute count difference to its limits. Standalone validation
can reuse saved values without rereading the original calibration file.

`valid` means mandatory geometry/audits and **every element's** profile and
composition checks passed. It does not certify chemical validity. Total profile
error is diagnostic, with no separate acceptance threshold. `--allow-approximate`
permits saving a soft failure; it does not force an approximate status.

Search stops immediately when all those checks pass, even at the first feasible
geometry. `--refine-updates` and `--time-limit` are upper limits, not work that
must be completed. For example, errors C=0.0327, N=0.1267, O=0.0922 and S=0
all pass tolerances C=0.0753, N=0.141, O=0.113 and S=0.864. If composition also
passes, that run can finish with **zero profile pushes**. A zero refinement budget
also stops at first feasible geometry, but does not itself confer valid status.

| Exit code | Meaning |
|---|---|
| 0 | Analysis/calibration completed, or generation/validation valid within supplied tolerances. |
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

Complete 1CWP benchmarks and reference-resampling calibrations are recorded
above. Broader validation still needs independent experimental full-particle
comparisons, more seeds and capsids/settings, and automated VMD compatibility
checks. Reference-derived thresholds do not establish universal physical validity.

A repeatable synthetic timing probe is available:

```bash
python benchmarks/search_shell.py --residues 600 --iterations 30 --repeats 3
```

It reports timings and remaining violations on a carbon-only shell. This measures
iteration cost and is not a demonstration of full-capsid convergence.

To exercise the adaptive controller and collect push/damage/recovery histories:

```bash
python benchmarks/search_shell.py --residues 60 --iterations 400 --repeats 3 \
  --refine-updates 50 --selection-objective total
```

Compare push counts, completed recovery lengths, rollbacks, initial/final errors
and termination reasons. Different stopping budgets imply different amounts of
work; timings alone are not an equal-work speed comparison or 1CWP validation.
