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

## Command-line help

`mastercap -h` (or `mastercap --help`) prints the complete reference for all three
commands: every option, its meaning, concrete or inferred/unset defaults, accepted
choices, units, examples, output files, acceptance rules and exit codes.

For a shorter reference limited to one command, use:

```bash
mastercap analyze -h
mastercap generate -h
mastercap validate -h
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
Radii bound **atomic centers**, not vdW surfaces. Defaults use radial extrema;
`--radius-method percentile --radius-percentiles LOW HIGH` selects an interval
and reports excluded atoms. Excluded atoms no longer contribute to profiles or
the selected inventory; trimming may leave partial residues.

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

## Acceptance and reports

JSON is mandatory. CSV profiles (`--csv`) and PNG plots (`--plot`) are optional.
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

To exercise the adaptive controller and collect push/damage/recovery histories:

```bash
python benchmarks/search_shell.py --residues 60 --iterations 400 --repeats 3 \
  --refine-updates 50 --selection-objective total
```

Compare push counts, completed recovery lengths, rollbacks, initial/final errors
and termination reasons. Different stopping budgets imply different amounts of
work; timings alone are not an equal-work speed comparison or 1CWP validation.
