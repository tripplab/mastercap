# Technical design and implementation map

This implementation follows the approved Q&A specification. Scientific quality
thresholds remain provisional until calibrated on expanded CCMV 1CWP.

| Module | Responsibility |
|---|---|
| `model.py` | Records, internal identities, protein recognition, fixed vdW/covalent/mass tables. |
| `pdbio.py` | PDB/gzip/models, filtering, VIPERdb cache, TER writer. |
| `analysis.py` | Center, bounds, profiles, smoothing, pair statistics and shared target dispatcher. |
| `radial.py` | Array-only shell volumes and radial bin convention. |
| `step.py` | Analytic ramp integrals and moment-matched homogeneous targets. |
| `calibration.py` | Profile baselines, settings compatibility and tolerance suggestions. |
| `constraints.py` | Spatial checks, internal/global graphs, acceptance metrics. |
| `generate.py` | Integer inventory, initialization, bounded repair, rigid projection. |
| `refinement.py` | Candidate ranking, slack targets, checked partial pushes, acceptance and rollback controllers. |
| `reports.py` | JSON/CSV/PNG, staged output and rollback. |
| `cli.py` | Combined workflows, configuration, provenance and reread validation. |

Coordinates are float64 arrays. Atom identity is independent of serial numbers.
Templates contain observed atoms, including partial residues. MILP selects integer
template-variant counts by normalized element errors plus lower-weight frequency
errors. No missing chemistry is reconstructed.

The initial search samples residue centers, rotates templates, and repairs
geometry with bounded moves. Independent updates deform residues; rigid updates
project onto rotations and translations. cKDTree and sparse graphs avoid an
unconditional N×N distance matrix. Per-inventory metadata caches element codes,
vdW radii, residue groups, element indices and pair limits. Each repair iteration
builds local geometry once on rounded coordinates and reuses it for both checks
and repair proposals. Final PDB reread validation independently rebuilds checks.

Contacts use atom boundaries rather than arbitrary component representatives.
Batched local queries retain the nearest compatible contact per component pair;
components without outgoing proposals use external queries partitioned by element
compatibility. Kruskal selection reduces proposals to a sparse spanning forest.
This forest can remain disconnected; it is a repair proposal, never evidence of
actual connectivity. No component-ID chain is inserted. Internal residue repair
uses the same compatibility rules. Rounded coordinates determine acceptance.

Preparation and repair clocks are separate. Initialization and inventory do not
consume the cooperative repair budget; explicit fallback shares only remaining
repair time. Reports distinguish time and iteration exhaustion and record phase
timings. See README for report field semantics.

Once geometry is feasible, refinement is budgeted in pushes (`--refine-updates`,
default 50), including rollbacks; repairs do not increment that counter.
`--refine-iterations` is an optional additional hard iteration cap, off by default.
Either budget set to zero selects first-feasible stopping. The default ranking is
atom-weighted total error, then mean element error, then maximum, then count
difference; `worst-element` restores maximum/mean/total/count ranking. The helper
and controller live in `refinement.py`; noise-aware selection is not implemented.

Reference statistics retain their measured values and pooled normalized factor.
Canonical override records are separate. `steric` derives a ratio for each element
from non-overridden, measured (non-fallback) pairs containing that element; use
the smaller element ratio times the overridden pair's vdW sum. Missing evidence
requires a numeric override. S-S defaults to 2.786177 A on 1CWP, sourced from C-S.
`pooled` retains the old formula, but every new override must pass the Pyykko
single-bond floor, 0.9 times the covalent-radius sum. Numeric values are checked
before input; derived values after measurement. There is no floor bypass.
Validation directly applies saved numeric `new_value` records; v0.1.7 steric
records without ratio emit a warning. Actual sub-floor coordinates still fail
the final audit. Older reports without override records use reference limits.
Explicit non-reference override pairs are excluded from statistical neighbor forces.

Optional sulfur placement anchors the first S by element, retaining seeded
rotations. `profile` shuffles quantile radii (regular uses bin midpoints), while
`reference` uses each atom's own coordinate-derived reference radius through
inventory provenance, without quantization. Incompatible resized bounds fail
clearly. A bounded greedy pass exchanges crowded S/non-S directions against
S-S limit + 2 A, preserving bin counts. Solve the positive-root offset equation
for the translated whole residue. Remaining residues sample the aggregate target
minus placed contributions. Off and S-free paths preserve old RNG draws.
Report distinct radii at 1e-6 A and shells separated by > 0.01 A separately.

The repair-only 0.05 S spring has a configurable 1.5 A dead zone, clipped inside
the target bin by 0.25 A per edge (quarter-bin padding for narrower bins).
Collapsed intervals use the nearest safe point. Tolerance 0 is an exact-radius
comparison; a hidden test-only `--no-sulfur-anchor-spring` switch supports the
predeclared diagnostic control. No spring acts after first feasibility. Trial
acceptance and consolidation preserve each anchor's target bin; rigid rejection
restores the whole residue. Crossing selection also filters out-of-bin anchor
moves. Private anchor arrays are not JSON. Per-progress repair_history records
disconnected residue names, violation pair types and anchored participation.

Search metadata separately enforces inter-residue covalent floors; measured
analysis limits are unchanged. A fresh final pair_distance_audit computes exact
inter-residue minima with element trees, below-limit/floor counts and atom
examples, excluding expected same-chain/segment sequential C-N peptide links
from the floor check. Any actual floor violation is a hard error, including for
legacy reports. Connectivity audit success alone is not distance validation.

Default crossing selection computes integer bin counts and target counts scaled
to each element's inventory. Reserve adjacent surplus/deficit crossing quotas,
rounded down, preferring the nearest atoms. Request the edge distance plus
0.002 A, capped at 0.05 A. Drift remaining surplus atoms toward useful edges.
Crossing plus drift uses at most floor(0.2*N_element) atoms and skips the previous
push's rejected mask. The mask remains private. Mismatch mode retains the v0.1.5
volume-quantile ranking and rounds its selection count up.

Radial strength is fixed at 0.005 by default. Gain control adjusts only after
next-iteration primary profile errors are known: two non-improving pushes reduce
strength by 0.7; a gain with >85% acceptance grows it by 1.25. Acceptance control
keeps v0.1.5 feedback. Both adaptive modes cap at 4x and stall below 1/16 of
starting strength; fixed mode does not shrink. Eligible nonzero drift/mismatch
requests have a 0.002 A minimum before the maximum-step clip. The explicit
rollback path keeps its independent damage controller.

First feasibility starts checked consolidation: up to 10 updates build 0.02 A distance/contact slack,
using strengths 0.3/0.15 and one-third-window guards when the window is narrow.
Use a separate accumulated force capped at slack/2 per atom, rigid projection,
PDB rounding and accept_trial. It applies no repair/profile/neighbor/centering
contributions and consumes no profile pushes. After at least one update, end
when each near-limit count is <=0.001*N, combined count improves <5%, or the
episode budget ends. Unexpected broken geometry logs a warning; three restarts
are a safety net. Restore slack below 70% acceptance or every 10 pushes, with
at least two pushes between episodes and up to three checked updates each.
Both triggers can be disabled. Final validation rules do not use slack.

Non-contact windows (contact threshold <= minimum separation) use full distance
slack and no contact pull. Positive narrow windows still use window/3. The
on/off switch permits a strict v0.1.6 control when combined with S-S=reference;
the default is on. This affects consolidation proposals only, never limits.

Nearest-neighbor contributions are off by default. The required reporting-only
1CWP A/B comparison reached total error 0.058776 with off versus 0.060828 with
scaled, and mean acceptance 62.3% versus 57.6%. Both stalled below rounding and
left sulfur unchanged. This supports disabling a force that spends slack without
directly targeting radial count error. The optional `scaled` mode uses coefficient
`0.002 * current_strength / starting_strength`, affects only radial-selected
atoms, and caps each atom's accumulated neighbor vector at half the radial
step cap. `off` disables this contribution; `fixed` uses coefficient 0.002,
all atoms and no additional cap. Both active modes guard pair targets by
the requested slack inside the lower-distance/contact window, skipping empty windows.
The overall 0.2 A update cap and rigid projection still apply, so guarded
targets alone do not guarantee feasibility of the combined update. Scalar
neighbor move statistics are included with each push. The historical exact 1CWP
off/fixed comparison at df200e5 confirmed strength-independent neighbor damage;
see the checked-in benchmark measurements.

Default per-atom acceptance isolates radial and neighbor contributions from
repair, radial clipping and centering. Apply the 0.2 A cap and rigid Kabsch
projection to a copy, round to 0.001 A, and check moved atoms against snapshot
pairs, existing edges, and new spatial neighbors (including rare-element large
bounds). Candidate pairs cover both trial/trial and trial/snapshot positions,
so reverting one atom cannot hide a mixed-state collision. Reject guarded
distance violations, broken old contacts and out-of-bounds moves. Broken edges
reject the farther-moving endpoint, or both for comparable moves. Recheck up
to three passes, then revert offenders with their direct neighbors. Any remaining
conflict or center error above 1 A restores the trial. Rigid mode rejects complete
residues. Existing edges are retained, so no trial-time component calculation is
needed. Next-iteration and independent export validation still rebuild geometry.
In acceptance strength-control mode, >85% grows strength by 1.25 and <50% shrinks
it by 0.7. The acceptance denominator
counts selected proposals, including feasible stationary rounded proposals,
expanded to complete residues in rigid mode. Actual moving/retained atoms are
reported separately. Rounded zero-move trials count toward budget/patience; rejected proposals
also consume a push. A private snapshot provides a validation safety net.

Too-close attribution deduplicates rejected atom/pair-type occurrences across
trial passes. Global and per-element counters may overlap across pair types;
rigid undo expansion is separate from direct offenders. Sum counters across
pushes and log the top three sulfur pair types.

After writing/rereading, connectivity_audit builds a fresh spatial tree, contact
edges, degree and global graph without calling local_geometry or components.
Per-residue sparse subgraphs are independently checked, including singleton
residues' global degree. Bounded examples retain file order, PDB serial and
residue identities. Any isolation, extra component, disconnected residue or
disagreement with validate_geometry raises before publication. Standalone validate
runs the same independent audit; neither status nor approximation bypasses it.

The revision-02 three-seed matrix improves total error beyond the seed spread
and completes all 50 default pushes with 83-87% global acceptance, but its full
scientific gate remains false. Seed 42 S has 52 crossing proposals and zero
actual crossings, with only 33.4% sulfur acceptance and predominantly lower-bound
rejections. A separate feasible-checkpoint probe at 0.05 A slack also leaves S
unchanged. These traces identify locally blocked rare-element motion; they do
not justify declaring sulfur convergence from aggregate acceptance or lowering
the prescribed validation constraints. See `revision02` in the benchmark JSON.

Revision 03 separates the S-S spacing override from opt-in S-shell placement.
The full CLI matrix uses three seeds for reference spacing, steric spacing and
profile placement, plus one smoothed profile-placement run. A separate seed-42
compatibility run disables noncontact slack as well as the S-S override; compare
the resulting PDB hash with v0.1.6 rather than assuming that changing only one
default restores all old behavior. Keep paired C/N/O deltas and completed-push
counts: reference-shell retention can improve S while increasing repair time or
slightly worsening another element. This is a reported scientific trade-off,
not a reason to weaken connectivity checks or silently change the default
placement mode. See `revision03` in the benchmark JSON.

The optional rollback path retains a snapshot and its ranking key. More than
0.05*N distance violations, at least 5000 components, or failure to recover
within 10 repair updates restores it; thresholds are configurable. Smaller
damage gets a repair attempt. Retain recovered geometry only if its complete
ranking key improves; otherwise restore and halve strength. Immediate improved
recovery grows strength by 1.25 up to 4x starting strength. Re-evaluations of
restored snapshots go into a separate list, excluding feasible/best histories.
Across recent damaging rollbacks, a fourfold strength reduction without a 25%
damage reduction stops further halving with `refinement_blocked` and evidence.
Below starting strength/16,
stop with `refinement_stalled`. Patience also stops after 10 pushes without a
cumulative >0.1% relative primary-error improvement (floor 1e-12); 0 disables it.
Budget/quality/clock/iteration precedence is documented in README. Best-candidate
coordinates remain an independent rounded copy throughout.

Record feasible iterations, every best replacement and its key/element errors,
initial metrics, and scalar push histories (never coordinate snapshots). Record
next-state damage, completed repair length (0 for immediate feasibility), rollback
reason, strength changes, before/after profile summaries, per-force mean/max
requested moves and actual moves. Per-atom checks add acceptance/rejection counts,
reasons and pass counts. Coordinate/index arrays remain private. Attempts count
toward the push budget even if rejected or rolled back. Always log new best candidates and
label final infeasible samples separately from the exported candidate. Expose
`refinement_updates`, `refinement_steps`, `rollback_count`, stopping limit and the
last evaluated geometry, retaining existing per-element reports.
Smoothing metadata controls plot legends, including when validation reads an
older generation report through its saved effective sigma.

VIPERdb requests use urllib with certificate verification by default. The explicit
`--no-check-certificate` option is limited to downloads and recorded in provenance;
it is never an automatic retry on certificate errors. Cache receipts bind the
recorded TLS mode to content hashes; an unchecked cached download is not silently
reused by a later verified request.

Schema-v1 generation JSON includes analysis settings, reference hash, profiles,
target geometry, effective mode, inventory, atom mapping, transforms, checks,
metrics, PDB hash and execution metadata. Independent validation recomputes the
analysis and checks provenance. Failed diagnostics use a distinct filename.

Not in v1: symmetry expansion, atom completion, force-field topology, resume,
GPU/distributed search, a general chemical-component database, or mmCIF output.
The search does not guarantee convergence, optimality, chemical realism or final
isotropy. Regularity describes initialization. Repeated-serial PDBs need VMD tests.

Acceptance at full-capsid scale still requires 1CWP runs: record content hash,
command, dependencies, hardware, timing, RSS, all constraints, errors and VMD atom
count. Preserve failure diagnostics and refine the search from measured behavior.

Revision 04 benchmarks use 1800 s and compare best feasible `errors_at_push` at
10/20/30/40/50 attempts plus the last reached attempt. best_history records the
cumulative push count, making the checkpoints independent of repair iteration
count. Record first feasibility, elapsed time, all final errors, both audits and
standalone validation. A time-limit warning explicitly states that a seed alone
cannot reproduce clock-limited results. The exact/dead-zone/disabled spring
comparison is recorded before tuning; do not replace a refuted diagnosis with
an assumed improvement. See `revision04` in the benchmark evidence.

Measured revision-04 outcomes: all ten main cases finish 50 pushes, all thirteen
PDBs (including spring controls) pass both independent audits and standalone
validation, and every unsmoothed placed run has S error 0. The main matrix's
smallest S-S distance is 2.789399 A, above 2.786177 A. Scientific expectations
remain unmet: every placed case has a C/N/O regression at equal 50-push budget,
and profile first feasibility is 455/248/121 versus off's 62/64/68. Controls
exact/dead-zone/disabled reach 531/455/63; the disabled case loses one anchor's
bin. A probe finds three ASP fragments next to one active CYS spring outlier
(N-S 3.49 A), suggesting a coupled local packing problem. Do not equate fewer
restrained S atoms or perfect S bin counts with faster geometry repair. The
smoothed case has S error 0.001207 against a different target and is recorded
separately. All outputs remain uncalibrated; the quality gates are explicit
false values in revision04 rather than hidden by successful validity checks.

The terminal logs tolerance provenance and effective per-element thresholds before
search or validation. Final acceptance logs compare each element's profile error
and count difference to the same limits used by acceptance(); Total remains a
diagnostic. A valid candidate ends search immediately, so refinement/time budgets
are ceilings, including when no pushes have occurred. Explicit zero budgets also
stop at first feasible geometry without changing its quality classification.
Validation reuses saved tolerances without requiring the calibration file.

Calibration plot bars are replicate p05-p95 ranges, dots are means, diamonds are
comparison shape errors, and triangles are explicitly supplied generated samples.
The dashed suggestion is a profile-only threshold, not proof of geometry or
chemical validity. No generated sample is inferred from the output prefix.

Revision 05 adds profile-only calibration. analyze(compute_distances=False) keeps
the identical selected reference, center, edges and smoothing path while omitting
all distance inference/override logic. Calibration never changes input coordinates
or invokes generation/repair. profile_errors uses count/volume densities and the
same per-element L1 integral and atom-weighted total as profile_metrics; the nine
v0.1.8 comparison errors reproduce the recorded metrics.

Counting uses multinomial(N_reference, target*volume/target_count). Orientation
computes each residue centroid with bincount and rotates only its offsets with
independent Haar SO(3) matrices. Placement additionally permutes centroid radii
across residues along their original directions. Reference counts are retained
for resized targets too, so these baselines can include an inventory mismatch;
there is no implicit remapping/rebuilding of a new reference. Means, population
SD (ddof=0), percentiles and extrema are seeded and reproducible at fixed R.
These model scales are not mathematical minima or experimental uncertainty.

Experimental comparisons use their own all-protein geometric center, reference
edges and both raw and shape-normalized errors. Shape divides by in-range count
before rescaling to target count; actual/in-range counts remain explicit. Missing
elements or zero support fail. Chain-label profiles use the reference center,
require all labels nonempty and >=1000 atoms, scale to full target counts and
report pairwise L1 spread separately. They do not raise suggestions automatically.
Generated PDBs use generated=True; sibling reports bind hashes/settings/center.
Ratios to zero means and nonpositive placement-orientation gaps are null with
reasons. gap_closed is raw (may exceed 1); its capped counterpart is also recorded.

Suggestions are ceil-to-three-significant-figures of K * floor statistic, raised
by the largest explicit experimental shape error. Defaults are orientation,
p95 and K=2; composition remains zero with the existing one-atom allowance.
Nothing is applied implicitly. The profile tolerance parser normalizes scalar,
per-element and wildcard forms to a dict and retains the original CLI string.
Coverage is checked before search; acceptance annotates each element and keeps
mandatory geometry dominant. Saved floats remain compatible. Calibration loading
checks content identity, model, width, effective sigma, radii, edges and selection
settings. Validation can reuse saved numeric tolerances without the calibration
file, or explicitly load a matching new report; it never rewrites the original.

Calibration outputs have a distinct _calibration suffix and use staged publish.
A prefix owned by a generation report is rejected even with overwrite. All
comparison/generated inputs and sibling reports are protected. CSV has fixed
columns and uses shape for experimental rows, with raw errors in JSON. PNG uses
independent log panels, population p05-p95 ranges/means and explicit zero display
positions; no Total acceptance tolerance is invented.

The first R=50 1CWP calibration finds orientation means C=.035818,N=.065827,
O=.051661,S=.369778 and counting means .006668/.013554/.013103/.072148.
The permutation placement total (.099487) differs from actual initialization
(~.06); it is a distinct null model. All ten old particles pass with explicitly
loaded K=2/p95 suggestions, while all mandatory audits remain independent.
C/O mean mode effects are below within-mode seed spread; N profile improvement
is ~1.4 pooled seed SD, with one paired reversal. The n=3 analysis is descriptive,
not a significance claim. Full results and raw/capped gaps live in the benchmark
JSON's calibration section. No second experimental full capsid was supplied.


## Revision 06: homogeneous step control arm

The reference model remains unchanged by default. `build_target` dispatches all
four CLI workflows to the same `Target`: edges, densities, an optional reference
on those edges, and a step report. Plain-array `fit_step` remains independent of
CLI/Analysis objects and can later fit weighted samples of other radial targets.

A step with half-height radii a,b and ramp width w has unit height h(r): it rises
linearly from 0 to 1 over [a-w/2,a+w/2], stays 1 through b-w/2, then falls to 0
at b+w/2. Hard steps use w=0. Nonoverlap and nonnegative support require
b-a>=w and a-w/2>=0. All target integrals are closed-form polynomials:
I_k = integral h(r) r^(2+k) dr, k=0..3. A midpoint polynomial expansion avoids
cancellation when integrating thin ramp fragments at large radii. No numerical
quadrature is used outside tests.

Count-weighted atom moments m1=<r>,m2=<r^2> determine a,b from I1/I0=m1 and
I2/I0=m2. Least squares uses relative residuals and nonnegative support-inner
radius/plateau-length parameters. Both residuals must be <1e-10 and the solution
must lie strictly inside parameter bounds. Atom weighting represents material
per spherical volume, whereas per-area and mass fits answer different questions;
they are sensitivity diagnostics only. Shared bounds give rho_e=N_e/(4*pi*I0),
so elemental composition is homogeneous in every bin. Analytical clipped shell
integrals conserve counts independently of ramp/bin alignment.

Fit once to selected reference atoms (after percentile trimming), then resize
with user half-height bounds, keeping w and rho_e fixed. Target counts scale by
I0_new/I0_fit; the existing integer inventory handles nonintegral desired counts.
Atom support is [a-w/2,b+w/2], used by geometry and profile checks. Diagnostic
hard, mass, per-area and per-element fits never override the count-weighted fit.
Skewness records the outer-tail information lost by the symmetric-wall control.

Reference comparison profiles must be re-binned on step edges at fitted size;
atoms outside support are reported. A resized step has no mapped reference curve.
Every caller, including search metrics and CSV/PNG, handles that absence. Step
plots draw their target even though Gaussian smoothing is disabled. Validation
reconstructs the target from saved settings, checks generated bounds/support to
1e-9 A, counts to relative 1e-9 and edges to 1e-9 A, before publishing a report.
Sulfur reference-radius placement is forbidden by both CLI and search API.

Calibration self-error measures reference-versus-step mismatch. Only counting
is a floor: orientation/placement would measure an intentionally removed model,
so they are explicitly skipped. Resized counting draws use rounded target counts,
not the original particle inventory. Orientation multiples and gap closure are
null with reasons. Suggestions use counting and exclude experimental variation,
which is reported separately. CSV/PNG omit skipped levels; step plots label the
reference mismatch. Model/edge width join settings checks, with missing keys
interpreted as the legacy reference model. The stricter counting-based quality
rule can yield approximate particles, while geometry remains a hard constraint.

Limitations: crystallographic disorder truncates the visible inner protein
(1CWP N-terminal arms), so both layered/reference and step controls omit that
material. The fitted b lies inside the geometric reference outer radius; FEM
indentation contact should use b and explicitly distinguish its ramp support.
No skin-core thickness rule, Gaussian smoothing change, or JSON size redesign
is included in this revision.
