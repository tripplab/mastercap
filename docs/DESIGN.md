# Technical design and implementation map

This implementation follows the approved Q&A specification. Scientific quality
thresholds remain provisional until calibrated on expanded CCMV 1CWP.

| Module | Responsibility |
|---|---|
| `model.py` | Records, internal identities, protein recognition, fixed vdW table. |
| `pdbio.py` | PDB/gzip/models, filtering, VIPERdb cache, TER writer. |
| `analysis.py` | Center, bounds, volume-aware profiles, smoothing, pair statistics. |
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

Reference neighbor statistics retain their measured values. Analysis separately
applies canonical, recorded pair-limit overrides and returns the original pooled
normalized factor. S-S defaults to that factor times 3.60 A: reference sulfur
spacing is a symmetry/sparsity statistic, not a useful minimum-contact floor.
Explicit numeric/steric pairs are excluded from statistical neighbor forces;
reference mode restores the original limit. Validation rebuilds analysis with
the generation config's effective overrides. A legacy report without override
records explicitly selects reference limits. No saved-analysis loader exists.

Optional sulfur-residue initialization selects by S element and anchors the first
S of each residue. It preserves the existing seed's directions/rotations, then
assigns shuffled S quantile radii (bin midpoints for regular). Solve the positive
root of the offset equation for residue-center distance, translating the whole
rotated residue. Other residues sample aggregate target density minus the placed
residues' contribution. Off and S-free paths preserve all old RNG draws. Private
anchor arrays support a 0.05 radial spring before first feasibility only; rigid
projection applies normally. Reports retain summary counts, target-bin counts,
S errors and anchor deviations/hit fractions.

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
