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

Use deterministic volume-aware radial quantiles, rank all atoms, and push the
largest-mismatch fraction per element (default 0.2). Radial strength starts at
0.005 and requested radial moves are capped at 0.05 A. Repair forces are computed
before profile contributions in the rollback path. First feasibility starts
best-effort consolidation: up to 10 updates build 0.02 A distance/contact slack,
using strengths 0.3/0.15 and one-third-window guards when the window is narrow.
Radial clipping and centering continue. No profile/neighbor force is applied.
Broken geometry returns to repair; at most three consolidation restarts are
allowed. Record minimum/5th-percentile gaps and counts within 0.005 A at first
feasibility and after consolidation. Final validation rules do not use slack.

Nearest-neighbor contributions run in the feasible refinement block. The default `scaled` mode uses coefficient
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
Acceptance >85% grows strength by 1.25; <50% shrinks by 0.7. Its denominator
counts selected proposals, including feasible stationary rounded proposals,
expanded to complete residues in rigid mode. Actual moving/retained atoms are
reported separately. Rounded zero-move trials count toward budget/patience; rejected proposals
also consume a push. A private snapshot provides a validation safety net.

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
