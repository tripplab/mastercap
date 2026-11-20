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

Once geometry is feasible, bounded profile refinement is attempted even without
quality tolerances. Keep the best feasible candidate, rank errors by worst/mean
relative element error before aggregate error, and use volume-aware deterministic
quantiles as radial targets. Refinement can re-enter repair and is bounded by
the shared search limits; `--refine-iterations 0` selects first-feasible stopping.
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
