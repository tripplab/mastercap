"""Bounded geometric search. Failure to converge is not proof of infeasibility."""
from collections import defaultdict
from dataclasses import replace
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .model import Structure, MastercapError
from .analysis import shell_volumes
from .constraints import GeometryMetadata, local_geometry, validate_geometry, profile_metrics, acceptance


def residue_groups(structure):
    groups = defaultdict(list)
    for i, atom in enumerate(structure.atoms):
        groups[atom.residue].append(i)
    return [np.asarray(v, dtype=int) for v in groups.values()]


def select_inventory(analysis, edges, profiles, rng, time_limit=15.0):
    source = analysis.structure
    elements = sorted(profiles)
    target = np.array([np.sum(profiles[e] * shell_volumes(edges)) for e in elements])
    groups = residue_groups(source)
    buckets = defaultdict(list)
    for group in groups:
        signature = (source.atoms[group[0]].resname,
                     tuple((source.atoms[i].name, source.atoms[i].element) for i in group))
        buckets[signature].append(group)
    variants = list(buckets.values())
    matrix = np.array([[sum(source.atoms[i].element == e for i in v[0]) for v in variants]
                       for e in elements], dtype=float)
    scale = target.sum() / len(source.atoms)
    expected_variants = np.array([len(v) * scale for v in variants])
    # Exact unchanged inventories need no optimization or sampling replacement.
    if np.allclose(target, np.array([sum(source.elements == e) for e in elements]), atol=1e-7):
        chosen = groups
    else:
        # Integer counts, absolute element residuals, absolute variant residuals.
        n, m = len(variants), len(elements)
        rows = m + n
        design = np.block([[matrix, np.eye(m), -np.eye(m), np.zeros((m, 2*n))],
                           [np.eye(n), np.zeros((n, 2*m)), np.eye(n), -np.eye(n)]])
        objective = np.concatenate([np.zeros(n), 1/np.maximum(target, 1), 1/np.maximum(target, 1),
                                    0.1/np.maximum(expected_variants, 1), 0.1/np.maximum(expected_variants, 1)])
        rhs = np.concatenate([target, expected_variants])
        result = milp(objective, integrality=np.r_[np.ones(n), np.zeros(2*m+2*n)],
                      bounds=Bounds(np.zeros(n+2*m+2*n), np.full(n+2*m+2*n, np.inf)),
                      constraints=LinearConstraint(design, rhs, rhs),
                      options={"time_limit": max(0.01, time_limit), "mip_rel_gap": 0.001})
        if result.x is None:
            raise MastercapError("Inventory integer optimization did not find a feasible solution within its budget")
        counts = np.rint(result.x[:n]).astype(int)
        chosen = []
        for variant, count in zip(variants, counts):
            # Without replacement until exhausted; repeat exemplars only as needed.
            order = rng.permutation(len(variant))
            chosen.extend(variant[order[j % len(order)]] for j in range(count))
    if not chosen:
        raise MastercapError("Requested geometry produces an empty residue inventory")
    atoms, xyz, trace = [], [], []
    for residue, group in enumerate(chosen):
        for index in group:
            atom = source.atoms[index]
            origin = {**atom.origin, "reference_atom_index": int(index), "reference_residue": atom.residue}
            atoms.append(replace(atom, residue=residue, origin=origin))
            xyz.append(source.xyz[index])
        trace.append({"residue": residue, "reference_residue": source.atoms[group[0]].residue,
                      "resname": source.atoms[group[0]].resname, "atom_count": len(group)})
    return Structure(atoms, np.asarray(xyz), provenance=source.provenance.copy()), trace


def sample_radii(profile, edges, n, rng):
    weights = profile * shell_volumes(edges)
    if weights.sum() <= 0:
        raise MastercapError("Cannot sample an empty element profile")
    bins = rng.choice(len(weights), size=n, p=weights / weights.sum())
    # Uniform VOLUME within each constant-density shell.
    return (edges[bins]**3 + rng.random(n) * (edges[bins+1]**3-edges[bins]**3))**(1/3)


def profile_quantiles(profile, edges, n):
    """Deterministic, volume-aware radial targets without sampling noise."""
    if n == 0:
        return np.array([], dtype=float)
    weights = profile*shell_volumes(edges)
    if weights.sum() <= 0:
        raise MastercapError("Cannot refine an empty element profile")
    cumulative = np.cumsum(weights)/weights.sum()
    u = (np.arange(n)+0.5)/n
    bins = np.minimum(np.searchsorted(cumulative, u, side="right"), len(weights)-1)
    lower = np.r_[0.0, cumulative[:-1]]
    fraction = (u-lower[bins])/(cumulative[bins]-lower[bins])
    return (edges[bins]**3+fraction*(edges[bins+1]**3-edges[bins]**3))**(1/3)


def initialize(structure, analysis, edges, profiles, center, rng, mode, distribution, *, groups=None):
    groups = residue_groups(structure) if groups is None else groups
    n = len(groups)
    if distribution == "regular":
        k = np.arange(n)
        z = 1 - 2*(k+0.5)/n
        phi = k * np.pi * (3-np.sqrt(5))
        directions = np.column_stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z])
        rng.shuffle(directions)
    else:
        directions = rng.normal(size=(n, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
    # Residue centers use the aggregate radial profile; element-specific radial
    # transport in the repair stage refines the independent mode.
    total = sum(profiles.values())
    radial = sample_radii(total, edges, n, rng)
    if distribution == "regular":
        # Quantize centers to representative shell radii; repairs may move them.
        bins = np.clip(np.searchsorted(edges, radial)-1, 0, len(edges)-2)
        radial = (edges[bins]+edges[bins+1])/2
    rotations = Rotation.random(n, random_state=rng).as_matrix()
    trace = []
    for k, group in enumerate(groups):
        old_center = structure.xyz[group].mean(axis=0)
        target = center + directions[k]*radial[k]
        structure.xyz[group] = (structure.xyz[group]-old_center) @ rotations[k].T + target
        trace.append({"rotation": rotations[k].tolist(),
                      "translation": (target-old_center@rotations[k].T).tolist()})
    return trace


def connect_edges(xyz, labels, *, metadata=None, atom_indices=None, margin=0.0):
    """A sparse forest of nearby, compatible component-boundary contacts.

    Start with batched local queries. Components without a compatible outgoing
    proposal use an external-component query, not arbitrary representatives.
    The forest is a repair proposal, never evidence of actual connectivity.
    """
    _, labels = np.unique(labels, return_inverse=True)
    count = int(labels.max())+1 if len(labels) else 0
    if count <= 1:
        return np.empty((0, 2), dtype=int)
    if metadata is None:
        radii, codes, matrix = np.zeros(len(xyz)), None, None
    else:
        indices = np.arange(len(xyz)) if atom_indices is None else atom_indices
        radii, codes, matrix = metadata.radii[indices], metadata.codes[indices], metadata.lower_matrix
    has_contact = np.zeros(count, dtype=bool)
    contacts = {}

    def consider(src, dst):
        keep = labels[src] != labels[dst]
        src, dst = src[keep], dst[keep]
        if matrix is not None:
            compatible = matrix[codes[src], codes[dst]] < radii[src]+radii[dst]-margin-0.003
            src, dst = src[compatible], dst[compatible]
        if not len(src):
            return
        gaps = np.maximum(0.0, np.linalg.norm(xyz[src]-xyz[dst], axis=1)-radii[src]-radii[dst]+margin)
        component, other = labels[src], labels[dst]
        has_contact[component] = has_contact[other] = True
        keys = np.minimum(component, other)*count+np.maximum(component, other)
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        # Keep the best atom contact per component pair, so local alternatives
        # remain available to the spanning forest (without millions of forces).
        order = np.lexsort((hi, lo, gaps, keys))
        first = np.r_[True, np.diff(keys[order]) != 0]
        take = order[first]
        for index in take:
            item = (float(gaps[index]), int(lo[index]), int(hi[index]))
            key = int(keys[index])
            if key not in contacts or item < contacts[key]:
                contacts[key] = item

    tree = cKDTree(xyz)
    k = min(16, len(xyz))
    # Bound neighbor working memory independently of particle size.
    for start in range(0, len(xyz), 4096):
        source = np.arange(start, min(start+4096, len(xyz)))
        nearest = tree.query(xyz[source], k=k)[1].reshape(len(source), k)
        consider(np.repeat(source, k), nearest.ravel())
    for component in np.flatnonzero(~has_contact):
        if has_contact[component]:
            continue
        sources = np.flatnonzero(labels == component)
        outside = labels != component
        partitions = [sources] if codes is None else [sources[codes[sources] == c] for c in np.unique(codes[sources])]
        for part in partitions:
            allowed = outside.copy()
            if matrix is not None:
                allowed &= matrix[codes[part[0]], codes] < radii[part[0]]+radii-margin-0.003
            targets = np.flatnonzero(allowed)
            if not len(targets):
                continue
            nearest = cKDTree(xyz[targets]).query(xyz[part], k=1)[1]
            consider(part, targets[nearest])

    if not contacts:
        return np.empty((0, 2), dtype=int)
    parent = np.arange(count)

    def root(component):
        while parent[component] != component:
            parent[component] = parent[parent[component]]
            component = parent[component]
        return component

    forest = []
    for _, a, b in sorted(contacts.values()):
        ra, rb = root(labels[a]), root(labels[b])
        if ra != rb:
            parent[ra] = rb
            forest.append((a, b))
    return np.asarray(forest, dtype=int).reshape(-1, 2)


def pair_forces(force, xyz, pairs, desired, strength=0.5):
    if not len(pairs):
        return
    delta = xyz[pairs[:, 1]]-xyz[pairs[:, 0]]
    distance = np.linalg.norm(delta, axis=1)
    zero = distance < 1e-12
    delta[zero] = [1, 0, 0]
    unit = delta / np.maximum(distance, 1e-12)[:, None]
    unit[zero] = [1, 0, 0]
    move = strength * (distance-desired)[:, None] * unit
    np.add.at(force, pairs[:, 0], move)
    np.add.at(force, pairs[:, 1], -move)


def update_positions(structure, force, groups, mode):
    norm = np.linalg.norm(force, axis=1)
    force *= np.minimum(1, 0.2/np.maximum(norm, 1e-12))[:, None]
    if mode == "independent":
        structure.xyz += force
        return
    for group in groups:
        original = structure.xyz[group]
        center = original.mean(axis=0)
        wanted = original + force[group]
        target_center = wanted.mean(axis=0)
        # Kabsch projection: each update is rigid and preserves internal geometry.
        u, _, vt = np.linalg.svd((original-center).T @ (wanted-target_center))
        rotation = u @ np.diag([1, 1, np.linalg.det(u@vt)]) @ vt
        structure.xyz[group] = (original-center) @ rotation + target_center


def characteristic_forces(force, structure, analysis, metadata):
    """Weak nearest-neighbor statistical target, subordinate to feasibility."""
    for (a, b), lower in analysis.limits.items():
        summary = analysis.neighbor_samples["-".join((a, b))]["percentiles"]
        if "50" not in summary:
            continue
        ai, bi = metadata.element_indices[a], metadata.element_indices[b]
        if not len(ai) or not len(bi) or (a == b and len(ai) < 2):
            continue
        if a == b:
            nearest = cKDTree(structure.xyz[bi]).query(structure.xyz[ai], k=2)[1][:, 1]
        else:
            nearest = cKDTree(structure.xyz[bi]).query(structure.xyz[ai], k=1)[1]
        pairs = np.column_stack([ai, bi[nearest]])
        desired = np.full(len(pairs), max(lower+0.003, summary["50"]))
        pair_forces(force, structure.xyz, pairs, desired, strength=0.002)


def search(structure, analysis, edges, profiles, center, rng, mode="independent",
           distribution="random", max_iterations=1000, time_limit=300.0, margin=0.0,
           profile_tolerance=None, composition_tolerance=None, progress=None,
           progress_interval=5, refine_iterations=100, stage_progress=None):
    preparation_started = time.monotonic()
    if max_iterations < 0 or time_limit <= 0 or margin < 0 or refine_iterations < 0 or progress_interval < 1:
        raise MastercapError("Iterations, refinement steps and overlap margin must be nonnegative; time/progress interval positive")
    if stage_progress:
        stage_progress("Building inventory metadata and initializing particle coordinates")
    metadata = GeometryMetadata.build(structure, analysis.limits)
    groups = list(metadata.groups.values())
    rotations = initialize(structure, analysis, edges, profiles, center, rng, mode, distribution, groups=groups)
    if mode == "residue":
        check = local_geometry(structure, analysis.limits, margin, mode, metadata=metadata)
        if check["disconnected_residues"]:
            raise MastercapError("A required rigid residue template is internally disconnected")
    quantiles = {}
    for e, profile in profiles.items():
        indices = metadata.element_indices[e]
        quantiles[e] = profile_quantiles(profile, edges, len(indices))
    best, best_key = None, (float("inf"),)*4
    first_feasible = None
    profile_updates = 0
    feasible_iterations, push_history, best_history = [], [], []
    initial_metrics = None
    pending_push = None
    last_report = None
    preparation_seconds = time.monotonic()-preparation_started
    # The repair budget starts only after inventory-dependent initialization.
    started = time.monotonic()
    termination = "iteration_limit"
    if stage_progress:
        stage_progress("Starting geometric repair and element-profile search")
    for iteration in range(max_iterations+1):
        # All feasibility decisions use the same precision as exported PDB.
        candidate = Structure(structure.atoms, np.round(structure.xyz, 3),
                              structure.warnings, structure.provenance)
        local = local_geometry(candidate, analysis.limits, margin, mode, metadata=metadata)
        geometry = validate_geometry(candidate, center, edges, analysis.limits, margin, mode,
                                     local=local, metadata=metadata)
        metrics = profile_metrics(candidate, center, edges, profiles, analysis.original, metadata=metadata)
        if initial_metrics is None:
            initial_metrics = {"total_profile_error": metrics["total_profile_error"], "elements": metrics["elements"]}
        if pending_push is not None:
            if "damage" not in pending_push:
                pending_push["damage"] = {"iteration": iteration,
                    "distance_violations": geometry["distance_violations"],
                    "global_components": geometry["global_components"],
                    "disconnected_residues": len(geometry["disconnected_residues"]),
                    "outside_radial_bounds": geometry["outside_radial_bounds"],
                    "mandatory_pass": geometry["mandatory_pass"]}
            if geometry["mandatory_pass"]:
                pending_push["recovery_iterations"] = iteration-pending_push["iteration"]-1
                pending_push["recovered_at_iteration"] = iteration
                pending_push = None
        status = acceptance(geometry, metrics, profile_tolerance, composition_tolerance)
        last_report = {"geometry": geometry, "metrics": metrics, "status": status,
                       "iterations": iteration, "elapsed_seconds": time.monotonic()-started}
        if geometry["mandatory_pass"]:
            feasible_iterations.append(iteration)
            if first_feasible is None:
                first_feasible = iteration
                if stage_progress and refine_iterations and status != "valid":
                    stage_progress(f"Geometry feasible at iteration {iteration}; refining per-element profiles "
                                   f"for up to {refine_iterations} additional steps within remaining search limits")
            errors = [m["profile_error"] or 0.0 for m in metrics["elements"].values()]
            key = (max(errors), float(np.mean(errors)), metrics["total_profile_error"],
                   sum(abs(m["count_difference"]) for m in metrics["elements"].values()))
            if key < best_key:
                best, best_key = (candidate, last_report.copy()), key
                worst = max(metrics["elements"], key=lambda e: metrics["elements"][e]["profile_error"] or 0)
                best_history.append({"iteration": iteration, "ranking_key": list(key),
                    "total_profile_error": metrics["total_profile_error"], "max_element_error": max(errors),
                    "mean_element_error": float(np.mean(errors)), "count_difference": key[3],
                    "worst_element": worst,
                    "element_errors": {e: m["profile_error"] for e, m in metrics["elements"].items()}})
                if stage_progress:
                    stage_progress(f"New best feasible candidate at iteration {iteration}: "
                                   f"total={metrics['total_profile_error']:.6f}, max element={max(errors):.6f} ({worst})")
        reason = None
        if status == "valid" or (geometry["mandatory_pass"] and refine_iterations == 0):
            reason = "constraints_satisfied"
        elif first_feasible is not None and iteration-first_feasible >= refine_iterations:
            reason = "refinement_complete"
        elif time.monotonic()-started >= time_limit:
            reason = "time_limit"
        elif iteration == max_iterations:
            reason = "iteration_limit"
        if progress and (iteration % progress_interval == 0 or reason is not None):
            displayed = dict(geometry)
            if reason is not None and not geometry["mandatory_pass"]:
                displayed["final_search_state"] = True
                displayed["export_candidate_available"] = best is not None
            progress(iteration, displayed, metrics)
        if reason is not None:
            termination = reason
            break
        force = np.zeros_like(structure.xyz)
        repair = local["distances"] < local["lower"]+0.003
        if mode == "residue":
            residues = metadata.residues
            repair &= residues[local["pairs"][:, 0]] != residues[local["pairs"][:, 1]]
        pairs = local["pairs"][repair]
        lower = local["lower"][repair]
        # Safety spacing protects against 0.001 A PDB rounding.
        pair_forces(force, structure.xyz, pairs, lower+0.003, strength=0.6)
        # Internal graphs are repaired separately from the global graph.
        for residue in local["disconnected_residues"]:
            group = metadata.groups[residue]
            bridges = group[connect_edges(candidate.xyz[group], local["internal_labels"][group],
                                         metadata=metadata, atom_indices=group, margin=margin)]
            cap = metadata.radii[bridges].sum(axis=1)-margin-0.003
            pair_forces(force, structure.xyz, bridges, cap, strength=0.3)
        if local["components"] > 1:
            bridges = connect_edges(candidate.xyz, local["labels"], metadata=metadata, margin=margin)
            cap = metadata.radii[bridges].sum(axis=1)-margin-0.003
            pair_forces(force, structure.xyz, bridges, cap, strength=0.15)
        offset = structure.xyz-center
        radial = np.linalg.norm(offset, axis=1)
        direction = offset/np.maximum(radial, 1e-12)[:, None]
        direction[radial < 1e-12] = [1, 0, 0]
        clipped = np.clip(radial, edges[0]+0.002, edges[-1]-0.002)
        force += (clipped-radial)[:, None]*direction*0.8
        # Soft radial transport; rank by current radius for each element.
        if geometry["mandatory_pass"]:
            requested_moves = []
            for e, target_radial in quantiles.items():
                indices = metadata.element_indices[e]
                order = indices[np.argsort(radial[indices], kind="stable")]
                requested = 0.015*(target_radial-radial[order])
                force[order] += requested[:, None]*direction[order]
                requested_moves.append(np.abs(requested))
            requested_moves = np.concatenate(requested_moves)
            pending_push = {"iteration": iteration, "strength": 0.015, "selected_atoms": len(radial),
                            "atoms_pushed": int(np.count_nonzero(requested_moves)),
                            "mean_requested_radial_move_angstrom": float(np.mean(requested_moves)),
                            "max_requested_radial_move_angstrom": float(np.max(requested_moves)),
                            "recovery_iterations": None}
            push_history.append(pending_push)
            profile_updates += 1
        force += (center-structure.xyz.mean(axis=0))*0.1
        if geometry["mandatory_pass"]:
            characteristic_forces(force, structure, analysis, metadata)
        update_positions(structure, force, groups, mode)
        del local, force
    if best is not None:
        output, report = best
        report["candidate_iteration"] = report["iterations"]
    else:
        output, report = None, last_report
    report["elapsed_seconds"] = time.monotonic()-started
    report["preparation_seconds"] = preparation_seconds
    report["iterations"] = iteration
    report["termination"] = termination
    report["candidate_available"] = output is not None
    report["first_feasible_iteration"] = first_feasible
    report["refinement_steps"] = iteration-first_feasible if first_feasible is not None else 0
    report["profile_updates"] = profile_updates
    report.update({"feasible_iterations": feasible_iterations, "push_history": push_history,
                   "best_history": best_history, "initial_metrics": initial_metrics})
    report["profile_error_objective"] = "worst_element_then_mean"
    report["initial_transforms"] = rotations
    return output, report


def residue_transforms(structure, reference):
    result = []
    for group in residue_groups(structure):
        source = np.array([reference.xyz[structure.atoms[i].origin["reference_atom_index"]] for i in group])
        target = structure.xyz[group]
        a, b = source.mean(axis=0), target.mean(axis=0)
        u, _, vt = np.linalg.svd((source-a).T@(target-b))
        rotation = u@np.diag([1, 1, np.linalg.det(u@vt)])@vt
        fitted = (source-a)@rotation+b
        result.append({"residue": structure.atoms[group[0]].residue,
                       "rotation_row_vector": rotation.tolist(), "translation": (b-a@rotation).tolist(),
                       "rigid_fit_rmsd_angstrom": float(np.sqrt(np.mean(np.sum((target-fitted)**2, axis=1))))})
    return result
