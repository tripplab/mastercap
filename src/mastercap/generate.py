"""Bounded geometric search. Failure to converge is not proof of infeasibility."""
from collections import Counter, defaultdict
from dataclasses import replace
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .model import Structure, MastercapError
from .analysis import shell_volumes
from .constraints import local_geometry, validate_geometry, profile_metrics, acceptance


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


def initialize(structure, analysis, edges, profiles, center, rng, mode, distribution):
    groups = residue_groups(structure)
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


def connect_edges(xyz, labels):
    """Sparse nearby-component bridges; spanning forest, not a claim of connectivity."""
    if len(np.unique(labels)) <= 1:
        return np.empty((0, 2), dtype=int)
    # Bounded nearest-neighbor graph. Increase k only when no cross-component
    # edges are found; large separated components use representative bridges.
    n = len(xyz)
    k = min(16, n)
    _, neighbors = cKDTree(xyz).query(xyz, k=k)
    neighbors = np.asarray(neighbors).reshape(n, k)
    src = np.repeat(np.arange(n), k)
    dst = neighbors.ravel()
    keep = labels[src] != labels[dst]
    pairs = np.column_stack([src[keep], dst[keep]])
    # Include a representative spanning tree to connect separated clusters even
    # when each cluster has more than k internal neighbors.
    _, representatives = np.unique(labels, return_index=True)
    reps = xyz[representatives]
    kr = min(8, len(reps))
    _, nearest = cKDTree(reps).query(reps, k=kr)
    nearest = np.asarray(nearest).reshape(len(reps), kr)
    a = np.repeat(np.arange(len(reps)), kr)
    b = nearest.ravel()
    distance = np.linalg.norm(reps[a]-reps[b], axis=1)
    graph = coo_matrix((np.maximum(distance, 1e-12), (a, b)), shape=(len(reps), len(reps))).tocsr()
    mst = minimum_spanning_tree(graph + graph.T).tocoo()
    bridges = np.column_stack([representatives[mst.row], representatives[mst.col]])
    # At least a chain of representatives also bridges disjoint kNN subgraphs.
    chain = np.column_stack([representatives[:-1], representatives[1:]])
    return np.unique(np.sort(np.vstack([pairs, bridges, chain]), axis=1), axis=0)


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


def characteristic_forces(force, structure, analysis):
    """Weak nearest-neighbor statistical target, subordinate to feasibility."""
    elements = structure.elements
    for (a, b), lower in analysis.limits.items():
        summary = analysis.neighbor_samples["-".join((a, b))]["percentiles"]
        if "50" not in summary:
            continue
        ai, bi = np.flatnonzero(elements == a), np.flatnonzero(elements == b)
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
           profile_tolerance=None, composition_tolerance=None, started=None, progress=None):
    started = time.monotonic() if started is None else started
    if max_iterations < 0 or time_limit <= 0 or margin < 0:
        raise MastercapError("Iterations and overlap margin must be nonnegative; time limit positive")
    groups = residue_groups(structure)
    rotations = initialize(structure, analysis, edges, profiles, center, rng, mode, distribution)
    if mode == "residue":
        check = local_geometry(structure, analysis.limits, margin, mode)
        if check["disconnected_residues"]:
            raise MastercapError("A required rigid residue template is internally disconnected")
    elements = structure.elements
    quantiles = {}
    for e, profile in profiles.items():
        indices = np.flatnonzero(elements == e)
        # Fixed order statistics avoid introducing fresh noise each iteration.
        quantiles[e] = np.sort(sample_radii(profile, edges, len(indices), rng))
    best, best_key = None, (float("inf"), float("inf"))
    last_report = None
    for iteration in range(max_iterations+1):
        # All feasibility decisions use the same precision as exported PDB.
        candidate = structure.copy()
        candidate.xyz = np.round(candidate.xyz, 3)
        geometry = validate_geometry(candidate, center, edges, analysis.limits, margin, mode)
        metrics = profile_metrics(candidate, center, edges, profiles, analysis.original)
        status = acceptance(geometry, metrics, profile_tolerance, composition_tolerance)
        last_report = {"geometry": geometry, "metrics": metrics, "status": status,
                       "iterations": iteration, "elapsed_seconds": time.monotonic()-started}
        if geometry["mandatory_pass"]:
            key = (metrics["total_profile_error"], sum(abs(m["count_difference"]) for m in metrics["elements"].values()))
            if key < best_key:
                best, best_key = (candidate, last_report.copy()), key
            if status == "valid" or status == "uncalibrated":
                break
        if progress and iteration % 25 == 0:
            progress(iteration, geometry, metrics)
        if iteration == max_iterations or time.monotonic()-started >= time_limit:
            break
        local = local_geometry(structure, analysis.limits, margin, mode)
        force = np.zeros_like(structure.xyz)
        repair = local["distances"] < local["lower"]+0.003
        if mode == "residue":
            residues = structure.residues
            repair &= residues[local["pairs"][:, 0]] != residues[local["pairs"][:, 1]]
        pairs = local["pairs"][repair]
        lower = local["lower"][repair]
        # Safety spacing protects against 0.001 A PDB rounding.
        pair_forces(force, structure.xyz, pairs, lower+0.003, strength=0.6)
        # Internal graphs are repaired separately from the global graph.
        for group in groups:
            labels = local["internal_labels"][group]
            if len(np.unique(labels)) > 1:
                bridges = group[connect_edges(structure.xyz[group], labels)]
                cap = structure.radii[bridges].sum(axis=1)-margin-0.003
                if np.any(cap <= 0):
                    raise MastercapError("Overlap margin makes a required residue connection impossible")
                pair_forces(force, structure.xyz, bridges, cap, strength=0.3)
        if local["components"] > 1:
            bridges = connect_edges(structure.xyz, local["labels"])
            cap = structure.radii[bridges].sum(axis=1)-margin-0.003
            lower = np.array([analysis.limits[tuple(sorted((elements[a], elements[b])))] for a, b in bridges])
            allowed = lower < cap
            pair_forces(force, structure.xyz, bridges[allowed], cap[allowed], strength=0.15)
        offset = structure.xyz-center
        radial = np.linalg.norm(offset, axis=1)
        direction = offset/np.maximum(radial, 1e-12)[:, None]
        direction[radial < 1e-12] = [1, 0, 0]
        clipped = np.clip(radial, edges[0]+0.002, edges[-1]-0.002)
        force += (clipped-radial)[:, None]*direction*0.8
        # Soft radial transport; rank by current radius for each element.
        for e, target_radial in quantiles.items():
            indices = np.flatnonzero(elements == e)
            order = indices[np.argsort(radial[indices], kind="stable")]
            if geometry["mandatory_pass"]:
                force[order] += 0.015*(target_radial-radial[order])[:, None]*direction[order]
        force += (center-structure.xyz.mean(axis=0))*0.1
        if geometry["mandatory_pass"]:
            characteristic_forces(force, structure, analysis)
        update_positions(structure, force, groups, mode)
    if best is not None:
        output, report = best
    else:
        output, report = None, last_report
    report["elapsed_seconds"] = time.monotonic()-started
    report["termination"] = "feasible_candidate" if output is not None else "search_budget_exhausted"
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
