"""Bounded geometric search. Failure to converge is not proof of infeasibility."""
from collections import defaultdict
from dataclasses import replace
import logging
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .model import Structure, MastercapError
from .analysis import shell_volumes
from .constraints import GeometryMetadata, local_geometry, validate_geometry, profile_metrics, acceptance
from .refinement import (PushController, apply_radial_push, candidate_key, error_summary,
                         slack_summary, consolidation_targets, AcceptanceController,
                         accept_trial, force_summary, crossing_metrics, classify_stall)


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


def characteristic_forces(force, structure, analysis, metadata, *, mode="fixed",
                          margin=0.0, guard=0.003, strength=0.002,
                          max_move=None, selected_indices=None):
    """Build a separate neighbor contribution with guarded contact targets.

    Guarding a target is not a guarantee for the combined update. In scaled
    mode the caller supplies the adaptive coefficient, selected atoms and cap;
    the cap is applied AFTER all element-pair contributions have accumulated.
    """
    if mode not in ("scaled", "off", "fixed"):
        raise MastercapError("Neighbor forces must be scaled, off or fixed")
    if mode == "off":
        return {"mode": mode, "mean_move_angstrom": 0.0, "max_move_angstrom": 0.0,
                "active_atoms": 0, "strength": 0.0}
    neighbor = np.zeros_like(force)
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
        safe_lower = lower+guard
        safe_upper = metadata.radii[pairs].sum(axis=1)-margin-guard
        valid = safe_lower < safe_upper
        pairs = pairs[valid]
        if not len(pairs):
            continue
        desired = np.clip(summary["50"], safe_lower, safe_upper[valid])
        pair_forces(neighbor, structure.xyz, pairs, desired, strength=strength)
    if selected_indices is not None:
        selected = np.zeros(len(neighbor), dtype=bool)
        selected[selected_indices] = True
        neighbor[~selected] = 0
    norm = np.linalg.norm(neighbor, axis=1)
    if max_move is not None:
        neighbor *= np.minimum(1.0, max_move/np.maximum(norm, 1e-12))[:, None]
        norm = np.linalg.norm(neighbor, axis=1)
    force += neighbor
    return {"mode": mode, "strength": float(strength), "active_atoms": int(np.count_nonzero(norm)),
            "mean_move_angstrom": float(norm.mean()), "max_move_angstrom": float(norm.max())}


def search(structure, analysis, edges, profiles, center, rng, mode="independent",
           distribution="random", max_iterations=1000, time_limit=300.0, margin=0.0,
           profile_tolerance=None, composition_tolerance=None, progress=None,
           progress_interval=5, refine_iterations=None, stage_progress=None,
           selection_objective="total", refine_updates=50, refine_patience=10,
           refine_strength=0.005, refine_max_step=0.05, refine_fraction=0.2,
           refine_recovery_limit=10, neighbor_forces="off", refine_slack=0.02,
           consolidate_iterations=10, refine_acceptance="per-atom",
           rollback_violation_fraction=0.05, rollback_component_limit=5000):
    preparation_started = time.monotonic()
    if (max_iterations < 0 or time_limit <= 0 or margin < 0 or progress_interval < 1
            or (refine_iterations is not None and refine_iterations < 0)
            or refine_updates < 0 or refine_patience < 0 or refine_recovery_limit < 0
            or consolidate_iterations < 0 or rollback_component_limit < 1):
        raise MastercapError("Iteration/update/patience/recovery limits and margin must be nonnegative; time/progress interval positive")
    if (not np.isfinite([refine_strength, refine_max_step, refine_fraction, refine_slack, rollback_violation_fraction]).all()
            or refine_strength <= 0 or refine_strength > np.finfo(float).max/4
            or refine_max_step <= 0 or not 0 < refine_fraction <= 1 or refine_slack < 0
            or not 0 <= rollback_violation_fraction <= 1):
        raise MastercapError("Refinement strength/max step must be finite and positive; fraction must be in (0, 1]")
    if selection_objective not in ("total", "worst-element"):
        raise MastercapError("Selection objective must be total or worst-element")
    if neighbor_forces not in ("scaled", "off", "fixed"):
        raise MastercapError("Neighbor forces must be scaled, off or fixed")
    if refine_acceptance not in ("per-atom", "rollback"):
        raise MastercapError("Refinement acceptance must be per-atom or rollback")
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
    phase = "repair"
    consolidation_steps = consolidation_failures = 0
    slack_first = slack_after = None
    slack_refinement = None
    consolidation_history, consolidation_episode = [], None
    consolidation_completed = False
    consolidation_iterations_total = 0
    profile_updates = 0
    feasible_iterations, push_history, best_history = [], [], []
    restored_iterations = []
    restored_snapshot = False
    initial_metrics = None
    controller = (AcceptanceController(refine_strength, push_history) if refine_acceptance == "per-atom"
                  else PushController(refine_strength, refine_recovery_limit, len(structure.atoms), push_history,
                                      rollback_violation_fraction, rollback_component_limit))
    last_significant_key = None
    last_improvement_update = 0
    improvement_relative = 0.001
    stop_limit = None
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
        summary = error_summary(metrics)
        rolled_back = controller.observe(iteration, geometry, structure.xyz,
                                         candidate_key(metrics, selection_objective), summary)
        if rolled_back and stage_progress:
            record = push_history[-1]
            stage_progress(f"Rollback push from iteration {record['iteration']} at iteration {iteration}: "
                           f"{record['rollback_reason']}; next radial strength={controller.strength:.6g}; snapshot restored")
        status = acceptance(geometry, metrics, profile_tolerance, composition_tolerance)
        last_report = {"geometry": geometry, "metrics": metrics, "status": status,
                       "iterations": iteration, "elapsed_seconds": time.monotonic()-started}
        if restored_snapshot:
            restored_iterations.append(iteration)
        if geometry["mandatory_pass"] and not restored_snapshot and not rolled_back:
            feasible_iterations.append(iteration)
            if first_feasible is None:
                first_feasible = iteration
                slack_first = slack_summary(local, metadata, margin, mode)
                if stage_progress and refine_updates and refine_iterations != 0 and status != "valid":
                    stage_progress(f"Geometry feasible at iteration {iteration}; refining per-element profiles "
                                   f"for up to {refine_updates} pushes within remaining search limits")
            key = candidate_key(metrics, selection_objective)
            if key < best_key:
                best, best_key = (candidate, last_report.copy()), key
                summary = error_summary(metrics)
                best_history.append({"iteration": iteration, "ranking_key": list(key), **summary})
                if (last_significant_key is None or
                        last_significant_key[0]-key[0] > max(1e-12, abs(last_significant_key[0])*improvement_relative)):
                    last_significant_key = key
                    last_improvement_update = profile_updates
                if stage_progress:
                    stage_progress(f"New best feasible candidate at iteration {iteration}: "
                                   f"total={summary['total_profile_error']:.6f}, "
                                   f"max element={summary['max_element_error']:.6f} ({summary['worst_element']}); "
                                   f"objective={selection_objective}")
        if first_feasible is not None and geometry["mandatory_pass"] and phase == "repair":
            phase = "consolidate" if consolidate_iterations and consolidation_failures <= 3 else "refine"
            if phase == "consolidate":
                consolidation_episode = {"start_iteration": iteration, "end_iteration": None,
                                         "steps": 0, "trigger": "initial", "reason": None,
                                         "before": slack_summary(local, metadata, margin, mode), "after": None}
                consolidation_history.append(consolidation_episode)
            if phase == "consolidate" and stage_progress:
                stage_progress(f"Starting consolidation at iteration {iteration}: slack={refine_slack:g} A")
        if phase == "consolidate":
            if not geometry["mandatory_pass"]:
                consolidation_failures += 1
                consolidation_steps = 0
                phase = "repair"
                consolidation_episode.update(end_iteration=iteration, reason="interrupted",
                                             after=slack_summary(local, metadata, margin, mode))
                if stage_progress:
                    detail = (f"restart {consolidation_failures}/3" if consolidation_failures <= 3
                              else "restart budget exhausted; repair then refine")
                    stage_progress(f"Consolidation interrupted; repairing geometry ({detail})")
                    if consolidation_failures > 3:
                        stage_progress(f"Consolidation abandoned at iteration {iteration}; refining without slack after repair")
            else:
                gaps = slack_summary(local, metadata, margin, mode)
                ready = all(gaps[k] is None or gaps[k] >= refine_slack/2
                            for k in ("distance_gap_min_angstrom", "contact_gap_min_angstrom"))
                if ready or consolidation_steps >= consolidate_iterations:
                    phase, slack_after = "refine", gaps
                    consolidation_completed = True
                    consolidation_episode.update(end_iteration=iteration, after=gaps,
                                                 reason="target" if ready else "iteration_limit")
                    if stage_progress:
                        stage_progress(f"Consolidation complete at iteration {iteration}; starting profile refinement")
        if phase == "refine" and slack_refinement is None:
            slack_refinement = slack_summary(local, metadata, margin, mode)
        reason = None
        if status == "valid" or (geometry["mandatory_pass"] and (refine_iterations == 0 or refine_updates == 0)):
            reason = "constraints_satisfied"
        elif first_feasible is not None and profile_updates >= refine_updates:
            reason = "refinement_complete"
            stop_limit = "refine_updates"
        elif first_feasible is not None and refine_iterations is not None and iteration-first_feasible >= refine_iterations:
            reason = "refinement_complete"
            stop_limit = "refine_iterations"
        elif controller.blocked:
            reason, stop_limit = "refinement_blocked", "damage_independent_of_push"
            evidence = controller.blocked_evidence
            logging.getLogger(__name__).warning("Refinement blocked: damage did not decrease with push strength "
                               f"(strength {evidence['first_strength']:g}->{evidence['last_strength']:g}, "
                               f"damage {evidence['first_damage']}->{evidence['last_damage']}); check neighbor forces/slack")
        elif controller.stalled or (first_feasible is not None and refine_patience > 0
                                    and profile_updates-last_improvement_update >= refine_patience):
            reason = "refinement_stalled"
            stop_limit = "strength_floor" if controller.stalled else "refine_patience"
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
        if rolled_back:
            # All graphs describe the evaluated, damaged coordinates. The next
            # iteration checks the restored snapshot; never use stale graphs.
            del local
            restored_snapshot = True
            continue
        restored_snapshot = False
        pieces = {name: np.zeros_like(structure.xyz) for name in
                  ("repair", "bridges", "radial_clip", "centering", "radial_push", "nearest_neighbor")}
        force = pieces["repair"]
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
            pair_forces(pieces["bridges"], structure.xyz, bridges, cap, strength=0.3)
        if local["components"] > 1:
            bridges = connect_edges(candidate.xyz, local["labels"], metadata=metadata, margin=margin)
            cap = metadata.radii[bridges].sum(axis=1)-margin-0.003
            pair_forces(pieces["bridges"], structure.xyz, bridges, cap, strength=0.15)
        offset = structure.xyz-center
        radial = np.linalg.norm(offset, axis=1)
        direction = offset/np.maximum(radial, 1e-12)[:, None]
        direction[radial < 1e-12] = [1, 0, 0]
        clipped = np.clip(radial, edges[0]+0.002, edges[-1]-0.002)
        pieces["radial_clip"][:] = (clipped-radial)[:, None]*direction*0.8
        pieces["centering"][:] = (center-structure.xyz.mean(axis=0))*0.1
        if phase == "consolidate":
            near, desired, touching, contact = consolidation_targets(local, metadata, refine_slack, margin, mode)
            pair_forces(force, structure.xyz, near, desired, strength=0.3)
            pair_forces(pieces["bridges"], structure.xyz, touching, contact, strength=0.15)
            consolidation_steps += 1
            consolidation_iterations_total += 1
            consolidation_episode["steps"] += 1
        force = sum(pieces.values())
        before_push = None
        # Soft radial transport; rank by current radius for each element.
        if geometry["mandatory_pass"] and phase == "refine":
            push_force = pieces["radial_push"]
            push_source = candidate if refine_acceptance == "per-atom" else structure
            push_offset = push_source.xyz-center
            push_radial = np.linalg.norm(push_offset, axis=1)
            push_direction = push_offset/np.maximum(push_radial, 1e-12)[:, None]
            push_direction[push_radial < 1e-12] = [1, 0, 0]
            details, selected = apply_radial_push(push_force, push_radial, push_direction, metadata.element_indices, quantiles,
                                                  controller.strength, refine_max_step, refine_fraction)
            scaled = neighbor_forces == "scaled"
            neighbor_details = characteristic_forces(
                pieces["nearest_neighbor"], push_source, analysis, metadata, mode=neighbor_forces, margin=margin, guard=refine_slack,
                strength=0.002*(controller.strength/refine_strength) if scaled else 0.002,
                max_move=refine_max_step/2 if scaled else None,
                selected_indices=selected if scaled else None)
            details["neighbor_forces"] = neighbor_details
            details["profile_before"] = summary
            details["force_components"] = {name: force_summary(values) for name, values in pieces.items()}
            details["applied_force_components"] = (["radial_push", "nearest_neighbor"] if refine_acceptance == "per-atom"
                                                   else list(pieces))
            push_force = push_force+pieces["nearest_neighbor"]
            if refine_acceptance == "per-atom":
                trial = Structure(structure.atoms, candidate.xyz.copy())
                update_positions(trial, push_force, groups, mode)
                accepted, checked = accept_trial(candidate.xyz, trial.xyz, local, metadata, center, edges,
                                                 margin, mode, refine_max_step, selected_indices=selected)
                for e, values in checked.pop("per_element").items():
                    details["per_element"].setdefault(e, {}).update(values)
                crossings = crossing_metrics(push_radial, np.linalg.norm(accepted-center, axis=1),
                                             metadata.element_indices, edges, profiles)
                for e, values in crossings.items():
                    details["per_element"].setdefault(e, {}).update(values)
                details.update(checked)
                details["actual_move"] = force_summary(accepted-candidate.xyz)
                controller.start(iteration, candidate.xyz, details, candidate_key(metrics, selection_objective))
                structure.xyz[:] = accepted
            else:
                force += push_force
                before_push = structure.xyz.copy()
                controller.start(iteration, structure.xyz, details, candidate_key(metrics, selection_objective))
            profile_updates += 1
            if refine_acceptance == "per-atom":
                del local, force, push_force, pieces
                continue
        update_positions(structure, force, groups, mode)
        if before_push is not None:
            push_history[-1]["actual_move"] = force_summary(structure.xyz-before_push)
        del local, force, pieces
    controller.finish()
    if consolidation_episode is not None and consolidation_episode["end_iteration"] is None:
        consolidation_episode.update(end_iteration=iteration, reason="search_stopped",
                                     after=slack_summary(local, metadata, margin, mode))
    last_search_geometry = last_report["geometry"]
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
    report["refinement_updates"] = profile_updates
    completed_recoveries = [p["recovery_iterations"] for p in push_history if p["recovery_iterations"] is not None]
    report["median_recovery_iterations"] = float(np.median(completed_recoveries)) if completed_recoveries else None
    report.update({"feasible_iterations": feasible_iterations, "push_history": push_history,
                   "best_history": best_history, "initial_metrics": initial_metrics})
    report["profile_error_objective"] = selection_objective
    report.update({"slack_at_first_feasible": slack_first, "slack_after_consolidation": slack_after,
                   "slack_at_refinement_start": slack_refinement,
                   "consolidation_completed": consolidation_completed,
                   "consolidation_iterations_total": consolidation_iterations_total,
                   "consolidation_history": consolidation_history,
                   "stall_cause": classify_stall(push_history, refine_patience) if stop_limit == "refine_patience" else None,
                   "consolidation_restarts": min(consolidation_failures, 3),
                   "consolidation_failures": consolidation_failures, "search_phase": phase})
    report.update({"restored_snapshot_iterations": restored_iterations,
                   "blocked_evidence": getattr(controller, "blocked_evidence", None)})
    fractions = [p["acceptance_fraction"] for p in push_history if "acceptance_fraction" in p]
    report.update({"acceptance_mode": refine_acceptance,
                   "committed_pushes": sum(p.get("atoms_moved", p.get("outcome") == "recovered_improved") > 0 for p in push_history),
                   "mean_acceptance_fraction": float(np.mean(fractions)) if fractions else None,
                   "recoveries_improved": sum(p["outcome"] == "recovered_improved" for p in push_history),
                   "recoveries_worse": sum(p["outcome"] == "recovered_worse" for p in push_history),
                   "repair_attempts": sum(p.get("recovery_iterations", 0) not in (None, 0)
                                          or p.get("rollback_after_repair_iterations", 0) > 0 for p in push_history)})
    report.update({"rollback_count": controller.rollbacks, "refinement_strength_final": controller.strength,
                   "refinement_stop_limit": stop_limit, "last_search_geometry": last_search_geometry,
                   "last_significant_improvement_update": last_improvement_update,
                   "refinement_settings": {"updates": refine_updates, "iteration_cap": refine_iterations,
                       "patience": refine_patience, "improvement_relative": improvement_relative,
                       "starting_strength": refine_strength, "minimum_strength": controller.minimum,
                       "maximum_strength": controller.maximum, "fraction": refine_fraction,
                       "max_step_angstrom": refine_max_step, "recovery_limit": refine_recovery_limit,
                       "neighbor_forces": neighbor_forces,
                       "refine_slack_angstrom": refine_slack, "consolidate_iterations": consolidate_iterations,
                       "acceptance": refine_acceptance,
                       "big_damage_distance_fraction": rollback_violation_fraction,
                       "big_damage_component_limit": rollback_component_limit}})
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
