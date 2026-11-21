"""Candidate ranking, bounded partial radial pushes, and damage feedback."""
import numpy as np
from scipy.spatial import cKDTree

from .model import MastercapError


def error_summary(metrics):
    errors = {e: float(m["profile_error"] or 0.0) for e, m in metrics["elements"].items()}
    worst = max(errors, key=errors.get)
    return {"total_profile_error": float(metrics["total_profile_error"]),
            "mean_element_error": float(np.mean(list(errors.values()))),
            "max_element_error": errors[worst], "worst_element": worst,
            "count_difference": float(sum(abs(m["count_difference"]) for m in metrics["elements"].values())),
            "element_errors": errors}


def candidate_key(metrics, objective="total"):
    summary = error_summary(metrics)
    if objective == "total":
        names = ("total_profile_error", "mean_element_error", "max_element_error", "count_difference")
    elif objective == "worst-element":
        names = ("max_element_error", "mean_element_error", "total_profile_error", "count_difference")
    else:
        raise MastercapError("Selection objective must be total or worst-element")
    return tuple(summary[name] for name in names)


def apply_radial_push(force, radial, direction, element_indices, targets,
                      strength, max_step, fraction):
    """Select the largest radial mismatches per element; cap requested moves.

    These are radial contributions BEFORE rigid projection, other forces, and
    the overall update_positions cap. The adaptive controller observes the
    resulting combined move, not just this contribution. Return scalar move
    statistics and selected indices; indices remain private to the search.
    """
    sizes, moves, selected = [], [], []
    for element, target in targets.items():
        indices = element_indices[element]
        if not len(indices):
            continue
        order = indices[np.argsort(radial[indices], kind="stable")]
        delta = target-radial[order]
        count = min(len(order), max(1, int(np.ceil(fraction*len(order)))))
        chosen = np.argsort(-np.abs(delta), kind="stable")[:count]
        requested = np.clip(strength*delta[chosen], -max_step, max_step)
        force[order[chosen]] += requested[:, None]*direction[order[chosen]]
        selected.append(order[chosen])
        sizes.append(count)
        moves.append(np.abs(requested))
    absolute = np.concatenate(moves) if moves else np.array([])
    details = {"strength": float(strength), "selected_atoms": sum(sizes),
            "atoms_pushed": int(np.count_nonzero(absolute)),
            "mean_requested_radial_move_angstrom": float(absolute.mean()) if len(absolute) else 0.0,
            "max_requested_radial_move_angstrom": float(absolute.max()) if len(absolute) else 0.0,
            "fraction": float(fraction), "max_step_angstrom": float(max_step)}
    return details, np.concatenate(selected) if selected else np.array([], dtype=int)


def damage_summary(iteration, geometry):
    return {"iteration": iteration, "distance_violations": geometry["distance_violations"],
            "global_components": geometry["global_components"],
            "disconnected_residues": len(geometry["disconnected_residues"]),
            "outside_radial_bounds": geometry["outside_radial_bounds"],
            "isolated_atoms": geometry["isolated_atoms"],
            "center_error_angstrom": geometry["center_error_angstrom"],
            "mandatory_pass": geometry["mandatory_pass"]}


def slack_summary(local, metadata, margin=0.0, mode="independent"):
    """Measure margins on the already computed rounded-coordinate graph."""
    pairs = local["pairs"]
    gaps = local["distances"]-local["lower"]
    if mode == "residue":
        gaps = gaps[metadata.residues[pairs[:, 0]] != metadata.residues[pairs[:, 1]]]
    edges = local["overlap_edges"]
    contact = (metadata.radii[edges].sum(axis=1)-margin
               -np.linalg.norm(local["xyz"][edges[:, 0]]-local["xyz"][edges[:, 1]], axis=1))
    result = {}
    for name, values in (("distance", gaps), ("contact", contact)):
        result.update({f"{name}_gap_min_angstrom": float(values.min()) if len(values) else None,
                       f"{name}_gap_p05_angstrom": float(np.percentile(values, 5)) if len(values) else None,
                       f"{name}_within_0_005_angstrom": int(np.count_nonzero(values < 0.005))})
    return result


def consolidation_targets(local, metadata, slack, margin=0.0, mode="independent"):
    """Return pairs and nonconflicting guarded targets for slack building."""
    pairs, distance, lower = local["pairs"], local["distances"], local["lower"]
    upper = metadata.radii[pairs].sum(axis=1)-margin
    window = upper-lower
    guard = np.where(window < 2*slack, np.maximum(window, 0)/3, slack)
    eligible = np.ones(len(pairs), dtype=bool)
    if mode == "residue":
        eligible &= metadata.residues[pairs[:, 0]] != metadata.residues[pairs[:, 1]]
    close = eligible & (distance < lower+guard)
    touching = eligible & (window > 0) & (distance < upper) & (distance > upper-guard)
    return (pairs[close], (lower+guard)[close], pairs[touching], (upper-guard)[touching])


def force_summary(force):
    lengths = np.linalg.norm(force, axis=1)
    return {"mean_move_angstrom": float(lengths.mean()), "max_move_angstrom": float(lengths.max())}


def accept_trial(snapshot, trial, local, metadata, center, edges, margin=0.0,
                 mode="independent", max_step=0.05, selected_indices=None):
    """Reject conflicting moved atoms/residues while retaining every old contact.

    Candidate pairs cover trial/trial AND trial/snapshot combinations, since
    reverting one end can create a new collision. No component calculation is
    needed: preserving the snapshot edges preserves both connectivity graphs.
    """
    trial = np.round(trial, 3)
    proposed = np.any(trial != snapshot, axis=1)
    proposals = proposed.copy()
    if selected_indices is not None:
        proposals[selected_indices] = True
        if mode == "residue":
            proposals = np.isin(metadata.residues, np.unique(metadata.residues[proposals]))
    moved = np.flatnonzero(proposed)
    radius = 2*metadata.radii.max()+2*max_step
    blocks = [local["pairs"][proposed[local["pairs"]].any(axis=1)]]

    def collect(source, target, reach):
        if not len(source) or not len(target[0]):
            return
        tree = cKDTree(target[1])
        neighbors = tree.query_ball_point(trial[source], reach)
        sizes = np.fromiter((len(n) for n in neighbors), dtype=int, count=len(source))
        if sizes.sum():
            pairs = np.column_stack([np.repeat(source, sizes), target[0][np.concatenate(neighbors).astype(int)]])
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            blocks.append(np.sort(pairs, axis=1))

    indices = np.arange(len(snapshot))
    for positions in (trial, snapshot):
        collect(moved, (indices, positions), radius)
        # Preserve the rare-element large-bound optimization of local_geometry.
        for a, ai in metadata.element_indices.items():
            source = ai[proposed[ai]]
            for b, bi in metadata.element_indices.items():
                bound = metadata.lower_matrix[metadata.codes[ai[0]], metadata.codes[bi[0]]]+0.003
                if bound > radius:
                    collect(source, (bi, positions[bi]), bound)
    pairs = np.unique(np.vstack(blocks), axis=0)
    contacts = local["overlap_edges"]
    lower = metadata.lower_matrix[metadata.codes[pairs[:, 0]], metadata.codes[pairs[:, 1]]]
    upper = metadata.radii[contacts].sum(axis=1)-margin-0.003
    eligible = np.ones(len(pairs), dtype=bool)
    if mode == "residue":
        eligible &= metadata.residues[pairs[:, 0]] != metadata.residues[pairs[:, 1]]
    reasons = {name: np.zeros(len(snapshot), dtype=bool)
               for name in ("too_close", "contact_broken", "out_of_bounds")}

    def offenders():
        moving = np.any(trial != snapshot, axis=1)
        reject = np.zeros(len(snapshot), dtype=bool)
        distance = np.linalg.norm(trial[pairs[:, 0]]-trial[pairs[:, 1]], axis=1)
        bad = eligible & moving[pairs].any(axis=1) & (distance < lower+0.003)
        close = np.unique(pairs[bad])
        reasons["too_close"][close] |= moving[close]
        reject[close] = True
        length = np.linalg.norm(trial-snapshot, axis=1)
        distance = np.linalg.norm(trial[contacts[:, 0]]-trial[contacts[:, 1]], axis=1)
        broken = contacts[(distance >= upper) & moving[contacts].any(axis=1)]
        if len(broken):
            left, right = broken.T
            tie = np.isclose(length[left], length[right], rtol=0.05, atol=1e-9)
            undo = np.concatenate([left[(length[left] > length[right]) | tie],
                                   right[(length[right] > length[left]) | tie]])
            reasons["contact_broken"][undo] |= moving[undo]
            reject[undo] = True
        radial = np.linalg.norm(trial-center, axis=1)
        outside = moving & ((radial < edges[0]) | (radial > edges[-1]))
        reasons["out_of_bounds"] |= outside
        reject |= outside
        reject &= moving
        if mode == "residue" and reject.any():
            reject = np.isin(metadata.residues, np.unique(metadata.residues[reject])) & moving
        return reject

    passes = 0
    for passes in range(1, 4):
        reject = offenders()
        if not reject.any():
            break
        trial[reject] = snapshot[reject]
    reject = offenders()
    expanded = 0
    if reject.any():
        adjacent = pairs[reject[pairs].any(axis=1)].ravel()
        reject[adjacent] = True
        if mode == "residue":
            reject = np.isin(metadata.residues, np.unique(metadata.residues[reject]))
        expanded = int(np.count_nonzero(reject & np.any(trial != snapshot, axis=1)))
        trial[reject] = snapshot[reject]
    safety_reverted_all = bool(offenders().any())
    center_rejected = bool(np.linalg.norm(trial.mean(axis=0)-center) > 1.0)
    if safety_reverted_all or center_rejected:
        trial[:] = snapshot
    accepted = proposed & np.any(trial != snapshot, axis=1)
    retained = proposals & ~(proposed & ~accepted)
    if safety_reverted_all or center_rejected:
        retained[:] = False
    count = int(proposals.sum())
    return trial, {"atoms_proposed": count, "atoms_accepted": int(retained.sum()),
                   "atoms_undone": int(count-retained.sum()), "atoms_moved": int(accepted.sum()),
                   "proposed_moving_atoms": int(proposed.sum()),
                   "stationary_proposals": int(np.count_nonzero(proposals & ~proposed)),
                   "undone_by_reason": {name: int(mask.sum()) for name, mask in reasons.items()},
                   "acceptance_fraction": float(retained.sum()/count) if count else 1.0,
                   "check_passes": passes, "expanded_undo_atoms": expanded,
                   "safety_reverted_all": safety_reverted_all, "center_rejected": center_rejected}


class AcceptanceController:
    """Adapt to accepted moves; retain a snapshot only as a validation safety net."""

    def __init__(self, strength, history):
        self.strength, self.minimum, self.maximum = strength, strength/16, 4*strength
        self.history, self.pending, self.rollbacks = history, None, 0
        self.blocked = False

    @property
    def stalled(self):
        return self.strength < self.minimum

    def start(self, iteration, xyz, details, key=None):
        record = {"iteration": iteration, **details, "rollback": False,
                  "recovery_iterations": None, "outcome": "pending"}
        self.history.append(record)
        self.pending = record, xyz.copy()
        fraction = details["acceptance_fraction"]
        if fraction > 0.85:
            self.strength = min(self.maximum, self.strength*1.25)
        elif fraction < 0.5:
            self.strength *= 0.7
        record["strength_after"] = self.strength

    def observe(self, iteration, geometry, xyz, key=None, summary=None):
        if self.pending is None:
            return False
        record, snapshot = self.pending
        record.update({"damage": damage_summary(iteration, geometry), "profile_after": summary})
        self.pending = None
        if geometry["mandatory_pass"]:
            record.update({"outcome": "accepted", "recovery_iterations": 0})
            return False
        xyz[:] = snapshot
        self.rollbacks += 1
        self.strength *= 0.5
        record.update({"rollback": True, "rollback_reason": "validation_safety_net",
                       "outcome": "rolled_back", "strength_after": self.strength})
        return True

    def finish(self):
        if self.pending is not None:
            self.pending[0]["outcome"] = "unverified_at_stop"
            self.pending = None


class PushController:
    """Keep only one private coordinate snapshot; history contains scalars only."""

    def __init__(self, strength, recovery_limit, atom_count, history,
                 violation_fraction=0.05, component_limit=5000):
        self.strength = float(strength)
        self.maximum = 4*strength
        self.minimum = strength/16
        self.recovery_limit = recovery_limit
        self.atom_count = atom_count
        self.history = history
        self.pending = None
        self.rollbacks = 0
        self.violation_fraction, self.component_limit = violation_fraction, component_limit
        self.blocked = False
        self.blocked_evidence = None
        self.recent_damage = []

    @property
    def stalled(self):
        return self.strength < self.minimum

    def start(self, iteration, xyz, details, key=None):
        record = {"iteration": iteration, **details, "recovery_iterations": None,
                  "rollback": False, "outcome": "pending"}
        self.history.append(record)
        self.pending = (record, xyz.copy(), key)

    def observe(self, iteration, geometry, xyz, key=None, summary=None):
        """Return True when coordinates were restored (skip the current move)."""
        if self.pending is None:
            return False
        record, snapshot, snapshot_key = self.pending
        record.setdefault("profile_after", summary)
        recovery_age = iteration-record["iteration"]-1
        big = (geometry["distance_violations"] > self.violation_fraction*self.atom_count
               or geometry["global_components"] >= self.component_limit)
        if "damage" not in record:
            record["damage"] = damage_summary(iteration, geometry)
            record["next_state_classification"] = ("fine" if geometry["mandatory_pass"] else "big" if big else "small")
        recovered_worse = (geometry["mandatory_pass"] and key is not None
                           and snapshot_key is not None and key >= snapshot_key)
        if geometry["mandatory_pass"] and not recovered_worse:
            record["profile_recovered"] = summary
            record["recovery_iterations"] = recovery_age
            record["recovered_at_iteration"] = iteration
            record["outcome"] = "recovered_improved" if snapshot_key is not None else "feasible" if recovery_age == 0 else "recovered"
            if recovery_age == 0:
                self.strength = min(self.maximum, self.strength*1.25)
            record["strength_after"] = self.strength
            self.pending = None
            return False
        if recovered_worse or big or recovery_age >= self.recovery_limit:
            xyz[:] = snapshot
            if not recovered_worse:
                self.recent_damage.append((record["strength"], geometry["distance_violations"]+geometry["global_components"]))
                self.recent_damage = self.recent_damage[-6:]
                first_strength, first_damage = self.recent_damage[0]
                latest_strength, latest_damage = self.recent_damage[-1]
                if latest_strength <= first_strength/4 and latest_damage > first_damage*0.75:
                    self.blocked = True
                    self.blocked_evidence = {"reason": "damage_independent_of_push",
                                             "first_strength": first_strength, "last_strength": latest_strength,
                                             "first_damage": first_damage, "last_damage": latest_damage}
            if not self.blocked:
                self.strength *= 0.5
            self.rollbacks += 1
            record.update({"rollback": True, "outcome": "recovered_worse" if recovered_worse else "rolled_back", "rollback_iteration": iteration,
                           "rollback_after_repair_iterations": recovery_age,
                           "rollback_reason": "recovered_worse" if recovered_worse else "big_damage" if big else "recovery_limit",
                           "rollback_geometry": damage_summary(iteration, geometry),
                           "strength_after": self.strength})
            if geometry["mandatory_pass"]:
                record["profile_recovered"] = summary
                record["recovery_iterations"] = recovery_age
            self.pending = None
            return True
        return False

    def finish(self):
        if self.pending is not None:
            self.pending[0]["outcome"] = "unrecovered_at_stop"
            self.pending = None
