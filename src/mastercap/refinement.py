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
                      strength, max_step, fraction, min_step=0.0):
    """Select the largest radial mismatches per element; cap requested moves.

    These are radial contributions BEFORE rigid projection, other forces, and
    the overall update_positions cap. The adaptive controller observes the
    resulting combined move, not just this contribution. Return scalar move
    statistics and selected indices; indices remain private to the search.
    """
    sizes, moves, selected, per_element = [], [], [], {}
    for element, target in targets.items():
        indices = element_indices[element]
        if not len(indices):
            continue
        order = indices[np.argsort(radial[indices], kind="stable")]
        delta = target-radial[order]
        count = min(len(order), max(1, int(np.ceil(fraction*len(order)))))
        chosen = np.argsort(-np.abs(delta), kind="stable")[:count]
        requested = minimum_moves(delta[chosen], strength, min_step, max_step)
        force[order[chosen]] += requested[:, None]*direction[order[chosen]]
        selected.append(order[chosen])
        sizes.append(count)
        per_element[element] = {"selected": count}
        moves.append(np.abs(requested))
    absolute = np.concatenate(moves) if moves else np.array([])
    details = {"strength": float(strength), "selected_atoms": sum(sizes),
            "atoms_pushed": int(np.count_nonzero(absolute)),
            "mean_requested_radial_move_angstrom": float(absolute.mean()) if len(absolute) else 0.0,
            "max_requested_radial_move_angstrom": float(absolute.max()) if len(absolute) else 0.0,
            "fraction": float(fraction), "max_step_angstrom": float(max_step), "per_element": per_element}
    return details, np.concatenate(selected) if selected else np.array([], dtype=int)


def minimum_moves(delta, strength, min_step, max_step):
    requested = strength*delta
    small = (np.abs(delta) >= min_step) & (np.abs(requested) > 0) & (np.abs(requested) < min_step)
    requested[small] = np.sign(requested[small])*min_step
    return np.clip(requested, -max_step, max_step)


def bin_inventory(radial, element_indices, edges, profiles):
    """Integer observations and composition-normalized target counts per bin."""
    from .analysis import shell_volumes
    bins, volume = radial_bins(radial, edges), shell_volumes(edges)
    result = {}
    for e, indices in element_indices.items():
        valid = bins[indices]
        valid = valid[(valid >= 0) & (valid < len(volume))]
        counts = np.bincount(valid, minlength=len(volume)).astype(float)
        target = np.asarray(profiles[e])*volume
        if target.sum():
            target = target*(len(indices)/target.sum())
        result[e] = (counts, target, counts-target)
    return bins, result


def resolve_drift_step(value, bin_width, min_step=.002, max_step=.05):
    if not np.isfinite([bin_width,min_step,max_step]).all() or bin_width<=0 or max_step<=0 or not 0<=min_step<=max_step:
        raise MastercapError('Require positive bin/max step and 0 <= min step <= max step')
    if value == 'legacy':return None
    if value == 'auto':return max(min_step,min(max_step,.02*bin_width))
    try:step=float(value)
    except (ValueError,TypeError):raise MastercapError('Refine drift step must be auto, legacy, or a positive A value') from None
    if not np.isfinite(step) or step<=0 or step>max_step:
        raise MastercapError('Refine drift step must be positive and no larger than refine max step')
    return step


def effective_patience(patience, bin_width, drift_step):
    return max(patience,2*int(np.ceil(.5*bin_width/drift_step))) if patience and drift_step is not None else patience


def apply_crossing_push(force, radial, direction, element_indices, profiles, edges,
                        strength, max_step, fraction, min_step=0.002, blocked=None, anchor_bins=None, drift_step="auto", transport_factor=2.0, routing="cumulative"):
    """Reserve cumulative (or legacy local) edge quotas, then select near-edge drifters."""
    drift_step = resolve_drift_step(drift_step, float(np.mean(np.diff(edges))), min_step, max_step)
    if not np.isfinite(transport_factor) or transport_factor<=0:
        raise MastercapError("Refine transport factor must be positive")
    if routing not in ('local','cumulative'):raise MastercapError('Transport routing must be local or cumulative')
    if drift_step is None:routing='local'
    def quota_for(balance,edge,source,destination,sign):
        if routing=='cumulative':
            flux=float(np.sum(balance[:edge]))*sign
            return flux if flux>=1-1e-8 else 0.
        return min(balance[source],-balance[destination]) if balance[source]>0 and balance[destination]<0 else 0.
    bins, inventory = bin_inventory(radial, element_indices, edges, profiles)
    blocked = np.zeros(len(radial), dtype=bool) if blocked is None else blocked
    used = np.zeros(len(radial), dtype=bool)
    anchor_target = np.full(len(radial), -1, dtype=int)
    if anchor_bins is not None:
        anchor_target[anchor_bins[0]] = anchor_bins[1]
    selected, absolute, per_element = [], [], {}
    all_drift_distances=[]
    for e, indices in element_indices.items():
        _, _, balance = inventory[e]
        capacity = int(np.floor(fraction*len(indices)))
        chosen, requested = [], []
        crossing_count = 0
        for edge in range(1, len(edges)-1):
            for source, destination, sign in ((edge-1, edge, 1), (edge, edge-1, -1)):
                remaining=quota_for(balance,edge,source,destination,sign)
                if remaining<=0:continue
                quota = int(np.floor(remaining+1e-8))
                count = min(quota, capacity-len(chosen))
                if count <= 0:
                    continue
                candidates = indices[(bins[indices] == source) & ~blocked[indices] & ~used[indices]]
                distance = np.abs(radial[candidates]-edges[edge])
                order = np.argsort(distance, kind="stable")
                order = order[distance[order] <= max_step][:count]
                take = candidates[order]
                move = sign*np.minimum(distance[order]+0.002, max_step)
                chosen.extend(take.tolist())
                requested.extend(move.tolist())
                used[take] = True
                balance[source] -= len(take)
                balance[destination] += len(take)
                crossing_count += len(take)
        # Prefer the nearest useful edge, deduplicating atoms with two choices.
        drift_indices, drift_distance, drift_sign = [], [], []
        for edge in range(1, len(edges)-1):
            for source, destination, sign in ((edge-1, edge, 1), (edge, edge-1, -1)):
                remaining=quota_for(balance,edge,source,destination,sign)
                if remaining<=0:continue
                candidates = indices[(bins[indices] == source) & ~blocked[indices] & ~used[indices]]
                distance = np.abs(radial[candidates]-edges[edge])
                eligible = np.flatnonzero(distance > max_step)
                if drift_step is not None:
                    quota=int(np.ceil(transport_factor*remaining-1e-8))
                    eligible=eligible[np.argsort(distance[eligible],kind="stable")[:quota]]
                drift_indices.append(candidates[eligible])
                drift_distance.append(distance[eligible])
                drift_sign.append(np.full(len(eligible), sign))
        if drift_indices and len(chosen) < capacity:
            ix, distance, sign = map(np.concatenate, (drift_indices, drift_distance, drift_sign))
            order = np.argsort(distance, kind="stable")
            _, first = np.unique(ix[order], return_index=True)
            order = order[np.sort(first)][:capacity-len(chosen)]
            take = ix[order]
            move = (minimum_moves(sign[order]*distance[order], strength, min_step, max_step)
                    if drift_step is None else sign[order]*np.minimum(distance[order]+.002,drift_step))
            all_drift_distances.extend(distance[order].tolist())
            chosen.extend(take.tolist())
            requested.extend(move.tolist())
            used[take] = True
        chosen, requested = np.asarray(chosen, dtype=int), np.asarray(requested, dtype=float)
        destination = radial_bins(radial[chosen]+requested, edges)
        bad_anchor = (anchor_target[chosen] >= 0) & (destination != anchor_target[chosen])
        anchor_rejected = int(bad_anchor.sum())
        crossing_count -= int(bad_anchor[:crossing_count].sum())
        chosen, requested = chosen[~bad_anchor], requested[~bad_anchor]
        force[chosen] += requested[:, None]*direction[chosen]
        selected.append(chosen)
        absolute.append(np.abs(requested))
        per_element[e] = {"selected": len(chosen), "crossing_proposals": crossing_count,
                          "drift_proposals": len(chosen)-crossing_count,
                          "anchor_bin_selection_rejections": anchor_rejected}
    moves = np.concatenate(absolute) if absolute else np.array([])
    chosen = np.concatenate(selected) if selected else np.array([], dtype=int)
    return {"strength": float(strength), "selected_atoms": len(chosen),
            "atoms_pushed": int(np.count_nonzero(moves)),
            "mean_requested_radial_move_angstrom": float(moves.mean()) if len(moves) else 0.0,
            "max_requested_radial_move_angstrom": float(moves.max()) if len(moves) else 0.0,
            "fraction": float(fraction), "max_step_angstrom": float(max_step), "per_element": per_element,
            "transport_routing":routing, "mean_distance_to_edge_of_drifters":float(np.mean(all_drift_distances)) if all_drift_distances else None,
            "drift_proposals":sum(p['drift_proposals'] for p in per_element.values()),
            "crossing_proposals":sum(p['crossing_proposals'] for p in per_element.values())}, chosen


def damage_summary(iteration, geometry):
    return {"iteration": iteration, "distance_violations": geometry["distance_violations"],
            "global_components": geometry["global_components"],
            "disconnected_residues": len(geometry["disconnected_residues"]),
            "outside_radial_bounds": geometry["outside_radial_bounds"],
            "isolated_atoms": geometry["isolated_atoms"],
            "center_error_angstrom": geometry["center_error_angstrom"],
            "mandatory_pass": geometry["mandatory_pass"]}


def slack_summary(local, metadata, margin=0.0, mode="independent", slack=0.02):
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
                       f"{name}_within_0_005_angstrom": int(np.count_nonzero(values < 0.005)),
                       f"{name}_within_half_slack": int(np.count_nonzero(values < slack/2))})
    return result


def consolidation_targets(local, metadata, slack, margin=0.0, mode="independent", noncontact_slack=True):
    """Return pairs and nonconflicting guarded targets for slack building."""
    pairs, distance, lower = local["pairs"], local["distances"], local["lower"]
    upper = metadata.radii[pairs].sum(axis=1)-margin
    window = upper-lower
    guard = np.where((window > 0) & (window < 2*slack), window/3, slack)
    if not noncontact_slack:
        guard = np.where(window <= 0, 0, guard)
    eligible = np.ones(len(pairs), dtype=bool)
    if mode == "residue":
        eligible &= metadata.residues[pairs[:, 0]] != metadata.residues[pairs[:, 1]]
    close = eligible & (distance < lower+guard)
    touching = eligible & (window > 0) & (distance < upper) & (distance > upper-guard)
    return (pairs[close], (lower+guard)[close], pairs[touching], (upper-guard)[touching])


def force_summary(force):
    lengths = np.linalg.norm(force, axis=1)
    return {"mean_move_angstrom": float(lengths.mean()), "max_move_angstrom": float(lengths.max())}


def radial_bins(radial, edges):
    """Match numpy.histogram, including its closed rightmost bin."""
    bins = np.searchsorted(edges, radial, side="right")-1
    bins[radial == edges[-1]] = len(edges)-2
    return bins


def crossing_metrics(before, after, element_indices, edges, profiles):
    from .analysis import shell_volumes
    source, destination = radial_bins(before, edges), radial_bins(after, edges)
    result = {}
    n = len(edges)-1
    for e, indices in element_indices.items():
        a, b = source[indices], destination[indices]
        counts = np.bincount(a[(a >= 0) & (a < n)], minlength=n).astype(float)
        target = profiles[e]*shell_volumes(edges)
        improved = worsened = neutral = 0
        changed = (a != b) & (a >= 0) & (a < n) & (b >= 0) & (b < n)
        for old, new in zip(a[changed], b[changed]):
            error_before = abs(counts[old]-target[old])+abs(counts[new]-target[new])
            counts[old] -= 1
            counts[new] += 1
            gain = error_before-abs(counts[old]-target[old])-abs(counts[new]-target[new])
            improved += gain > 1e-9
            worsened += gain < -1e-9
            neutral += abs(gain) <= 1e-9
        result[e] = {"bin_crossings": int(changed.sum()), "crossings_improved": int(improved),
                     "crossings_worsened": int(worsened), "crossings_neutral": int(neutral)}
    return result


def finish_push_metrics(record, summary):
    if summary is None or not record.get("profile_before"):
        return
    for e, after in summary["element_errors"].items():
        record.setdefault("per_element", {}).setdefault(e, {})["profile_error_change"] = (
            after-record["profile_before"]["element_errors"][e])


def classify_stall(history, patience):
    window = history[-patience:] if patience else history
    if (window and not any(p.get('bin_crossings',sum(v.get('bin_crossings',0) for v in p.get('per_element',{}).values())) for p in window)
            and any(p.get('drift_proposals',sum(v.get('drift_proposals',0) for v in p.get('per_element',{}).values())) for p in window)):
        return 'transport_starved'
    if window and np.mean([p.get("actual_move", {}).get("mean_move_angstrom", 0) for p in window]) < 0.001:
        return "moves_below_rounding"
    if window and np.mean([p.get("acceptance_fraction", 1) for p in window]) < 0.3:
        return "low_acceptance"
    return "converged"


def accept_trial(snapshot, trial, local, metadata, center, edges, margin=0.0,
                 mode="independent", max_step=0.05, selected_indices=None, return_undone=False,
                 anchor_bins=None, backtrack=False):
    """Reject conflicting moved atoms/residues while retaining every old contact.

    Candidate pairs cover trial/trial AND trial/snapshot combinations, since
    reverting one end can create a new collision. No component calculation is
    needed: preserving the snapshot edges preserves both connectivity graphs.
    """
    backtrack=bool(backtrack and mode=='independent')
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
                code_a,code_b=metadata.codes[ai[0]],metadata.codes[bi[0]]
                bound = metadata.lower_matrix[code_a,code_b]
                if metadata.inter_residue_lower is not None:bound=max(bound,metadata.inter_residue_lower[code_a,code_b])
                bound+=.003
                # Partial trial positions lie between endpoints: cover the full
                # movement envelope for new floors/backtracking, preserving legacy.
                if backtrack or metadata.inter_residue_enabled:bound+=2*max_step
                if bound > radius:
                    collect(source, (bi, positions[bi]), bound)
    pairs = np.unique(np.vstack(blocks), axis=0)
    contacts = local["overlap_edges"]
    lower = metadata.pair_lower_bounds(pairs)
    upper = metadata.radii[contacts].sum(axis=1)-margin-0.003
    eligible = np.ones(len(pairs), dtype=bool)
    if mode == "residue":
        eligible &= metadata.residues[pairs[:, 0]] != metadata.residues[pairs[:, 1]]
    reasons = {name: np.zeros(len(snapshot), dtype=bool)
               for name in ("too_close", "contact_broken", "out_of_bounds", "anchor_bin")}
    pair_rejections = {}

    def offenders():
        moving = np.any(trial != snapshot, axis=1)
        reject = np.zeros(len(snapshot), dtype=bool)
        distance = np.linalg.norm(trial[pairs[:, 0]]-trial[pairs[:, 1]], axis=1)
        bad = eligible & moving[pairs].any(axis=1) & (distance < lower+0.003)
        offending_pairs = pairs[bad]
        if len(offending_pairs):
            types = np.sort(metadata.codes[offending_pairs], axis=1)
            for kind in np.unique(types, axis=0):
                affected = np.unique(offending_pairs[np.all(types == kind, axis=1)])
                affected = affected[moving[affected]]
                example = offending_pairs[np.flatnonzero(np.all(types == kind, axis=1))[0]]
                name = "-".join(sorted(metadata.elements[example]))
                pair_rejections.setdefault(name, set()).update(affected.tolist())
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
        if anchor_bins is not None:
            ix, bins = anchor_bins
            wrong = ix[moving[ix] & (radial_bins(radial[ix], edges) != bins)]
            reasons['anchor_bin'][wrong] = True
            reject[wrong] = True
        reasons["out_of_bounds"] |= outside
        reject |= outside
        reject &= moving
        if mode == "residue" and reject.any():
            reject = np.isin(metadata.residues, np.unique(metadata.residues[reject])) & moving
        return reject

    backtracked=np.zeros(len(snapshot),dtype=bool)
    backtracking_rounds=0
    passes = 0
    for passes in range(1, 8 if backtrack else 4):
        reject = offenders()
        if not reject.any():
            break
        if backtrack and passes<=4:
            backtracked |= reject
            backtracking_rounds+=1
            trial[reject]=np.round((trial[reject]+snapshot[reject])/2,3)
        else:
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
    proposal_count = int(proposals.sum())
    undone = proposals & ~retained
    guard_reasons={name:int(mask.sum()) for name,mask in reasons.items()} if backtrack else None
    if backtrack:
        reasons={name:mask & undone for name,mask in reasons.items()}
        pair_rejections={pair:{i for i in indices if undone[i]} for pair,indices in pair_rejections.items()}
    per_element = {e: {"accepted": int(retained[ix].sum()), "undone": int(undone[ix].sum()),
                       "undone_too_close": int(reasons["too_close"][ix].sum()),
                       "undone_contact_broken": int(reasons["contact_broken"][ix].sum())}
                   for e, ix in metadata.element_indices.items()}
    by_pair = {name: len(indices) for name, indices in sorted(pair_rejections.items()) if indices}
    for e in per_element:
        per_element[e]["undone_too_close_by_pair"] = {
            name: element_count for name, indices in sorted(pair_rejections.items())
            if (element_count := int(np.count_nonzero(metadata.elements[np.fromiter(indices, dtype=int)] == e)))}
    details = {"atoms_proposed": proposal_count, "atoms_accepted": int(retained.sum()),
                   "atoms_undone": int(proposal_count-retained.sum()), "atoms_moved": int(accepted.sum()),
                   "proposed_moving_atoms": int(proposed.sum()),
                   "stationary_proposals": int(np.count_nonzero(proposals & ~proposed)),
                   "undone_by_reason": {name: int(mask.sum()) for name, mask in reasons.items()},
                   "undone_too_close_by_pair": by_pair,
                   "acceptance_fraction": float(retained.sum()/proposal_count) if proposal_count else 1.0,
                   "check_passes": passes, "expanded_undo_atoms": expanded,
                   "safety_reverted_all": safety_reverted_all, "center_rejected": center_rejected,
                   "per_element": per_element}
    if backtrack:
        details.update(backtracked_atoms=int(backtracked.sum()),backtracked_accepted_atoms=int((backtracked & accepted).sum()),
                       backtracking_rounds=backtracking_rounds,backtracking_guard_reasons=guard_reasons)
    return (trial, details, undone) if return_undone else (trial, details)


class AcceptanceController:
    """Adapt to accepted moves; retain a snapshot only as a validation safety net."""

    def __init__(self, strength, history, control="fixed"):
        if control not in ("fixed", "gain", "acceptance"):
            raise MastercapError("Strength control must be fixed, gain or acceptance")
        self.strength, self.minimum, self.maximum = strength, strength/16, 4*strength
        self.history, self.pending, self.rollbacks = history, None, 0
        self.blocked = False
        self.control, self.no_gain = control, 0

    @property
    def stalled(self):
        return self.control != "fixed" and self.strength < self.minimum

    def start(self, iteration, xyz, details, key=None):
        record = {"iteration": iteration, **details, "rollback": False,
                  "recovery_iterations": None, "outcome": "pending"}
        self.history.append(record)
        self.pending = record, xyz.copy(), key
        fraction = details["acceptance_fraction"]
        if self.control == "acceptance" and fraction > 0.85:
            self.strength = min(self.maximum, self.strength*1.25)
        elif self.control == "acceptance" and fraction < 0.5:
            self.strength *= 0.7
        record["strength_after"] = self.strength

    def observe(self, iteration, geometry, xyz, key=None, summary=None):
        if self.pending is None:
            return False
        record, snapshot, before_key = self.pending
        record.update({"damage": damage_summary(iteration, geometry), "profile_after": summary})
        finish_push_metrics(record, summary)
        self.pending = None
        if geometry["mandatory_pass"]:
            if self.control == "gain" and before_key is not None and key is not None:
                gain = before_key[0]-key[0]
                record["primary_profile_gain"] = float(gain)
                self.no_gain = self.no_gain+1 if gain <= 0 else 0
                if self.no_gain >= 2:
                    self.strength *= 0.7
                    self.no_gain = 0
                elif gain > 0 and record["acceptance_fraction"] > 0.85:
                    self.strength = min(self.maximum, self.strength*1.25)
                record["strength_after"] = self.strength
            record.update({"outcome": "accepted", "recovery_iterations": 0})
            return False
        xyz[:] = snapshot
        self.rollbacks += 1
        if self.control != "fixed":
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
        if "damage" not in record:
            finish_push_metrics(record, summary)
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


def errors_at_push(best_history, push_history):
    """Best feasible errors after equal numbers of attempted pushes (not time)."""
    reached = len(push_history)
    checkpoints = sorted({n for n in (10,20,30,40,50,reached) if n <= reached})
    result = {}
    for count in checkpoints:
        eligible = []
        for record in best_history:
            updates = record.get('profile_updates')
            if updates is None:
                updates = sum(push['iteration'] < record['iteration'] for push in push_history)
            if updates <= count:
                eligible.append(record)
        if eligible:
            best = eligible[-1]
            result[str(count)] = {k: best[k] for k in ('iteration','total_profile_error','element_errors')}
            result[str(count)]['push_count'] = count
    return result


def restore_anchor_bins(xyz, snapshot, center, edges, anchor_bins, residues=None):
    """Rollback-mode bin protection uses the precision of the next PDB candidate."""
    indices, target = anchor_bins
    radial = np.linalg.norm(np.round(xyz[indices],3)-center,axis=1)
    wrong = indices[radial_bins(radial,edges) != target]
    undo = wrong if residues is None else np.flatnonzero(np.isin(residues,residues[wrong]))
    xyz[undo] = snapshot[undo]
    return len(wrong)
