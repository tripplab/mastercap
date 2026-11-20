"""Candidate ranking, bounded partial radial pushes, and damage feedback."""
import numpy as np

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
    resulting combined move, not just this contribution.
    """
    sizes, moves = [], []
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
        sizes.append(count)
        moves.append(np.abs(requested))
    absolute = np.concatenate(moves) if moves else np.array([])
    return {"strength": float(strength), "selected_atoms": sum(sizes),
            "atoms_pushed": int(np.count_nonzero(absolute)),
            "mean_requested_radial_move_angstrom": float(absolute.mean()) if len(absolute) else 0.0,
            "max_requested_radial_move_angstrom": float(absolute.max()) if len(absolute) else 0.0,
            "fraction": float(fraction), "max_step_angstrom": float(max_step)}


def damage_summary(iteration, geometry):
    return {"iteration": iteration, "distance_violations": geometry["distance_violations"],
            "global_components": geometry["global_components"],
            "disconnected_residues": len(geometry["disconnected_residues"]),
            "outside_radial_bounds": geometry["outside_radial_bounds"],
            "isolated_atoms": geometry["isolated_atoms"],
            "center_error_angstrom": geometry["center_error_angstrom"],
            "mandatory_pass": geometry["mandatory_pass"]}


class PushController:
    """Keep only one private coordinate snapshot; history contains scalars only."""

    def __init__(self, strength, recovery_limit, atom_count, history):
        self.strength = float(strength)
        self.maximum = 4*strength
        self.minimum = strength/16
        self.recovery_limit = recovery_limit
        self.atom_count = atom_count
        self.history = history
        self.pending = None
        self.rollbacks = 0

    @property
    def stalled(self):
        return self.strength < self.minimum

    def start(self, iteration, xyz, details):
        record = {"iteration": iteration, **details, "recovery_iterations": None,
                  "rollback": False, "outcome": "pending"}
        self.history.append(record)
        self.pending = (record, xyz.copy())

    def observe(self, iteration, geometry, xyz):
        """Return True when coordinates were restored (skip the current move)."""
        if self.pending is None:
            return False
        record, snapshot = self.pending
        recovery_age = iteration-record["iteration"]-1
        big = (geometry["distance_violations"] > 0.0005*self.atom_count
               or geometry["global_components"] >= 50)
        if "damage" not in record:
            record["damage"] = damage_summary(iteration, geometry)
            record["next_state_classification"] = ("fine" if geometry["mandatory_pass"] else "big" if big else "small")
        if geometry["mandatory_pass"]:
            record["recovery_iterations"] = recovery_age
            record["recovered_at_iteration"] = iteration
            record["outcome"] = "feasible" if recovery_age == 0 else "recovered"
            if recovery_age == 0:
                self.strength = min(self.maximum, self.strength*1.25)
            record["strength_after"] = self.strength
            self.pending = None
            return False
        if big or recovery_age >= self.recovery_limit:
            xyz[:] = snapshot
            self.strength *= 0.5
            self.rollbacks += 1
            record.update({"rollback": True, "outcome": "rolled_back", "rollback_iteration": iteration,
                           "rollback_after_repair_iterations": recovery_age,
                           "rollback_reason": "big_damage" if big else "recovery_limit",
                           "rollback_geometry": damage_summary(iteration, geometry),
                           "strength_after": self.strength})
            self.pending = None
            return True
        return False

    def finish(self):
        if self.pending is not None:
            self.pending[0]["outcome"] = "unrecovered_at_stop"
            self.pending = None
