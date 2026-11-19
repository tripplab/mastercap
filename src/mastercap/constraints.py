"""Exact local checks using spatial neighbors, without an N x N matrix."""
from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .analysis import histogram, shell_volumes


def components(n, edges):
    if n == 0:
        return 0, np.array([], dtype=int)
    if len(edges) == 0:
        return n, np.arange(n)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    return connected_components(graph, directed=False)


def local_geometry(structure, limits, margin=0.0, mode="independent"):
    xyz, radii, residues = structure.xyz, structure.radii, structure.residues
    maximum = 2 * radii.max()
    pairs = cKDTree(xyz).query_pairs(maximum, output_type="ndarray")
    # A rare element can have a large inferred separation. Do not use that
    # radius for ALL atoms: query only the affected element pairs.
    elements = structure.elements
    extra = []
    for (a, b), bound in limits.items():
        if bound <= maximum:
            continue
        ai, bi = np.flatnonzero(elements == a), np.flatnonzero(elements == b)
        if not len(ai) or not len(bi):
            continue
        near = cKDTree(xyz[ai]).sparse_distance_matrix(cKDTree(xyz[bi]), bound, output_type="coo_matrix")
        candidate = np.column_stack([ai[near.row], bi[near.col]])
        candidate = candidate[candidate[:, 0] != candidate[:, 1]]
        extra.append(np.sort(candidate, axis=1))
    if extra:
        pairs = np.unique(np.vstack([pairs, *extra]), axis=0)
    if not len(pairs):
        pairs = np.empty((0, 2), dtype=int)
    distances = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    codes = {e: i for i, e in enumerate(sorted(set(elements)))}
    matrix = np.zeros((len(codes), len(codes)))
    for (a, b), lower in limits.items():
        if a in codes and b in codes:
            matrix[codes[a], codes[b]] = matrix[codes[b], codes[a]] = lower
    encoded = np.array([codes[e] for e in elements])
    lower = matrix[encoded[pairs[:, 0]], encoded[pairs[:, 1]]]
    internal = residues[pairs[:, 0]] == residues[pairs[:, 1]]
    violation = distances < lower - 1e-9
    exemptions = violation & internal if mode == "residue" else np.zeros(len(pairs), dtype=bool)
    violation &= ~exemptions
    overlap = distances < radii[pairs[:, 0]] + radii[pairs[:, 1]] - margin
    edges = pairs[overlap]
    degree = np.bincount(edges.ravel(), minlength=len(xyz))
    count, labels = components(len(xyz), edges)
    internal_edges = edges[residues[edges[:, 0]] == residues[edges[:, 1]]]
    _, internal_labels = components(len(xyz), internal_edges)
    unique = np.unique(residues)
    # Count distinct (residue, component) pairs, including one-atom residues.
    compound = np.unique(np.column_stack([residues, internal_labels]), axis=0)
    per_residue = Counter(compound[:, 0])
    disconnected = [int(r) for r in unique if per_residue[r] > 1]
    return {"pairs": pairs, "distances": distances, "lower": lower,
            "violations": violation, "overlap_edges": edges, "labels": labels,
            "internal_labels": internal_labels, "components": int(count), "isolated_atoms": int(np.sum(degree == 0)),
            "disconnected_residues": disconnected, "internal_exemptions": int(exemptions.sum())}


def validate_geometry(structure, center, edges, limits, margin=0.0, mode="independent", center_tolerance=1.0):
    radial = np.linalg.norm(structure.xyz - center, axis=1)
    local = local_geometry(structure, limits, margin, mode)
    outside = int(np.sum((radial < edges[0] - 1e-9) | (radial > edges[-1] + 1e-9)))
    center_error = float(np.linalg.norm(structure.xyz.mean(axis=0) - center))
    report = {"atom_count": len(structure.atoms), "outside_radial_bounds": outside,
              "observed_inner_radius": float(radial.min()), "observed_outer_radius": float(radial.max()),
              "center_error_angstrom": center_error, "center_tolerance_angstrom": center_tolerance,
              "distance_violations": int(local["violations"].sum()),
              "preserved_internal_distance_exceptions": local["internal_exemptions"],
              "global_components": local["components"],
              "isolated_atoms": local["isolated_atoms"],
              "disconnected_residues": local["disconnected_residues"]}
    report["mandatory_pass"] = bool(outside == 0 and center_error <= center_tolerance and
                                     report["distance_violations"] == 0 and local["components"] == 1 and local["isolated_atoms"] == 0 and
                                     not local["disconnected_residues"])
    return report


def profile_metrics(structure, center, edges, target, original=None):
    generated = histogram(structure, center, edges, target)
    volumes = shell_volumes(edges)
    reports = {}
    total_error = total_target = 0.0
    for e, goal in target.items():
        expected = float(np.sum(goal * volumes))
        actual = int(np.sum(structure.elements == e))
        error = float(np.sum(np.abs(generated[e] - goal) * volumes))
        reports[e] = {"target_count": expected, "generated_count": actual,
                      "count_difference": actual - expected,
                      "relative_count_error": abs(actual - expected) / expected if expected else None,
                      "profile_error": error / expected if expected else None}
        if original is not None:
            denom = np.sum(original[e] * volumes)
            reports[e]["original_profile_error"] = float(np.sum(np.abs(generated[e] - original[e]) * volumes) / denom) if denom else None
        total_error += error
        total_target += expected
    return {"elements": reports, "total_profile_error": total_error / total_target if total_target else None,
            "generated_profiles": {e: p.tolist() for e, p in generated.items()}}


def acceptance(geometry, metrics, profile_tolerance, composition_tolerance):
    if not geometry["mandatory_pass"]:
        return "failed"
    if profile_tolerance is None or composition_tolerance is None:
        return "uncalibrated"
    soft_pass = all((m["profile_error"] or 0) <= profile_tolerance and
                    abs(m["count_difference"]) <= max(1.0, composition_tolerance * m["target_count"])
                    for m in metrics["elements"].values())
    return "valid" if soft_pass else "approximate"
