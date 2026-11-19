"""Exact local checks using spatial neighbors, without an N x N matrix."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .analysis import histogram, shell_volumes


@dataclass
class GeometryMetadata:
    """Per-inventory constants, shared by rounded validation and repair.

    Rebuild if atom identity, residue membership, or limits change. Coordinates
    are intentionally not cached: every local geometry uses current positions.
    """
    elements: np.ndarray
    radii: np.ndarray
    residues: np.ndarray
    codes: np.ndarray
    lower_matrix: np.ndarray
    element_indices: dict
    groups: dict

    @classmethod
    def build(cls, structure, limits):
        elements, radii, residues = structure.elements, structure.radii, structure.residues
        names, codes = np.unique(elements, return_inverse=True)
        lookup = {str(e): i for i, e in enumerate(names)}
        matrix = np.zeros((len(names), len(names)))
        for (a, b), lower in limits.items():
            if a in lookup and b in lookup:
                matrix[lookup[a], lookup[b]] = matrix[lookup[b], lookup[a]] = lower
        order = np.argsort(residues, kind="stable")
        blocks = np.split(order, np.flatnonzero(np.diff(residues[order]))+1)
        groups = {int(residues[g[0]]): g for g in blocks if len(g)}
        return cls(elements, radii, residues, codes, matrix,
                   {str(e): np.flatnonzero(elements == e) for e in names}, groups)


def components(n, edges):
    if n == 0:
        return 0, np.array([], dtype=int)
    if len(edges) == 0:
        return n, np.arange(n)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    return connected_components(graph, directed=False)


def local_geometry(structure, limits, margin=0.0, mode="independent", *, metadata=None):
    metadata = GeometryMetadata.build(structure, limits) if metadata is None else metadata
    xyz, radii, residues = structure.xyz, metadata.radii, metadata.residues
    maximum = 2 * radii.max()
    pairs = cKDTree(xyz).query_pairs(maximum, output_type="ndarray")
    # A rare element can have a large inferred separation. Do not use that
    # radius for ALL atoms: query only the affected element pairs.
    extra = []
    for (a, b), bound in limits.items():
        if bound <= maximum:
            continue
        ai = metadata.element_indices.get(a, np.array([], dtype=int))
        bi = metadata.element_indices.get(b, np.array([], dtype=int))
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
    lower = metadata.lower_matrix[metadata.codes[pairs[:, 0]], metadata.codes[pairs[:, 1]]]
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
    # Count distinct (residue, component) pairs, including one-atom residues.
    compound = np.unique(np.column_stack([residues, internal_labels]), axis=0)
    unique, counts = np.unique(compound[:, 0], return_counts=True)
    disconnected = unique[counts > 1].astype(int).tolist()
    return {"pairs": pairs, "distances": distances, "lower": lower,
            "violations": violation, "overlap_edges": edges, "labels": labels,
            "internal_labels": internal_labels, "components": int(count), "isolated_atoms": int(np.sum(degree == 0)),
            "disconnected_residues": disconnected, "internal_exemptions": int(exemptions.sum())}


def validate_geometry(structure, center, edges, limits, margin=0.0, mode="independent", center_tolerance=1.0,
                      *, local=None, metadata=None):
    radial = np.linalg.norm(structure.xyz - center, axis=1)
    # A caller may reuse the local graph only for these exact coordinates and
    # settings. Export/revalidation callers omit it and rebuild independently.
    if local is None:
        local = local_geometry(structure, limits, margin, mode, metadata=metadata)
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


def profile_metrics(structure, center, edges, target, original=None, *, metadata=None):
    elements = structure.elements if metadata is None else metadata.elements
    generated = histogram(structure, center, edges, target, atom_elements=elements)
    volumes = shell_volumes(edges)
    reports = {}
    total_error = total_target = 0.0
    for e, goal in target.items():
        expected = float(np.sum(goal * volumes))
        actual = int(np.sum(elements == e))
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
