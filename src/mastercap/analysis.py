"""Volume-aware radial profiles and nearest-element reference statistics."""
from dataclasses import dataclass
from itertools import combinations_with_replacement
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from .model import MastercapError, VDW


@dataclass
class Analysis:
    structure: object
    center: np.ndarray
    edges: np.ndarray
    original: dict
    smoothed: dict
    limits: dict
    neighbor_samples: dict
    warnings: list
    excluded: int
    smoothing_sigma: float = 0.0

    @property
    def inner(self):
        return float(self.edges[0])

    @property
    def outer(self):
        return float(self.edges[-1])

    def summary(self):
        volumes = shell_volumes(self.edges)
        return {"reference": self.structure.provenance, "center": self.center.tolist(),
                "inner_radius": self.inner, "outer_radius": self.outer,
                "atom_count": len(self.structure.atoms),
                "residue_count": len(np.unique(self.structure.residues)),
                "element_counts": {e: int(np.sum(self.structure.elements == e)) for e in self.original},
                "excluded_radial_atoms": self.excluded,
                "warnings": self.warnings,
                "smoothing": {"enabled": self.smoothing_sigma > 0,
                              "sigma_angstrom": self.smoothing_sigma},
                "profiles": {"edges": self.edges.tolist(), "volumes": volumes.tolist(),
                             "original": {e: p.tolist() for e, p in self.original.items()},
                             "smoothed": {e: p.tolist() for e, p in self.smoothed.items()}},
                "distance_limits": {"-".join(k): v for k, v in self.limits.items()},
                "neighbor_samples": self.neighbor_samples}


def shell_volumes(edges):
    return 4 * np.pi / 3 * np.diff(np.asarray(edges) ** 3)


def make_edges(inner, outer, width):
    if not np.isfinite([inner, outer, width]).all() or not 0 <= inner < outer or width <= 0:
        raise MastercapError("Require finite 0 <= inner radius < outer radius and bin width > 0")
    n = int(np.ceil((outer - inner) / width))
    if n > 1_000_000:
        raise MastercapError("Requested radial resolution exceeds one million bins")
    return np.linspace(inner, outer, n + 1)


def histogram(structure, center, edges, elements, *, atom_elements=None):
    radial = np.linalg.norm(structure.xyz - center, axis=1)
    volume = shell_volumes(edges)
    atom_elements = structure.elements if atom_elements is None else atom_elements
    return {e: np.histogram(radial[atom_elements == e], bins=edges)[0] / volume for e in elements}


def distance_statistics(structure, percentile=1, min_samples=100):
    if not 0 <= percentile <= 100 or min_samples < 1:
        raise MastercapError("Distance percentile must be in [0, 100], minimum samples >= 1")
    elements = sorted(set(structure.elements))
    coords = {e: structure.xyz[structure.elements == e] for e in elements}
    trees = {e: cKDTree(coords[e]) for e in elements}
    samples, normalized = {}, []
    for a, b in combinations_with_replacement(elements, 2):
        if a == b:
            if len(coords[a]) < 2:
                values = np.array([])
            else:
                values = trees[a].query(coords[a], k=2)[0][:, 1]
        else:
            values = np.concatenate([trees[b].query(coords[a])[0], trees[a].query(coords[b])[0]])
        values = values[np.isfinite(values)]
        if np.any(values <= 0):
            raise MastercapError(f"Coincident reference atoms for pair {a}-{b}")
        samples[(a, b)] = values
        if len(values):
            normalized.append(values / (VDW[a] + VDW[b]))
    if not normalized:
        raise MastercapError("At least two noncoincident reference atoms are required")
    grouped = float(np.percentile(np.concatenate(normalized), percentile))
    limits, report = {}, {}
    for pair, values in samples.items():
        fallback = len(values) < min_samples
        limits[pair] = float(grouped * sum(VDW[e] for e in pair) if fallback else np.percentile(values, percentile))
        report["-".join(pair)] = {"count": len(values), "grouped_fallback": fallback,
                                  "percentiles": {str(p): float(np.percentile(values, p)) for p in (1, 50, 99)} if len(values) else {}}
    return limits, report


def analyze(structure, width=1.0, sigma=1.0, radius_method="extrema", percentiles=(0.5, 99.5),
            distance_percentile=1.0, min_samples=100, progress=None):
    if not np.isfinite([width, sigma]).all() or width <= 0 or sigma < 0:
        raise MastercapError("Bin width must be positive and smoothing sigma nonnegative")
    if progress:
        progress("Analyzing reference center, radial bounds and selected atoms")
    center = structure.xyz.mean(axis=0)
    radial = np.linalg.norm(structure.xyz - center, axis=1)
    if radius_method == "percentile":
        if not 0 <= percentiles[0] < percentiles[1] <= 100:
            raise MastercapError("Require 0 <= lower percentile < upper percentile <= 100")
        inner, outer = np.percentile(radial, percentiles)
    else:
        inner, outer = radial.min(), radial.max()
    edges = make_edges(float(inner), float(outer), width)
    selected = structure.subset((radial >= inner) & (radial <= outer)).relabel()
    if len(selected.atoms) < 2:
        raise MastercapError("Fewer than two atoms remain in the selected radial region")
    if progress:
        progress(f"Computing radial profiles: bin width <= {width:g} A; "
                 +(f"Gaussian sigma {sigma:g} A" if sigma else "smoothing disabled"))
    original = histogram(selected, center, edges, sorted(set(selected.elements)))
    volume = shell_volumes(edges)
    smooth = {}
    for e, profile in original.items():
        filtered = gaussian_filter1d(profile, sigma / np.diff(edges)[0], mode="reflect") if sigma else profile.copy()
        smooth[e] = filtered * (np.sum(profile * volume) / np.sum(filtered * volume))
    if progress:
        progress("Estimating element-pair nearest-neighbor distance limits")
    limits, samples = distance_statistics(selected, distance_percentile, min_samples)
    warnings = selected.warnings.copy()
    warnings.append("Input is assumed to contain an already expanded complete capsid; assembly was not reconstructed")
    directions = selected.xyz - center
    norms = np.linalg.norm(directions, axis=1)
    directions = directions[norms > 0] / norms[norms > 0, None]
    occupied = len(np.unique((directions > 0).astype(int), axis=0))
    dipole = float(np.linalg.norm(directions.mean(axis=0))) if len(directions) else 1.0
    if occupied < 8 or dipole > 0.15:
        warnings.append(f"Suspicious angular coverage: {occupied}/8 octants, directional imbalance {dipole:.3f}")
    excluded = len(structure.atoms) - len(selected.atoms)
    if excluded:
        warnings.append(f"Excluded {excluded} atoms outside selected radial interval; residues may be partial")
    return Analysis(selected, center, edges, original, smooth, limits, samples, warnings, excluded, sigma)


def target_profiles(analysis, inner=None, outer=None):
    inner = analysis.inner if inner is None else inner
    outer = analysis.outer if outer is None else outer
    if not np.isfinite([inner, outer]).all() or not 0 <= inner < outer:
        raise MastercapError("Require finite 0 <= inner radius < outer radius")
    fraction = (analysis.edges - analysis.inner) / (analysis.outer - analysis.inner)
    edges = inner + fraction * (outer - inner)
    # Keep the piecewise constant local density; integrate in the NEW shell volumes.
    return edges, {e: p.copy() for e, p in analysis.smoothed.items()}
