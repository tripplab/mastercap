"""Volume-aware radial profiles and nearest-element reference statistics."""
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from .model import MastercapError, VDW, guard_override
from .radial import make_edges, shell_volumes


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
    distance_limit_overrides: list = field(default_factory=list)
    grouped_distance_factor: float | None = None
    diagnostics_cache: dict = field(default_factory=dict)
    inter_residue: dict = field(default_factory=dict)
    inter_statistics: dict = field(default_factory=dict)
    inter_scale: float = 1.0
    nominal_bin_width: float | None = None

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
                "distance_limit_overrides": self.distance_limit_overrides,
                "grouped_distance_factor": self.grouped_distance_factor,
                "inter_residue_limits": {"mode":"reference" if self.inter_residue else "off", "scale":self.inter_scale,
                                         "matrix":{"-".join(pair):value for pair,value in self.inter_residue.items()},
                                         "reference_statistics":self.inter_statistics},
                "neighbor_samples": self.neighbor_samples}


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
    return limits, report, grouped


def parse_limit_override(spec):
    try:
        pair, value = spec.split("=")
        names = pair.upper().split("-")
        if len(names) != 2 or any(name not in VDW for name in names):
            raise ValueError
        value = value.lower()
        if value not in ("steric", "pooled", "reference"):
            number = float(value)
            if not np.isfinite(number) or number <= 0:
                raise ValueError
            guard_override(tuple(sorted(names)), number)
            value = str(number)
        return "-".join(sorted(names))+"="+value
    except MastercapError:
        raise
    except (ValueError, AttributeError):
        raise MastercapError(f"Invalid pair-limit override {spec!r}; use ELEMENT-ELEMENT=positive_A|steric|pooled|reference") from None


def resolve_limit_overrides(overrides=None):
    resolved = {}
    for spec in overrides or []:
        spec = parse_limit_override(spec)
        pair, value = spec.split("=")
        if pair in resolved:
            raise MastercapError(f"Pair {pair} specified twice")
        resolved[pair] = value
    resolved.setdefault("S-S", "steric")
    return [f"{pair}={value}" for pair, value in sorted(resolved.items())]


def apply_limit_overrides(limits, grouped, overrides=None, neighbor_samples=None):
    result, records = limits.copy(), []
    specs = resolve_limit_overrides(overrides)
    # Explicit reference mode retains a measured pair, so it remains evidence.
    excluded = {tuple(spec.split('=')[0].split('-')) for spec in specs if spec.split('=')[1] != 'reference'}
    measured = neighbor_samples or {}
    ratios = {}
    for pair, limit in sorted(limits.items()):
        if pair in excluded or measured.get('-'.join(pair), {}).get('grouped_fallback', True):
            continue
        ratio = limit/sum(VDW[e] for e in pair)
        for element in pair:
            if element not in ratios or ratio < ratios[element][0]:
                ratios[element] = ratio, '-'.join(pair)
    for spec in specs:
        pair, value = spec.split("=")
        key = tuple(pair.split("-"))
        if key not in limits:
            continue
        mode = value if value in ("steric", "pooled", "reference") else "number"
        source = {}
        if mode == 'steric':
            missing = sorted({e for e in key if e not in ratios})
            if missing:
                raise MastercapError(f"No measured non-overridden pair for {', '.join(missing)} in {pair}; provide a numeric pair-limit override")
            candidates = [ratios[e] for e in key if e in ratios]
            ratio, source_pair = min(candidates)
            source = {'ratio': ratio, 'source_pair': source_pair}
        new = (ratio*sum(VDW[e] for e in key) if mode == 'steric' else
               grouped*sum(VDW[e] for e in key) if mode == "pooled" else
               limits[key] if mode == "reference" else float(value))
        guard_override(key, new)
        result[key] = float(new)
        records.append({"pair": pair, "original_value": limits[key], "new_value": float(new), "mode": mode, **source,
                        "reason": "statistical spacing, not a steric limit" if pair == "S-S" and mode == "steric"
                        else "user-selected distance limit"})
    return result, records


def analyze(structure, width=1.0, sigma=1.0, radius_method="extrema", percentiles=(0.5, 99.5),
            distance_percentile=1.0, min_samples=100, progress=None, pair_limit_overrides=None,
            saved_limit_overrides=None, compute_distances=True, step_profile=False, step_edge_width=4.0,
            inter_residue_limits="off", inter_residue_scale=1.0):
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
    if progress and compute_distances:
        progress("Estimating element-pair nearest-neighbor distance limits")
    limits, samples, grouped = distance_statistics(selected, distance_percentile, min_samples) if compute_distances else ({}, {}, None)
    if not compute_distances:
        override_records = []
    elif saved_limit_overrides is None:
        limits, override_records = apply_limit_overrides(limits, grouped, pair_limit_overrides, samples)
    else:
        # Saved numeric values are authoritative, including pre-floor reports.
        override_records = saved_limit_overrides
        seen = set()
        for entry in override_records:
            key = tuple(entry['pair'].split('-'))
            value = entry['new_value']
            if key not in limits or key in seen or not np.isfinite(value) or value <= 0:
                raise MastercapError('Invalid saved distance-limit override record')
            seen.add(key)
            limits[key] = float(value)
    if progress:
        for entry in override_records:
            progress(f"Distance limit override: {entry['pair']} {entry['original_value']:.3f} -> "
                     f"{entry['new_value']:.3f} A ({entry['mode']})")
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
    result=Analysis(selected, center, edges, original, smooth, limits, samples, warnings, excluded, sigma,
                    override_records, grouped, nominal_bin_width=width)
    if inter_residue_limits not in ('off','reference') or not np.isfinite(inter_residue_scale) or inter_residue_scale<=0:
        raise MastercapError('Inter-residue mode must be off/reference and scale finite positive')
    if inter_residue_limits=='reference':
        if progress:progress('Estimating non-bonded inter-residue nearest-neighbor limits')
        result.inter_residue,result.inter_statistics,_=inter_residue_statistics(selected,distance_percentile,min_samples)
        result.inter_scale=inter_residue_scale
    return result


def target_profiles(analysis, inner=None, outer=None):
    inner = analysis.inner if inner is None else inner
    outer = analysis.outer if outer is None else outer
    if not np.isfinite([inner, outer]).all() or not 0 <= inner < outer:
        raise MastercapError("Require finite 0 <= inner radius < outer radius")
    fraction = (analysis.edges - analysis.inner) / (analysis.outer - analysis.inner)
    edges = inner + fraction * (outer - inner)
    # Keep the piecewise constant local density; integrate in the NEW shell volumes.
    return edges, {e: p.copy() for e, p in analysis.smoothed.items()}


@dataclass
class Target:
    edges: np.ndarray
    profiles: dict
    reference_on_edges: dict | None
    step: dict | None = None
    geometry_scaling: dict | None = None
    splice: dict | None = None
    thickness_method: str | None = None


def resolve_geometry(analysis, *, step, edge_width=4.0, curvature_factor=None, thickness_factor=None,
                     splice_wall_keep=6.0, splice_window=2.0, splice_max_mismatch=0.10):
    """Use one step-fit-based absolute thickness delta for both target arms."""
    if curvature_factor is None and thickness_factor is None:return None,None,None
    from .step import fit_step, validate_bounds
    from .splice import plan_splice
    f=1. if curvature_factor is None else float(curvature_factor)
    g=1. if thickness_factor is None else float(thickness_factor)
    if not np.isfinite([f,g]).all() or not .25 <= f <= 4 or not .25 <= g <= 4:
        raise MastercapError(f'Geometry factors f={f:g}, g={g:g} outside the supported range [0.25, 4]')
    width=float(np.diff(analysis.edges)[0])
    requested=0.0
    try:
        fit=fit_step(np.linalg.norm(analysis.structure.xyz-analysis.center,axis=1),edge_width=edge_width)
        t_step=fit.outer-fit.inner
        requested=(g-1)*t_step
        n=int(round(abs(requested)/width))
        if g != 1 and n == 0:
            raise MastercapError(f'rounds to zero bins; minimum usable |G - 1| is about {width/(2*t_step):.8g}')
        signed_n=n if g > 1 else -n
        delta=signed_n*width
        a,b=(fit.inner,fit.outer) if step else (analysis.inner,analysis.outer)
        t0=b-a;outer=b/f;thickness=t0+delta;inner=outer-thickness
        nominal=getattr(analysis,'nominal_bin_width',None) or width
        if step:validate_bounds(inner,outer,edge_width)
        elif inner <= 0 or thickness < 2*nominal:
            raise MastercapError('Layered geometry needs inner > 0 and thickness >= 2 * nominal bin width')
        plan=None;paired=None
        if n:
            try:
                plan=plan_splice(analysis,fit,n,sign=1 if signed_n > 0 else -1,
                                 wall_keep=splice_wall_keep,window=splice_window,max_mismatch=splice_max_mismatch)
                paired={'feasible':True,'cut_bin':plan.cut_bin,'mismatch_total':plan.mismatch_total}
            except MastercapError as exc:
                if not step:raise
                paired={'feasible':False,'reason':str(exc)}
        if not step and plan is not None:
            # Store the actual uniform-grid endpoint, avoiding accumulated roundoff.
            inner=outer-width*(len(analysis.edges)-1+signed_n)
        block={'basis':'outer_radius','arm':'step' if step else 'layered','curvature_factor':f,'thickness_factor':g,
               'reference':{'inner':a,'outer':b,'thickness':t0,'mid_radius':(a+b)/2},
               'generated':{'inner':inner,'outer':outer,'thickness':outer-inner,'mid_radius':(inner+outer)/2},
               'outer_radius_ratio':b/outer,'mid_radius_ratio':(a+b)/(inner+outer),'thickness_ratio':(outer-inner)/t0,
               'thickness_rule':'shared_absolute_delta','thickness_basis':'step_fit_thickness',
               't_step_angstrom':t_step,'grid_width_angstrom':width,'delta_requested_angstrom':requested,
               'delta_angstrom':delta,'delta_bins':signed_n,'rounding':'nearest whole reference bin',
               'thickness_factor_requested':g,'thickness_factor_effective':1+delta/t_step,
               'arm_thickness_ratio':(outer-inner)/t0}
        if step:
            block.update(step_support=[inner-edge_width/2,outer+edge_width/2],thickness_method='plateau',
                         paired_layered_splice=paired or {'feasible':True,'cut_bin':None,'mismatch_total':None})
        else:
            block['thickness_method']='splice' if n else 'shift'
            if plan is not None:block['splice']=plan.report(analysis.edges,outer=outer)
        return inner,outer,block
    except MastercapError as exc:
        raise MastercapError(f'Geometry factors f={f:g}, G={g:g}, Delta_req={requested:.8g} A, w={width:.8g} A: {exc}') from exc


def target_on_edges(target, new_edges):
    """Integrate the unchanged continuous step or piecewise-constant target."""
    new_edges = np.asarray(new_edges, dtype=float)
    if (new_edges.ndim != 1 or len(new_edges) < 2 or not np.isfinite(new_edges).all()
            or np.any(np.diff(new_edges) <= 0) or new_edges[0] < target.edges[0]
            or new_edges[-1] > target.edges[-1]):
        raise MastercapError('New target edges must increase within the target support')
    volumes = shell_volumes(new_edges)
    if target.step:
        from .step import integral
        block = target.step
        a, b = (block['generated_bounds'][k] for k in ('inner', 'outer'))
        integrals = np.array([4*np.pi*integral(a, b, block['edge_width'], lower=lo, upper=hi)
                              for lo, hi in zip(new_edges[:-1], new_edges[1:])])
        return {e: block['density']['plateau_by_element'][e]*integrals/volumes
                for e in target.profiles}
    # Walk the two partitions in linear time; avoid an O(number_of_bins^2) array.
    counts = {e: np.zeros(len(volumes)) for e in target.profiles}
    i = j = 0
    while i < len(volumes) and j < len(target.edges)-1:
        lo, hi = max(new_edges[i], target.edges[j]), min(new_edges[i+1], target.edges[j+1])
        if hi > lo:
            volume = 4*np.pi/3*(hi**3-lo**3)
            for e, density in target.profiles.items():
                counts[e][i] += density[j]*volume
        if new_edges[i+1] <= target.edges[j+1]:
            i += 1
        else:
            j += 1
    return {e: value/volumes for e, value in counts.items()}


def held_out_target(target, fraction=0.5):
    from .radial import offset_edges
    edges = offset_edges(target.edges, fraction)
    return edges, target_on_edges(target, edges)


@dataclass
class Partition:
    fraction: float
    edges: np.ndarray
    profiles: dict


def dither_partitions(target, k):
    """Exact fixed offset targets, constructed once; fraction zero is untouched."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise MastercapError('Refine bin dither must be an integer >= 1')
    totals = {e: float(np.sum(p*shell_volumes(target.edges))) for e,p in target.profiles.items()}
    partitions = [Partition(0., target.edges, target.profiles)]
    for i in range(1, k):
        fraction = i/k
        edges, profiles = held_out_target(target, fraction)
        for e, expected in totals.items():
            actual = float(np.sum(profiles[e]*shell_volumes(edges)))
            if not np.isfinite(actual) or abs(actual-expected) > 1e-9*abs(expected):
                raise MastercapError(f'Dither partition {fraction:g} changes the {e} target total: {actual:g} vs {expected:g}')
        partitions.append(Partition(fraction, edges, profiles))
    return partitions


def build_target(analysis, inner=None, outer=None, *, step=False, edge_width=4.0, bin_width=None, splice=None):
    """Single target definition for fitting, generation, calibration and validation."""
    if not step and splice is not None:
        from .splice import apply_splice, _smooth
        outer=analysis.outer if outer is None else outer
        edges,original=apply_splice(analysis.edges,analysis.original,splice)
        edges=outer-float(np.diff(analysis.edges)[0])*np.arange(len(edges)-1,-1,-1)
        if inner is not None and not np.isclose(edges[0],inner,rtol=0,atol=1e-9):
            raise MastercapError('Spliced inner edge differs from resolved geometry')
        profiles={e:_smooth(p,edges,analysis.smoothing_sigma) for e,p in original.items()}
        return Target(edges,profiles,original,splice=splice.report(analysis.edges,outer=outer),thickness_method='splice')
    if not step:
        edges, profiles = target_profiles(analysis, inner, outer)
        stretched=abs((edges[-1]-edges[0])-(analysis.outer-analysis.inner)) > 1e-9
        return Target(edges, profiles, analysis.original,thickness_method='stretch' if stretched else None)
    from .step import fit_step, integral, step_profiles, validate_bounds
    from .model import MASS
    radii = np.linalg.norm(analysis.structure.xyz-analysis.center, axis=1)
    elements = analysis.structure.elements
    width = float(np.diff(analysis.edges)[0]) if bin_width is None else bin_width
    fit = fit_step(radii, edge_width=edge_width)
    a, b = fit.inner if inner is None else inner, fit.outer if outer is None else outer
    validate_bounds(a, b, edge_width)
    resized = a != fit.inner or b != fit.outer
    reference_counts = {e: int(np.count_nonzero(elements == e)) for e in analysis.original}
    ratio = integral(a, b, edge_width)/integral(fit.inner, fit.outer, edge_width)
    counts = {e: n*ratio for e, n in reference_counts.items()}
    edges, profiles, densities = step_profiles((a, b, edge_width), counts, width)
    reference = None if resized else histogram(analysis.structure, analysis.center, edges, analysis.original)
    def diagnostic(r, **kwargs):
        try:
            fitted = fit_step(r, **kwargs)
            return {k: getattr(fitted, k) for k in ('inner', 'outer', 'thickness')}
        except MastercapError as exc:
            return {'error': str(exc)}
    if np.any(radii == 0):
        depth_fit = {'error': 'Per-area weights 1/r^2 are undefined at r=0'}
    else:
        weights = 1/radii**2
        depth = float(np.average(radii, weights=weights))
        thickness = float(np.sqrt(12*np.average((radii-depth)**2, weights=weights)))
        depth_fit = {'inner': depth-thickness/2, 'outer': depth+thickness/2, 'thickness': thickness}
    mean = float(radii.mean())
    variance = float(np.mean((radii-mean)**2))
    i0 = integral(fit.inner, fit.outer, edge_width)
    moments = [integral(fit.inner, fit.outer, edge_width, k)/i0 for k in (1, 2, 3)]
    step_variance = moments[1]-moments[0]**2
    skew_ref = float(np.mean((radii-mean)**3)/variance**1.5)
    skew_step = float((moments[2]-3*moments[0]*moments[1]+2*moments[0]**3)/step_variance**1.5)
    geometric_mean = len(radii)/(4*np.pi/3*(analysis.outer**3-analysis.inner**3))
    # Rebin at the requested width, independent of any smoothing.
    reference_hist = histogram(analysis.structure, analysis.center,
                               make_edges(analysis.inner, analysis.outer, width), analysis.original)
    maximum = float(sum(reference_hist.values()).max())
    block = {'model': 'step', 'moments': 'atom', 'edge_width': edge_width, 'fit': fit.report(),
             'generated_bounds': {'inner': float(a), 'outer': float(b), 'support': [float(edges[0]), float(edges[-1])], 'resized': resized},
             'counts': {'reference': reference_counts, 'target': counts},
             'density': {'plateau_total': densities['Total'],
                         'plateau_by_element': {e: densities[e] for e in counts},
                         'reference_mean_geometric': geometric_mean, 'reference_max_bin': maximum,
                         'ratio_to_mean': densities['Total']/geometric_mean,
                         'ratio_to_max_bin': densities['Total']/maximum},
             'sensitivity': {'hard': diagnostic(radii, edge_width=0),
                             'mass': diagnostic(radii, weights=np.array([MASS[e] for e in elements]), edge_width=edge_width),
                             'depth': depth_fit},
             'per_element_bounds': {e: diagnostic(radii[elements == e], edge_width=edge_width) for e in counts},
             'skewness': {'reference': skew_ref, 'step': skew_step, 'difference': skew_step-skew_ref}}
    if not resized:
        block['reference_atoms_outside_support'] = int(np.count_nonzero((radii < edges[0]) | (radii > edges[-1])))
    return Target(edges, profiles, reference, block)


def inter_residue_statistics(structure, percentile=1., min_samples=100):
    """Nearest unlike-residue, non-peptide/non-disulfide observations."""
    from .diagnostics import bond_keys,nearest_nonbonded
    if not 0<=percentile<=100 or min_samples<1:
        raise MastercapError('Inter-residue percentile must be in [0,100], minimum samples >= 1')
    samples=nearest_nonbonded(structure,bond_keys(structure,disulfides=True))
    normalized=[v/sum(VDW[e] for e in pair) for pair,v in samples.items() if len(v)]
    if not normalized:raise MastercapError('No non-bonded inter-residue observations in reference')
    grouped=float(np.percentile(np.concatenate(normalized),percentile))
    limits={};report={}
    for pair,values in samples.items():
        fallback=len(values)<min_samples
        limits[pair]=float(grouped*sum(VDW[e] for e in pair) if fallback else np.percentile(values,percentile))
        report['-'.join(pair)]={'count':len(values),'grouped_fallback':fallback,
                               'percentiles':{str(p):float(np.percentile(values,p)) for p in (1,50,99)} if len(values) else {}}
    return limits,report,grouped
