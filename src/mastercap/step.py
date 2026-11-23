"""Moment-matched spherical step targets; array-only analytic numerics."""
from dataclasses import dataclass, asdict
from math import comb
import numpy as np
from scipy.optimize import least_squares
from .model import MastercapError
from .radial import make_edges, shell_volumes


def validate_bounds(inner, outer, width):
    if (not np.isfinite([inner, outer, width]).all() or width < 0 or
            inner-width/2 < 0 or outer <= inner or outer-inner < width):
        raise MastercapError(f'Step bounds {inner}..{outer} A are invalid for edge width {width:g} A: '
                             'need inner - width/2 >= 0 and outer - inner >= width')


def shape(radii, inner, outer, width):
    validate_bounds(inner, outer, width)
    r = np.asarray(radii)
    if width == 0:
        return ((r >= inner) & (r <= outer)).astype(float)
    return np.clip(np.minimum((r-inner+width/2)/width, (outer+width/2-r)/width), 0, 1)


def integral(inner, outer, width, k=0, lower=None, upper=None):
    """Integral h(r) r**(2+k) dr, optionally clipped. No quadrature.

    Expand each linear piece about the clipped interval midpoint. Integrating
    its even polynomial powers avoids subtraction of large, nearly equal powers.
    """
    validate_bounds(inner, outer, width)
    if k not in (0, 1, 2, 3):
        raise MastercapError('Step integral moment order must be 0..3')
    lo = inner-width/2 if lower is None else np.asarray(lower)
    hi = outer+width/2 if upper is None else np.asarray(upper)
    pieces = [(inner, outer, 1., 0.)] if width == 0 else [
        (inner-width/2, inner+width/2, .5, 1/width),
        (inner+width/2, outer-width/2, 1., 0.),
        (outer-width/2, outer+width/2, .5, -1/width)]
    result = np.zeros(np.broadcast_shapes(np.shape(lo), np.shape(hi)))
    n = 2+k
    for left, right, height, slope in pieces:
        l, u = np.maximum(lo, left), np.minimum(hi, right)
        half = np.maximum(u-l, 0)/2
        mid = (l+u)/2
        hmid = height+slope*(mid-(left+right)/2)
        for j in range(n+1):
            coefficient = comb(n, j)*mid**(n-j)
            if j % 2 == 0:
                result += coefficient*hmid*2*half**(j+1)/(j+1)
            else:
                result += coefficient*slope*2*half**(j+2)/(j+2)
    return result.item() if result.ndim == 0 else result


@dataclass
class StepFit:
    inner: float
    outer: float
    edge_width: float
    thickness: float
    support_inner: float
    support_outer: float
    target_moments: dict
    model_moments: dict
    residuals: list
    iterations: int

    def report(self):
        return asdict(self)


def fit_step(radii, weights=None, edge_width=4.0):
    r = np.asarray(radii, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all() or np.any(r < 0):
        raise MastercapError('Step fit requires at least two finite nonnegative radii')
    weight = np.ones_like(r) if weights is None else np.asarray(weights, dtype=float)
    if weight.shape != r.shape or not np.isfinite(weight).all() or np.any(weight < 0) or weight.sum() <= 0:
        raise MastercapError('Step fit weights must be finite, nonnegative and have positive sum')
    w = float(edge_width)
    if not np.isfinite(w) or w < 0:
        raise MastercapError('Step edge width must be finite and nonnegative')
    m1, m2 = float(np.average(r, weights=weight)), float(np.average(r*r, weights=weight))
    variance = float(np.average((r-m1)**2, weights=weight))
    if m1 <= 0 or variance <= np.finfo(float).eps*max(m2, 1):
        raise MastercapError('Step fit cannot represent zero radial variance or zero mean')
    # Nonnegative support inner radius and plateau width enforce both constraints.
    half = max(np.sqrt(3*variance), w/2+1e-5)
    start = [max(m1-half-w/2, 1e-5), max(2*half-w, 1e-5)]
    def bounds(x):
        a = x[0]+w/2
        return a, a+w+x[1]
    def residual(x):
        a, b = bounds(x)
        i0 = integral(a, b, w)
        return [integral(a, b, w, 1)/i0/m1-1, integral(a, b, w, 2)/i0/m2-1]
    solution = least_squares(residual, start, bounds=(0, np.inf), xtol=1e-14,
                             ftol=1e-14, gtol=1e-14, max_nfev=2000, x_scale='jac')
    a, b = bounds(solution.x)
    errors = residual(solution.x)
    on_bound = np.any(solution.x <= 1e-7*max(m1, 1))
    if not solution.success or max(np.abs(errors)) >= 1e-10 or on_bound:
        raise MastercapError(f'Step fit cannot represent moments with edge width {w:g} A: '
                             f'relative residuals {errors}; solution on bound={bool(on_bound)}')
    i0 = integral(a, b, w)
    return StepFit(float(a), float(b), w, float(b-a), float(a-w/2), float(b+w/2),
                   {'m1': m1, 'm2': m2},
                   {'m1': float(integral(a, b, w, 1)/i0), 'm2': float(integral(a, b, w, 2)/i0)},
                   [float(v) for v in errors], solution.nfev)


def step_profiles(bounds, element_counts, bin_width):
    a, b, w = (bounds.inner, bounds.outer, bounds.edge_width) if isinstance(bounds, StepFit) else bounds
    validate_bounds(a, b, w)
    edges = make_edges(a-w/2, b+w/2, bin_width)
    volume = 4*np.pi*integral(a, b, w)
    densities = {e: float(n/volume) for e, n in element_counts.items()}
    if not all(np.isfinite(n) and n >= 0 for n in element_counts.values()):
        raise MastercapError('Step element counts must be finite and nonnegative')
    fraction = 4*np.pi*integral(a, b, w, lower=edges[:-1], upper=edges[1:])/shell_volumes(edges)
    profiles = {e: rho*fraction for e, rho in densities.items()}
    return edges, profiles, {**densities, 'Total': sum(densities.values())}
