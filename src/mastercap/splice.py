"""Reference-derived, wall-preserving whole-bin core splices."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .model import MastercapError
from .radial import shell_volumes


def _smooth(profile, edges, sigma):
    filtered = gaussian_filter1d(profile, sigma / np.diff(edges)[0], mode='reflect') if sigma else profile.copy()
    volume = shell_volumes(edges)
    denominator = np.sum(filtered * volume)
    return filtered * (np.sum(profile * volume) / denominator) if denominator else filtered


def scoring_profile(analysis, score_sigma=1.0):
    return _smooth(sum(analysis.original.values()), analysis.edges, score_sigma)


@dataclass(frozen=True)
class SplicePlan:
    mode: str
    cut_bin: int
    n_bins: int
    window_bins: int
    mismatch_total: float
    mismatch_by_element: dict
    candidates: int
    candidate_median: float
    allowed_bins: tuple
    threshold: float
    candidate_mismatch: tuple = ()
    window_angstrom: float = 2.0
    wall_keep_angstrom: float = 6.0
    score_sigma_angstrom: float = 1.0
    reference_total: tuple = ()

    def report(self, edges, *, outer=None):
        edges = np.asarray(edges)
        outer = float(edges[-1]) if outer is None else outer
        total = np.asarray(self.reference_total)
        new_edges, profiles = apply_splice(edges, {'Total': total}, self)
        new_edges = outer - float(np.diff(edges)[0])*np.arange(len(new_edges)-1,-1,-1)
        i, n = self.cut_bin, self.n_bins
        seam = i if self.mode == 'remove' else i+n
        lo, hi = self.allowed_bins
        return {'mode': self.mode, 'cut_bin': i, 'n_bins': n, 'window_bins': self.window_bins,
                'window_angstrom': self.window_angstrom, 'score_sigma_angstrom': self.score_sigma_angstrom,
                'wall_keep_angstrom': self.wall_keep_angstrom, 'allowed_bins': [lo, hi],
                'allowed_reference_radii': [float(edges[lo]), float(edges[hi])],
                'band_reference_radii': [float(edges[i]), float(edges[i+n])],
                'seam_generated_radius': float(new_edges[seam]),
                'seam_definition': 'edge at removed band start' if self.mode == 'remove' else 'edge between original band and its copy',
                'mismatch_total': self.mismatch_total, 'mismatch_by_element': self.mismatch_by_element,
                'candidates': self.candidates, 'candidate_median_mismatch': self.candidate_median,
                'threshold': self.threshold, 'candidate_mismatch': [list(row) for row in self.candidate_mismatch],
                'reference_bins': len(edges)-1, 'generated_bins': len(new_edges)-1,
                'expected_atoms_reference': float(np.sum(total*shell_volumes(edges))),
                'expected_atoms_generated': float(np.sum(profiles['Total']*shell_volumes(new_edges))),
                'note': 'S is too sparse to splice meaningfully; its mismatch is informational'}

    @classmethod
    def from_report(cls, report, analysis):
        """Recover a just-resolved plan without repeating scoring/selection."""
        return cls(report['mode'], report['cut_bin'], report['n_bins'], report['window_bins'],
                   report['mismatch_total'], report['mismatch_by_element'], report['candidates'],
                   report['candidate_median_mismatch'], tuple(report['allowed_bins']), report['threshold'],
                   tuple(tuple(row) for row in report['candidate_mismatch']), report['window_angstrom'],
                   report['wall_keep_angstrom'], report['score_sigma_angstrom'],
                   tuple(sum(analysis.original.values())))


def apply_splice(edges, profiles, plan):
    """Keep densities unchanged; anchor whole-bin edges at the supplied outer radius."""
    i, n = plan.cut_bin, plan.n_bins
    size = len(edges)-1
    if plan.mode not in ('remove', 'duplicate') or n < 1 or not 0 <= i < i+n <= size:
        raise MastercapError('Invalid splice band or mode')
    def splice(p):
        return np.concatenate((p[:i], p[i+n:])) if plan.mode == 'remove' else np.concatenate((p[:i+n], p[i:i+n], p[i+n:]))
    result = {e: splice(p) for e,p in profiles.items()}
    size += -n if plan.mode == 'remove' else n
    return edges[-1] - float(np.diff(edges)[0])*np.arange(size, -1, -1), result


def plan_splice(analysis, fit, n_bins, *, sign, wall_keep=6.0, window=2.0,
                max_mismatch=0.10, score_sigma=1.0):
    edges = analysis.edges
    w = float(np.diff(edges)[0])
    if (not isinstance(n_bins, (int, np.integer)) or n_bins < 1 or sign not in (-1, 1)
            or not np.isfinite([wall_keep, window, max_mismatch, score_sigma]).all()
            or wall_keep <= 0 or window <= 0 or not 0 < max_mismatch <= 1 or score_sigma < 0):
        raise MastercapError('Invalid splice length, sign, window, wall keep or mismatch threshold')
    k = max(1, int(round(window/w)))
    eligible = np.flatnonzero((edges[:-1] >= fit.inner+wall_keep) & (edges[1:] <= fit.outer-wall_keep))
    lo, hi = (int(eligible[0]), int(eligible[-1])+1) if len(eligible) else (int(np.searchsorted(edges, fit.inner+wall_keep)), int(np.searchsorted(edges, fit.outer-wall_keep, side='right'))-1)
    largest = max(0, hi-lo-2*k)
    mode = 'remove' if sign < 0 else 'duplicate'
    indices = range(lo+k, hi-n_bins-k+1)
    if not len(indices):
        raise MastercapError(f'No splice candidates: n={n_bins}, K={k}, lo={lo}, hi={hi}, allowed radii '
                             f'{fit.inner+wall_keep:g}..{fit.outer-wall_keep:g} A; largest feasible '
                             f'{"duplicable " if sign > 0 else ""}n={largest}')
    total = scoring_profile(analysis, score_sigma)
    def mismatch(profile, i):
        x, y = profile[i-k:i+k], profile[i+n_bins-k:i+n_bins+k]
        mean = np.mean(np.concatenate((x,y)))
        return float(np.sqrt(np.mean((x-y)**2))/mean) if mean > 0 else float('inf')
    scores = [(i, mismatch(total, i)) for i in indices]
    best = min(m for _,m in scores)
    if not np.isfinite(best):
        raise MastercapError('Splice scoring windows have no density (mean <= 0)')
    i = min((i for i,m in scores if abs(m-best) <= 1e-12), key=lambda i: (abs(i+n_bins/2-(lo+hi)/2), i))
    chosen = dict(scores)[i]
    median = float(np.median([m for _,m in scores]))
    if best > max_mismatch:
        raise MastercapError(f'No acceptable splice: best i={i}, mismatch={chosen:.6g}, threshold={max_mismatch:g}, '
                             f'candidates={len(scores)}, median={median:.6g}; reduce |G - 1| or raise '
                             '--splice-max-mismatch knowingly')
    # Empty alternative windows remain auditable without non-finite JSON values.
    recorded_scores = tuple((j, m if np.isfinite(m) else None) for j,m in scores)
    per_element = {e: (lambda m: m if np.isfinite(m) else None)(mismatch(_smooth(p, edges, score_sigma), i))
                   for e,p in analysis.original.items()}
    return SplicePlan(mode, i, n_bins, k, chosen, per_element, len(scores),
                      median if np.isfinite(median) else None, (lo, hi), max_mismatch, recorded_scores,
                      window, wall_keep, score_sigma, tuple(sum(analysis.original.values())))
