"""Array-only spherical shell volumes and the shared radial bin convention."""
import numpy as np
from .model import MastercapError


def shell_volumes(edges):
    return 4 * np.pi / 3 * np.diff(np.asarray(edges) ** 3)


def make_edges(inner, outer, width):
    if not np.isfinite([inner, outer, width]).all() or not 0 <= inner < outer or width <= 0:
        raise MastercapError("Require finite 0 <= inner radius < outer radius and bin width > 0")
    n = int(np.ceil((outer - inner) / width))
    if n > 1_000_000:
        raise MastercapError("Requested radial resolution exceeds one million bins")
    return np.linspace(inner, outer, n + 1)

