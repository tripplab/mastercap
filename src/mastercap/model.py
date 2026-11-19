"""Internal identities never depend on PDB serial numbers."""
from dataclasses import dataclass, field, replace
import numpy as np


class MastercapError(ValueError):
    """An actionable input, geometry, or search failure."""


# Bondi, J. Phys. Chem. 68 (1964), 441-451, doi:10.1021/j100785a001.
# Deliberately finite coverage: do not invent radii for unsupported elements.
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
       "P": 1.80, "S": 1.80, "CL": 1.75, "SE": 1.90, "BR": 1.85,
       "I": 1.98}
PROTEIN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL SEC PYL ASX GLX HID HIE HIP CYX MSE HYP".split())


@dataclass(frozen=True)
class Atom:
    name: str
    element: str
    resname: str
    residue: int
    origin: dict = field(default_factory=dict, compare=False)


@dataclass
class Structure:
    atoms: list[Atom]
    xyz: np.ndarray
    warnings: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    @property
    def elements(self):
        return np.asarray([a.element for a in self.atoms])

    @property
    def residues(self):
        return np.asarray([a.residue for a in self.atoms], dtype=int)

    @property
    def radii(self):
        return np.asarray([VDW[a.element] for a in self.atoms])

    def subset(self, mask):
        idx = np.flatnonzero(mask)
        return Structure([self.atoms[i] for i in idx], self.xyz[idx].copy(),
                         self.warnings.copy(), self.provenance.copy())

    def copy(self):
        return Structure(self.atoms.copy(), self.xyz.copy(), self.warnings.copy(),
                         self.provenance.copy())

    def relabel(self):
        mapping = {}
        self.atoms = [replace(a, residue=mapping.setdefault(a.residue, len(mapping)))
                      for a in self.atoms]
        return self
