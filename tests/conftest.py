import numpy as np
import pytest
from mastercap.model import Atom, Structure
from mastercap.pdbio import pdb_text


@pytest.fixture
def small_structure():
    return Structure([Atom("CA", "C", "GLY", i//2) for i in range(4)],
                     np.array([[-2.4, 0, 0], [-0.8, 0, 0], [0.8, 0, 0], [2.4, 0, 0]]))


@pytest.fixture
def small_pdb(tmp_path, small_structure):
    path = tmp_path / "reference.pdb"
    path.write_text(pdb_text(small_structure)[0])
    return path
