import gzip
import numpy as np
import pytest
from mastercap.model import Atom, Structure, MastercapError
from mastercap.pdbio import read_pdb, pdb_text, download


def test_repeated_serials_and_residues_preserved(tmp_path, small_structure):
    text, mapping = pdb_text(small_structure, segment_atom_limit=2, segment_residue_limit=1)
    path = tmp_path/"split.pdb"
    path.write_text(text)
    result = read_pdb(path, generated=True)
    assert len(result.atoms) == 4
    assert list(result.residues) == [0, 0, 1, 1]
    assert [m["serial"] for m in mapping] == [1, 2, 1, 2]
    assert not any(line.startswith("MODEL ") for line in text.splitlines())
    assert np.allclose(result.xyz, small_structure.xyz)


def test_occupancy_is_not_used(small_pdb):
    lines = small_pdb.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            lines[i] = line[:54]+"BADVAL"+line[60:]
    small_pdb.write_text("\n".join(lines))
    assert len(read_pdb(small_pdb).atoms) == 4


def test_altloc_rejected(small_pdb):
    text = small_pdb.read_text()
    line = next(x for x in text.splitlines() if x.startswith("ATOM"))
    small_pdb.write_text(text.replace(line, line[:16]+"A"+line[17:], 1))
    with pytest.raises(MastercapError, match="Alternate"):
        read_pdb(small_pdb)


def test_alpha_carbon_inference(small_pdb):
    lines = [line[:76]+"    " if line.startswith("ATOM") else line for line in small_pdb.read_text().splitlines()]
    small_pdb.write_text("\n".join(lines))
    result = read_pdb(small_pdb)
    assert set(result.elements) == {"C"}
    assert len(result.warnings) == 4


def test_hetatm_and_nucleic_atoms_excluded(small_pdb):
    text = small_pdb.read_text()
    line = next(x for x in text.splitlines() if x.startswith("ATOM"))
    small_pdb.write_text(text+"HETATM"+line[6:]+"\n"+line[:17]+" DA"+line[20:]+"\n")
    result = read_pdb(small_pdb)
    assert len(result.atoms) == 4
    assert any("HETATM" in w for w in result.warnings)
    assert any("nonprotein" in w for w in result.warnings)


def test_gzip_and_models(tmp_path, small_pdb):
    text = small_pdb.read_text()
    second = text.replace("  -2.400", "  -3.400")
    path = tmp_path/"reference.vdb.gz"
    path.write_bytes(gzip.compress(("MODEL        1\n"+text+"ENDMDL\nMODEL        2\n"+second+"ENDMDL\n").encode()))
    assert read_pdb(path, model=2).xyz[0, 0] == -3.4
    with pytest.raises(MastercapError, match="does not exist"):
        read_pdb(path, model=3)


def test_coordinate_field_overflow(small_structure):
    small_structure.xyz[0, 0] = 10000
    with pytest.raises(MastercapError, match="8.3"):
        pdb_text(small_structure)


def test_download_validates_identifier(tmp_path):
    with pytest.raises(MastercapError, match="four"):
        download("../../evil", tmp_path)


def test_roundtrip_over_99999_atoms(tmp_path):
    count = 100_003
    structure = Structure([Atom("CA", "C", "GLY", i//10) for i in range(count)], np.zeros((count, 3)))
    text, mapping = pdb_text(structure)
    path = tmp_path/"large.pdb"
    path.write_text(text)
    result = read_pdb(path, generated=True)
    assert len(result.atoms) == count
    assert len(np.unique(result.residues)) == len(np.unique(structure.residues))
    assert max(m["serial"] for m in mapping) <= 99998
    assert len({(m["segment"], m["serial"]) for m in mapping}) == count
