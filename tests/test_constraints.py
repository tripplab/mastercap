import numpy as np
import pytest
from unittest.mock import patch
from mastercap.constraints import (validate_geometry, acceptance, profile_metrics,
                                   connectivity_audit, require_connectivity_audit)
from mastercap.model import Atom, Structure, MastercapError


def particle(x, residues):
    return Structure([Atom("CA", "C", "GLY", r) for r in residues],
                     np.column_stack([x, np.zeros((len(x), 2))]))


def test_neighbor_per_atom_is_not_global_connectivity():
    s = particle([-6, -4, 4, 6], [0, 0, 1, 1])
    result = validate_geometry(s, np.zeros(3), np.array([0, 10]), {("C", "C"): 1})
    assert result["global_components"] == 2
    assert not result["mandatory_pass"]


def test_internal_connectivity_is_independent_of_global():
    s = particle([0, 2, 4], [0, 1, 0])
    result = validate_geometry(s, np.array([2, 0, 0]), np.array([0, 5]), {("C", "C"): 1})
    assert result["global_components"] == 1
    assert result["disconnected_residues"] == [0]


def test_tangency_does_not_connect():
    s = particle([0, 3.4], [0, 1])
    result = validate_geometry(s, np.array([1.7, 0, 0]), np.array([0, 5]), {("C", "C"): 1})
    assert result["global_components"] == 2


def test_pdb_rounding_can_break_overlap():
    s = particle([0, 3.3996], [0, 1])
    assert validate_geometry(s, np.array([1.7, 0, 0]), [0, 5], {("C", "C"): 1})["mandatory_pass"]
    s.xyz = np.round(s.xyz, 3)
    assert not validate_geometry(s, np.array([1.7, 0, 0]), [0, 5], {("C", "C"): 1})["mandatory_pass"]


def test_rigid_internal_exception_and_center_tolerance():
    s = particle([0, 1.4], [0, 0])
    report = validate_geometry(s, np.zeros(3), [0, 2], {("C", "C"): 1.5}, mode="residue")
    assert report["mandatory_pass"]
    assert report["preserved_internal_distance_exceptions"] == 1
    report = validate_geometry(s, np.array([3, 0, 0]), [0, 5], {("C", "C"): 1.5}, mode="residue")
    assert not report["mandatory_pass"]


def test_no_soft_tolerance_can_override_mandatory_failure():
    assert acceptance({"mandatory_pass": False}, {}, 100, 100) == "failed"


def test_one_atom_does_not_satisfy_overlap_requirement():
    result = validate_geometry(particle([0], [0]), np.zeros(3), [0, 5], {("C", "C"): 1})
    assert result["isolated_atoms"] == 1
    assert not result["mandatory_pass"]


@pytest.mark.parametrize("x,residues", [([-6,-4,4,6],[0,0,1,1]), ([0,2,4],[0,1,0]),
    ([0,3.4],[0,1]), ([0,1.4],[0,0]), ([0],[0]), ([0,2],[0,1])])
def test_independent_audit_agrees_with_geometry(x, residues):
    s = particle(x, residues)
    geometry = validate_geometry(s, np.zeros(3), [0, 30], {("C", "C"): 1})
    audit = connectivity_audit(s)
    assert audit["isolated_atoms"] == geometry["isolated_atoms"]
    assert audit["global_components"] == geometry["global_components"]
    assert audit["disconnected_residues"] == len(geometry["disconnected_residues"])


def test_isolated_failure_names_the_actual_atom():
    s = particle([0, 2, 20], [0, 0, 1])
    s.atoms[2] = Atom("SG", "S", "CYS", 1, {"serial": 321, "chain": "A", "residue_number": "77"})
    audit = connectivity_audit(s)
    geometry = validate_geometry(s, np.zeros(3), [0, 30], {("C", "C"): 1})
    with pytest.raises(MastercapError, match="serial 321 SG CYS A 77"):
        require_connectivity_audit(audit, geometry)


def test_audit_reports_two_clusters_and_disconnected_internal_residue():
    s = particle([0,2,8,10], [0,1,0,2])
    audit = connectivity_audit(s)
    assert audit["global_components"] == 2
    assert audit["isolated_atoms"] == 0
    assert audit["disconnected_residues"] == 1
    with pytest.raises(MastercapError, match="2 components"):
        require_connectivity_audit(audit, validate_geometry(s, np.zeros(3), [0, 20], {}))


def test_audit_does_not_reuse_search_graph_or_component_helper():
    s = particle([0, 3.4], [0, 1])
    with patch("mastercap.constraints.local_geometry", side_effect=AssertionError), patch("mastercap.constraints.components", side_effect=AssertionError):
        audit = connectivity_audit(s)
    assert audit["edge_count"] == 0
    assert audit["global_components"] == 2
    with pytest.raises(MastercapError, match="disagrees with validate_geometry"):
        require_connectivity_audit(audit, {"isolated_atoms": 0, "global_components": 1, "disconnected_residues": []})


def test_generate_records_successful_final_audit(tmp_path, small_pdb, capsys):
    import json
    from mastercap.cli import main
    prefix = tmp_path/"audited"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix),
                 "--seed", "42", "--allow-approximate", "--refine-updates", "0"]) == 2
    audit = json.loads(prefix.with_suffix(".json").read_text())["connectivity_audit"]
    assert audit["passed"] and audit["isolated_atoms"] == audit["disconnected_residues"] == 0
    assert "Success! Final connectivity audit:" in capsys.readouterr().err


def test_final_audit_disagreement_blocks_pdb_even_with_allow_approximate(tmp_path, small_pdb, capsys):
    import json
    from mastercap.cli import main
    def wrong_audit(structure, margin):
        result = connectivity_audit(structure, margin)
        result.update(isolated_atoms=1, passed=False,
                      isolated_atom_examples=result['component_representatives'][:1])
        return result
    prefix = tmp_path/'blocked'
    with patch('mastercap.cli.connectivity_audit',side_effect=wrong_audit):
        assert main(['generate','--input',str(small_pdb),'--output-prefix',str(prefix),
                     '--seed','42','--allow-approximate','--refine-updates','0']) == 1
    assert not prefix.with_suffix('.pdb').exists()
    diagnostic = json.loads((tmp_path/'blocked.diagnostic.json').read_text())
    assert diagnostic['connectivity_audit']['isolated_atoms'] == 1
    assert 'serial 1' in diagnostic['error']
    assert 'disagrees with validate_geometry' in capsys.readouterr().err


def test_pair_audit_names_subcovalent_sulfur_atoms():
    from mastercap.constraints import pair_distance_audit, require_pair_distance_audit
    from mastercap.model import Atom,Structure,MastercapError
    structure=Structure([Atom('SG','S','CYS',0,{'serial':17,'residue_number':'51'}),
                         Atom('SD','S','MET',1,{'serial':29,'residue_number':'120'})],np.array([[0.,0,0],[1.39,0,0]]))
    audit=pair_distance_audit(structure,{('S','S'):1.371})
    assert audit['below_covalent_floor_count'] == 1
    assert audit['below_limit_count'] == 0
    with pytest.raises(MastercapError,match=r'17 SG CYS.*29 SD MET'):
        require_pair_distance_audit(audit)


def test_pair_audit_peptide_exception_is_specific():
    from mastercap.constraints import pair_distance_audit
    from mastercap.model import Atom,Structure
    atoms=[Atom('C','C','ALA',0,{'chain':'A','segment':0,'residue_number':'1'}),
           Atom('N','N','GLY',1,{'chain':'A','segment':0,'residue_number':'2'})]
    structure=Structure(atoms,np.array([[0.,0,0],[1.30,0,0]]))
    assert pair_distance_audit(structure,{('C','N'):1.2})['below_covalent_floor_count'] == 0
    atoms[1].origin['residue_number']='3'
    assert pair_distance_audit(structure,{('C','N'):1.2})['below_covalent_floor_count'] == 1
    atoms[1].origin.update(residue_number='2',segment=1)
    assert pair_distance_audit(structure,{('C','N'):1.2})['below_covalent_floor_count'] == 1


def test_pair_audit_minima_match_brute_force():
    from itertools import combinations_with_replacement
    from mastercap.constraints import pair_distance_audit
    from mastercap.model import Atom,Structure
    rng=np.random.default_rng(7)
    elements=np.array(['C','N','O','S']*5)
    structure=Structure([Atom(e,e,'ALA',i//4) for i,e in enumerate(elements)],rng.normal(size=(20,3))*10)
    audit=pair_distance_audit(structure,{})
    for a,b in combinations_with_replacement(sorted(set(elements)),2):
        distances=[np.linalg.norm(structure.xyz[i]-structure.xyz[j]) for i in np.flatnonzero(elements==a)
                   for j in np.flatnonzero(elements==b) if structure.atoms[i].residue != structure.atoms[j].residue]
        assert audit['pairs'][a+'-'+b]['minimum_distance_angstrom'] == pytest.approx(min(distances))
