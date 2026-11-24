import numpy as np
import pytest
from mastercap.model import Atom,Structure
from mastercap.diagnostics import particle_diagnostics,bond_keys,nearest_nonbonded,layering_summary,mesh_sizes


def atoms(x,names=None,residues=None):
    return Structure([Atom(name,'C' if name!='N' else 'N','GLY',r) for name,r in
                      zip(names or ['CA']*len(x),residues if residues is not None else range(len(x)))],np.asarray(x,float))


def test_overlap_and_intersect_thresholds():
    # C spheres sum to 3.4; one overlap, two positive gaps (.4 and 1.0).
    s=atoms([[0,0,0],[3.,0,0],[0,4.4,0]])
    d=particle_diagnostics(s,np.zeros(3),np.array([0.,10.]),None)
    assert d['contacts']['overlap_per_atom']==pytest.approx(2/3)
    assert d['contacts']['near_gap_per_atom']['0.5']==0
    assert d['contacts']['near_gap_per_atom']['1.0']==pytest.approx(2/3)
    assert d['contacts']['gap_threshold_angstrom']['0.5']==pytest.approx(np.sqrt(3)*.5)


def test_peptide_excludes_whole_residue_pair_not_nonbonded_pair():
    s=atoms([[0,0,0],[1.33,0,0],[10,0,0],[11.33,0,0]],['C','N','CA','CA'])
    d=particle_diagnostics(s,np.zeros(3),np.array([0.,15.]),None)
    assert d['inter_residue']['peptide_bonded_residue_pairs']==1
    assert d['inter_residue']['close_contacts_per_atom']['2.0']==.5
    samples=nearest_nonbonded(s,bond_keys(s))
    assert samples['C','C'].min()==pytest.approx(1.33)


def test_walls_and_expected_counts():
    r=np.array([10.001,10.02,10.4,14.6,14.95,14.999])
    s=atoms(np.column_stack([r,np.zeros((len(r),2))]))
    edges=np.array([10.,15.]);target={'C':np.array([.1])}
    d=particle_diagnostics(s,np.zeros(3),edges,target)
    assert d['walls']['inner']['atoms_within']=={'0.01':1,'0.1':2,'0.5':3}
    assert d['walls']['outer']['atoms_within']=={'0.01':1,'0.1':2,'0.5':3}
    assert d['walls']['outer']['expected_within_0p1']==pytest.approx(.1*4*np.pi/3*(15**3-14.9**3))
    assert d['walls']['inner']['expected_0p5'][0]==.1


def test_layering_uniform_vs_four_angstrom_bands():
    assert layering_summary(np.ones(40))['relative_sd']==0
    assert layering_summary(np.tile([4.,0,0,0],10))['relative_sd']>.2


@pytest.mark.parametrize('bad',['','0','-1','nan','0.5,0.5'])
def test_invalid_mesh_sizes(bad):
    with pytest.raises(ValueError):mesh_sizes(bad)


def test_layering_common_helper_and_optional_drift_summary():
    from mastercap.diagnostics import radial_layering
    s=atoms([[1,0,0],[2,0,0],[3,0,0]])
    d=particle_diagnostics(s,np.zeros(3),np.array([0.,20.]),None,transforms=[{'rigid_fit_rmsd_angstrom':.1},{'rigid_fit_rmsd_angstrom':.3}])
    assert d['layering']==radial_layering(np.array([1.,2.,3.]),np.array([0.,20.]))
    assert d['residue_drift']['median_angstrom']==pytest.approx(.2)
    assert d['residue_drift']['p95_angstrom']==pytest.approx(.29)


def test_mesh_size_changes_only_near_gap_measurements():
    s=atoms([[0,0,0],[3.2,0,0],[7.1,0,0]])
    a=particle_diagnostics(s,np.zeros(3),np.array([0.,10.]),None,element_sizes=(.5,1.))
    b=particle_diagnostics(s,np.zeros(3),np.array([0.,10.]),None,element_sizes=(.25,.5))
    for block in ('inter_residue','walls','layering','residue_drift'):assert a[block]==b[block]
    assert a['contacts']['overlap_per_atom']==b['contacts']['overlap_per_atom']
