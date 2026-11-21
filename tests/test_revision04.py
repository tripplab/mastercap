"""Cross-cutting revision-04 reporting and deterministic search guarantees."""
import numpy as np
import pytest
from mastercap.analysis import analyze,target_profiles
from mastercap.constraints import GeometryMetadata,local_geometry
from mastercap.generate import select_inventory,search,SulfurAnchors,repair_diagnostics
from mastercap.model import Atom,Structure
from mastercap.refinement import errors_at_push,apply_crossing_push


def test_equal_push_errors_use_best_feasible_state():
    best=[{'iteration':i,'profile_updates':n,'total_profile_error':1/(n+1),'element_errors':{'S':1/(n+1)}}
          for i,n in ((0,0),(12,10),(18,15),(29,25))]
    pushes=[{'iteration':i} for i in range(25)]
    sampled=errors_at_push(best,pushes)
    assert set(sampled)=={'10','20','25'}
    assert sampled['10']['iteration']==12
    assert sampled['20']['iteration']==18
    assert sampled['25']['iteration']==29


def test_push_limited_search_is_seed_reproducible(small_structure):
    analysis=analyze(small_structure,sigma=0)
    edges,profiles=target_profiles(analysis)
    outputs=[]
    for _ in range(2):
        rng=np.random.default_rng(42)
        inventory,_=select_inventory(analysis,edges,profiles,rng)
        result,report=search(inventory,analysis,edges,profiles,analysis.center,rng,
                             refine_updates=5,refine_patience=0,time_limit=60)
        outputs.append((result.xyz,report['errors_at_push'],report['iterations']))
    assert np.array_equal(outputs[0][0],outputs[1][0])
    assert outputs[0][1:] == outputs[1][1:]


def test_inter_residue_floor_is_enforced_without_changing_measured_limits():
    structure=Structure([Atom('C','C','ALA',0),Atom('O','O','GLY',1)],np.array([[0.,0,0],[1.23,0,0]]))
    limits={('C','O'):1.219}
    metadata=GeometryMetadata.build(structure,limits,enforce_covalent_floor=True)
    local=local_geometry(structure,limits,metadata=metadata)
    assert local['violations'].sum()==1
    assert local['lower'][0]==pytest.approx(1.242)
    assert limits=={('C','O'):1.219}
    structure.atoms[1]=Atom('O','O','ALA',0)
    metadata=GeometryMetadata.build(structure,limits,enforce_covalent_floor=True)
    assert not local_geometry(structure,limits,metadata=metadata)['violations'].any()


def test_crossing_selection_cannot_move_anchor_out_of_target_bin():
    radial=np.array([1.99,1.2])
    force=np.zeros((2,3)); direction=np.tile([1.,0,0],(2,1))
    profiles={'S':np.array([0.,2.])}
    report,chosen=apply_crossing_push(force,radial,direction,{'S':np.arange(2)},profiles,np.array([1.,2.,3.]),
                                     .005,.05,1.,anchor_bins=(np.array([0]),np.array([0])))
    assert 0 not in chosen
    assert report['per_element']['S']['anchor_bin_selection_rejections']==1
    assert np.array_equal(force[0],np.zeros(3))


def test_actual_disconnected_cys_is_reported_as_anchored():
    structure=Structure([Atom('SG','S','CYS',0),Atom('C','C','CYS',0),Atom('CA','C','ALA',1)],
                         np.array([[0.,0,0],[5.,0,0],[2.5,0,0]]))
    metadata=GeometryMetadata.build(structure,{})
    local=local_geometry(structure,{},metadata=metadata)
    assert local['disconnected_residues']==[0]
    result=repair_diagnostics(structure,local,metadata,SulfurAnchors(np.array([0]),np.array([1.])),0,0)
    assert result['disconnected_by_residue_name']=={'CYS':1}
    assert result['anchored_disconnected_residues']==1


def test_time_limit_warns_that_seed_does_not_reproduce_result(small_structure,caplog):
    analysis=analyze(small_structure,sigma=0)
    edges,profiles=target_profiles(analysis)
    _,report=search(small_structure.copy(),analysis,edges,profiles,analysis.center,np.random.default_rng(42),
                    time_limit=1e-12)
    assert report['termination']=='time_limit'
    assert 'after 0 of 50 pushes' in caplog.text
    assert 'not reproducible by seed' in caplog.text
