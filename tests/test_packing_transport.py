import numpy as np
import pytest
from mastercap.generate import wall_depths
from mastercap.refinement import apply_crossing_push,resolve_drift_step,effective_patience,classify_stall
from mastercap.radial import shell_volumes


def test_seeded_wall_depths_and_legacy_does_not_consume_rng():
    edges=np.array([90.,91.,139.,140.])
    for width in (None,4.,0.):
        a,summary=wall_depths(edges,1000,np.random.default_rng(42),'band',width)
        b,_=wall_depths(edges,1000,np.random.default_rng(42),'band',width)
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x,y)
            assert np.all(x>= (0 if width else .002)) and np.all(x<= (width or 1))
        assert summary['clamp_events']==0
    rng=np.random.default_rng(42)
    assert wall_depths(edges,10,rng,'legacy')[0] is None
    assert rng.random()==np.random.default_rng(42).random()


def simulate(drift):
    r=np.r_[np.full(100,.6),np.full(10,1.5)]
    edges=np.array([0.,1.,2.]);profiles={'C':np.array([55.,55.])/shell_volumes(edges)}
    first=None
    for push in range(50):
        f=np.zeros((len(r),3))
        apply_crossing_push(f,r,np.tile([1.,0,0],(len(r),1)),{'C':np.arange(len(r))},profiles,edges,
                            .005,.05,.2,drift_step=drift)
        before=np.count_nonzero(r>=1);r+=f[:,0]
        if np.count_nonzero(r>=1)>before and first is None:first=push+1
    return first,abs(np.count_nonzero(r<1)-55)


def test_edge_transport_arrives_without_two_hundred_pushes():
    first,error=simulate('auto')
    assert first<=21 and error<10
    assert simulate('legacy')==(None,45)


def test_quota_caps_drifters_and_patience_matches_transit():
    r=np.r_[np.full(90,.6),np.full(10,1.5)]
    edges=np.array([0.,1.,2.]);profiles={'C':np.array([89.,11.])/shell_volumes(edges)}
    info,chosen=apply_crossing_push(np.zeros((100,3)),r,np.tile([1.,0,0],(100,1)),{'C':np.arange(100)},
                                  profiles,edges,.005,.05,.2,transport_factor=2)
    assert len(chosen)<=2
    assert info['mean_distance_to_edge_of_drifters']==pytest.approx(.4)
    assert resolve_drift_step('auto',1)==.02
    assert resolve_drift_step('auto',4)==.05
    assert effective_patience(10,1,.02)==50
    assert effective_patience(0,1,.02)==0
    assert effective_patience(10,1,None)==10
    assert classify_stall([{'drift_proposals':2,'bin_crossings':0}],10)=='transport_starved'


def test_band_repairs_points_that_only_round_outside_the_support():
    from mastercap.generate import wall_clamp_targets
    raw=np.array([1.0001,1.1,1.9999])
    checked=np.array([.9999,1.1,2.0001])
    depths=[np.full(3,.4),np.full(3,.3)]
    np.testing.assert_allclose(wall_clamp_targets(raw,checked,np.array([1.,2.]),depths),[1.4,1.1,1.7])
    np.testing.assert_array_equal(wall_clamp_targets(raw,checked,np.array([1.,2.]),None),np.clip(raw,1.002,1.998))


def test_legacy_search_coordinates_match_v0111_fixture():
    from mastercap.analysis import analyze,target_profiles
    from mastercap.generate import search
    from mastercap.model import Atom,Structure
    s=Structure([Atom('CA','C','GLY',i//2) for i in range(4)],
                np.array([[-2.4,0,0],[-.8,0,0],[.8,0,0],[2.4,0,0]]))
    a=analyze(s,sigma=0);edges,profiles=target_profiles(a)
    particle,report=search(s.copy(),a,edges,profiles,a.center,np.random.default_rng(42),mode='independent',distribution='regular',
                           max_iterations=1000,time_limit=30,refine_updates=3,refine_patience=0,
                           wall_clamp='legacy',refine_drift_step='legacy')
    # Generated independently with main@275d754, not recomputed from current code.
    np.testing.assert_array_equal(particle.xyz,[[-1.846,.56,-1.105],[-.386,.54,-.447],[1.569,.256,1.399],[1.15,-.863,.3]])
    assert [p['iteration'] for p in report['push_history']]==[13,14,15]
    assert all(p['atoms_pushed']==0 and p['acceptance_fraction']==1 for p in report['push_history'])


def test_cumulative_routing_can_traverse_an_intervening_balanced_bin():
    # An endpoint surplus cannot reach its deficit via adjacent surplus/deficit matching.
    r=np.array([.6]*8+[1.6]*4+[2.6]*2,dtype=float)
    edges=np.arange(4.);profiles={'C':np.array([4.,4.,6.])/shell_volumes(edges)}
    kwargs=dict(radial=r,direction=np.tile([1.,0,0],(len(r),1)),element_indices={'C':np.arange(len(r))},
                profiles=profiles,edges=edges,strength=.005,max_step=.05,fraction=1.)
    local,_=apply_crossing_push(np.zeros((len(r),3)),**kwargs,routing='local')
    cumulative,selected=apply_crossing_push(np.zeros((len(r),3)),**kwargs,routing='cumulative')
    assert local['selected_atoms']==0
    assert cumulative['drift_proposals']>0
    assert (r[selected]<2).any()


def test_guarded_backtracking_keeps_a_nonzero_feasible_partial_move():
    from mastercap.model import Atom,Structure
    from mastercap.constraints import GeometryMetadata,local_geometry,validate_geometry
    from mastercap.refinement import accept_trial
    s=Structure([Atom('CA','C','GLY',i) for i in range(3)],np.array([[-1.51,0,0],[0.,0,0],[1.51,0,0]]))
    limits={('C','C'):1.5};metadata=GeometryMetadata.build(s,limits)
    local=local_geometry(s,limits,metadata=metadata);trial=s.xyz.copy();trial[1,0]=-.02
    full,_=accept_trial(s.xyz,trial,local,metadata,np.zeros(3),np.array([0.,4.]))
    np.testing.assert_array_equal(full,s.xyz)
    accepted,details=accept_trial(s.xyz,trial,local,metadata,np.zeros(3),np.array([0.,4.]),backtrack=True)
    assert accepted[1,0]==pytest.approx(-.005)
    assert details['backtracked_accepted_atoms']==1 and details['atoms_undone']==0
    assert details['undone_by_reason']['too_close']==0
    assert details['backtracking_guard_reasons']['too_close']==1
    final=Structure(s.atoms,accepted)
    assert validate_geometry(final,np.zeros(3),np.array([0.,4.]),limits)['mandatory_pass']


def test_rigid_trials_never_use_coordinate_interpolation():
    from mastercap.model import Atom,Structure
    from mastercap.constraints import GeometryMetadata,local_geometry
    from mastercap.refinement import accept_trial
    s=Structure([Atom('CA','C','GLY',i) for i in range(3)],np.array([[-1.51,0,0],[0.,0,0],[1.51,0,0]]))
    metadata=GeometryMetadata.build(s,{('C','C'):1.5});local=local_geometry(s,{('C','C'):1.5},mode='residue',metadata=metadata)
    trial=s.xyz.copy();trial[1,0]=-.02
    accepted,details=accept_trial(s.xyz,trial,local,metadata,np.zeros(3),np.array([0.,4.]),mode='residue',backtrack=True)
    np.testing.assert_array_equal(accepted,s.xyz)
    assert 'backtracking_rounds' not in details
