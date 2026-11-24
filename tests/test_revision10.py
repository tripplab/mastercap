"""Small synthetic unit fixtures only; no full-capsid or benchmark runs."""
import copy
import json

import numpy as np
import pytest

from mastercap.analysis import analyze, target_profiles
from mastercap.cli import main, parser
from mastercap.constraints import GeometryMetadata, local_geometry
from mastercap.generate import search, select_inventory, residue_transforms
from mastercap.model import Atom, Structure, MastercapError
from mastercap.radial import shell_volumes
from mastercap.repair import RepairRecovery, rigid_fit, support_shift
from mastercap.refinement import apply_crossing_push, accept_trial, bin_inventory, finish_wall_residues


def structure(xyz, residues=None, elements=None):
    xyz=np.asarray(xyz,float)
    residues=np.zeros(len(xyz),int) if residues is None else np.asarray(residues)
    elements=['C']*len(xyz) if elements is None else elements
    sulfur_residues=set(residues[np.asarray(elements)=='S'].tolist())
    atoms=[Atom('SG' if e=='S' else 'CA',e,'CYS' if r in sulfur_residues else 'GLY',
                int(r),{'reference_atom_index':i}) for i,(e,r) in enumerate(zip(elements,residues))]
    return Structure(atoms,xyz)


def local_state(disconnected=(),pairs=()):
    pairs=np.asarray(pairs,dtype=int).reshape(-1,2)
    return {'disconnected_residues':list(disconnected),'pairs':pairs,'violations':np.ones(len(pairs),bool)}


def recover(reference,current,local,anchors=(),edges=(90.,110.),limit=5):
    metadata=GeometryMetadata.build(current,{})
    tracker=RepairRecovery(metadata,stuck_iterations=5,reset_limit=limit)
    rng=np.random.default_rng(42)
    initial=copy.deepcopy(rng.bit_generator.state)
    for iteration in range(4):
        mask=tracker.apply(iteration,current,local,reference,np.zeros(3),np.asarray(edges),rng,anchors)
        assert not mask.any()
        assert rng.bit_generator.state==initial
    mask=tracker.apply(4,current,local,reference,np.zeros(3),np.asarray(edges),rng,anchors)
    return tracker,mask,rng


def test_rigid_fit_and_unchanged_transform_convention():
    source=np.array([[0,0,0],[1,0,0],[0,2,0],[0,0,3]],float)
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]],float)
    target=source@rotation+np.array([10.,20.,30.])
    fitted,r,t,error=rigid_fit(source,target)
    assert error<1e-6 and np.linalg.det(r)==pytest.approx(1.)
    np.testing.assert_allclose(fitted,target,atol=1e-6)
    np.testing.assert_allclose(t,[10,20,30],atol=1e-6)
    reference=structure(source);current=structure(target)
    old_a,old_b=source.mean(axis=0),target.mean(axis=0)
    u,_,vt=np.linalg.svd((source-old_a).T@(target-old_b))
    old_rotation=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
    old_fit=(source-old_a)@old_rotation+old_b
    record=residue_transforms(current,reference)[0]
    np.testing.assert_array_equal(record['rotation_row_vector'],old_rotation)
    np.testing.assert_array_equal(record['translation'],old_b-old_a@old_rotation)
    assert record['rigid_fit_rmsd_angstrom']==float(np.sqrt(np.mean(np.sum((target-old_fit)**2,axis=1))))


def test_disconnected_template_reset_restores_shape_and_centroid():
    reference=structure([[100,-.8,0],[100,.8,0]])
    current=structure([[100,-.8,0],[100,3.8,0]])
    centroid=current.xyz.mean(axis=0)
    tracker,mask,_=recover(reference,current,local_state([0]))
    assert mask.all() and tracker.report['total_resets']==1
    assert np.linalg.norm(current.xyz[0]-current.xyz[1])==pytest.approx(1.6,abs=1e-6)
    assert np.linalg.norm(current.xyz.mean(axis=0)-centroid)<.5
    assert tracker.disconnected_streak[0]==0
    assert tracker.report['events'][0]['kind']=='disconnected'


def test_reset_keeps_anchored_sulfur_radius():
    reference=structure([[100,-.8,0],[100,.8,0]],elements=['S','C'])
    current=structure([[100,-.8,0],[100,3.8,0]],elements=['S','C'])
    before=np.linalg.norm(current.xyz[0])
    tracker,mask,_=recover(reference,current,local_state([0]),anchors=[0])
    assert mask.all() and tracker.report['events'][0]['anchored']
    assert np.linalg.norm(current.xyz[0])==pytest.approx(before,abs=1e-6)
    assert np.linalg.norm(current.xyz[0]-current.xyz[1])==pytest.approx(1.6,abs=1e-6)


def test_wall_support_spare_and_impossible_reset():
    xyz=np.array([[111.,0,0],[112.,0,0]])
    fitted,shift=support_shift(xyz,np.zeros(3),np.array([90.,110.]))
    assert shift==pytest.approx(-2.01)
    radius=np.linalg.norm(fitted,axis=1)
    assert np.all(radius>=90.01-1e-9) and np.all(radius<=109.99+1e-9)
    assert np.linalg.norm(fitted[0]-fitted[1])==pytest.approx(1.)
    reference=structure([[99.2,0,0],[100.8,0,0]])
    current=structure([[98.,0,0],[102.,0,0]])
    original=current.xyz.copy()
    tracker,mask,_=recover(reference,current,local_state([0]),edges=(99.9,100.1))
    assert not mask.any() and tracker.report['skipped_support']==1
    assert tracker.disconnected_streak[0]==5 and tracker.report['total_resets']==0
    np.testing.assert_array_equal(current.xyz,original)


def test_unjam_moves_nonanchored_partner_tangentially():
    reference=structure([[100,0,0],[100,1.6,0],[100,4,0],[100,5.6,0]],
                        residues=[0,0,1,1],elements=['C','C','S','C'])
    current=structure([[100,0,0],[100,1.6,0],[100,.5,0],[100,2.1,0]],
                      residues=[0,0,1,1],elements=['C','C','S','C'])
    original=current.xyz.copy();before=np.linalg.norm(current.xyz[:2].mean(axis=0))
    tracker,mask,_=recover(reference,current,local_state(pairs=[(0,2)]),anchors=[2])
    assert mask.tolist()==[True,True,False,False]
    event=tracker.report['events'][0]
    assert event['kind']=='unjam' and event['residue']==0 and event['partner_residue']==1
    assert event['tangential_shift_angstrom']==.5
    assert abs(np.linalg.norm(current.xyz[:2].mean(axis=0))-before)<.05
    np.testing.assert_array_equal(current.xyz[2:],original[2:])
    assert not tracker.clash_streak.any()


def test_failed_unjam_restores_rng_state():
    reference=structure([[99.2,0,0],[100.8,0,0]])
    current=structure([[98.,0,0],[102.,0,0]])
    rng=np.random.default_rng(42);state=copy.deepcopy(rng.bit_generator.state)
    metadata=GeometryMetadata.build(current,{})
    tracker=RepairRecovery(metadata,stuck_iterations=5)
    for i in range(5):tracker.apply(i,current,local_state(pairs=[(0,1)]),reference,np.zeros(3),np.array([99.9,100.1]),rng)
    assert tracker.report['skipped_support']==1
    assert rng.bit_generator.state==state


def test_both_anchored_unjam_restores_moved_sulfur_radius():
    reference=structure([[100,0,0],[100,1.6,0],[100,4,0],[100,5.6,0]],
                        residues=[0,0,1,1],elements=['S','C','S','C'])
    current=structure([[100,0,0],[100,1.6,0],[100,.5,0],[100,2.1,0]],
                      residues=[0,0,1,1],elements=['S','C','S','C'])
    before=np.linalg.norm(current.xyz[2])
    tracker,mask,_=recover(reference,current,local_state(pairs=[(0,2)]),anchors=[0,2])
    assert mask.tolist()==[False,False,True,True]
    assert tracker.report['events'][0]['anchored']
    assert np.linalg.norm(current.xyz[2])==pytest.approx(before,abs=1e-6)


def test_streak_resets_limits_iteration_cap_and_event_bound():
    n=2000
    xyz=np.tile([[100.,-.8,0],[100.,.8,0]],(n,1));ids=np.repeat(np.arange(n),2)
    reference=structure(xyz,ids);current=structure(xyz,ids)
    current.xyz[1::2,1]+=3
    metadata=GeometryMetadata.build(current,{})
    tracker=RepairRecovery(metadata,stuck_iterations=5,reset_limit=1)
    assert tracker.cap==2
    local=local_state(range(n));rng=np.random.default_rng(7)
    for i in range(110):
        before=tracker.report['total_resets']
        tracker.apply(i,current,local,reference,np.zeros(3),np.array([90.,110.]),rng)
        assert tracker.report['total_resets']-before<=2
        if i<4:assert tracker.report['total_resets']==0
    assert tracker.resets.max()==1
    assert tracker.report['total_resets']==int(tracker.resets.sum())>200
    assert len(tracker.report['events'])==200
    assert len(tracker.report['reset_exhausted'])==tracker.report['total_resets']
    tracker.apply(111,current,local_state(),reference,np.zeros(3),np.array([90.,110.]),rng)
    assert not tracker.disconnected_streak.any() and not tracker.clash_streak.any()


def test_recovery_on_off_identity_before_stuck_limit(small_structure):
    analysis=analyze(small_structure,sigma=0);edges,profiles=target_profiles(analysis)
    outputs=[];states=[]
    for mode in ('on','off'):
        rng=np.random.default_rng(42)
        inventory,_=select_inventory(analysis,edges,profiles,rng)
        particle,report=search(inventory,analysis,edges,profiles,analysis.center,rng,
                               refine_updates=0,max_iterations=40,time_limit=30,repair_residue_reset=mode)
        assert particle is not None and report['first_feasible_iteration']<50
        assert report['repair_resets']['total_resets']==0
        outputs.append(particle.xyz);states.append(copy.deepcopy(rng.bit_generator.state))
    np.testing.assert_array_equal(*outputs)
    assert states[0]==states[1]


def wall_case(radial=None, elements=None, anchor=None, profiles=None, step=.5, blocked=None):
    radial=np.array([1.95,1.97,2.6,2.6] if radial is None else radial,float)
    elements=np.array(['C','N','C','N'] if elements is None else elements)
    ids=np.array([0,0,1,2]);groups={0:np.array([0,1]),1:np.array([2]),2:np.array([3])}
    indices={e:np.flatnonzero(elements==e) for e in sorted(set(elements))}
    edges=np.array([1.,2.,3.])
    profiles={e:np.array([0.,len(ix)])/shell_volumes(edges) for e,ix in indices.items()} if profiles is None else profiles
    force=np.zeros((4,3));direction=np.tile([1.,0,0],(4,1))
    info,selected=apply_crossing_push(force,radial,direction,indices,profiles,edges,.005,.05,1.,
        landing_mode='legacy',wall_selection='wall-residue',residues=ids,groups=groups,
        wall_residue_step=step,anchor_bins=anchor,blocked=blocked)
    return radial,elements,edges,profiles,force,info,selected,groups


def test_wall_residue_translation_once_and_both_directions():
    for radial,sign in [([1.1,1.2,2.6,2.6],1),([2.9,2.8,1.4,1.4],-1)]:
        profiles=None
        if sign<0:profiles={e:np.array([2.,0.])/shell_volumes(np.array([1.,2.,3.])) for e in ('C','N')}
        r,_,_,_,force,info,selected,_=wall_case(radial=radial,profiles=profiles)
        assert info['rigid_residues'].tolist()==[0]
        np.testing.assert_array_equal(force[0],force[1])
        assert 0<np.linalg.norm(force[0])<=.5
        assert np.sign(force[0,0])==sign
        assert len(np.unique(selected))==len(selected)
        assert np.isin([0,1],info['wall_atoms']).all()
        assert np.linalg.norm(np.c_[r,np.zeros((4,2))][0]-np.c_[r,np.zeros((4,2))][1])==pytest.approx(abs(r[0]-r[1]))


def test_multi_element_balances_match_fresh_histogram(monkeypatch):
    import mastercap.refinement as refinement
    captured=[];original=refinement.bin_inventory
    def inspect(*args,**kwargs):
        value=original(*args,**kwargs);captured.append(value[1]);return value
    monkeypatch.setattr(refinement,'bin_inventory',inspect)
    radial,elements,edges,profiles,force,info,selected,_=wall_case()
    assert selected.tolist()==[0,1]
    assert info['rigid_residues'].tolist()==[0]
    new=np.linalg.norm(np.c_[radial,np.zeros((4,2))]+force,axis=1)
    _,fresh=original(new,{e:np.flatnonzero(elements==e) for e in set(elements)},edges,profiles)
    for e in fresh:np.testing.assert_allclose(captured[0][e][2],fresh[e][2],atol=1e-12)
    assert all(np.allclose(v[2],0) for v in captured[0].values())


def test_wall_residue_anchor_support_blocked_and_missing_groups():
    profiles={'C':np.array([1.,1.])/shell_volumes(np.array([1.,2.,3.])),
              'S':np.array([0.,2.])/shell_volumes(np.array([1.,2.,3.]))}
    *_,info,selected,groups=wall_case(elements=['C','S','C','S'],anchor=(np.array([1]),np.array([0])),profiles=profiles)
    assert not len(info['rigid_residues'])
    *_,info,selected,groups=wall_case(radial=[1.95,2.99,2.6,2.6])
    assert not len(info['rigid_residues'])
    *_,info,selected,groups=wall_case(blocked=np.array([False,True,False,False]))
    assert not len(info['rigid_residues'])
    with pytest.raises(MastercapError,match='residue ids and groups'):
        apply_crossing_push(np.zeros((1,3)),np.array([1.1]),np.array([[1.,0,0]]),{'C':np.array([0])},
                            {'C':np.ones(2)},np.array([1.,2.,3.]),.005,.05,1.,landing_mode='legacy',wall_selection='wall-residue')


@pytest.mark.parametrize('backtrack',[False,True])
def test_rigid_rejection_or_uniform_backtracking_and_other_proposals(backtrack):
    s=structure([[-1.501,0,0],[.001,0,0],[1.251,0,0],[-4.,0,0]],residues=[0,0,1,2])
    limits={('C','C'):1.2};metadata=GeometryMetadata.build(s,limits)
    local=local_geometry(s,limits,metadata=metadata);trial=s.xyz.copy()
    trial[:2,0]+=.1;trial[3,0]+=.02
    accepted,info=accept_trial(s.xyz,trial,local,metadata,s.xyz.mean(axis=0),np.array([0.,10.]),
                               max_step=.1,rigid_residues=np.array([0]),backtrack=backtrack,wall_atoms=np.array([0,1]))
    delta=accepted[:2]-s.xyz[:2]
    np.testing.assert_allclose(delta, np.tile(delta[0],(2,1)),atol=1e-12)
    assert np.linalg.norm(accepted[0]-accepted[1])==pytest.approx(np.linalg.norm(s.xyz[0]-s.xyz[1]),abs=1e-9)
    assert accepted[3,0]-s.xyz[3,0]==pytest.approx(.02)
    if backtrack:assert 0<delta[0,0]<.1
    else:np.testing.assert_array_equal(accepted[:2],s.xyz[:2])
    first,details=accept_trial(s.xyz,trial,local,metadata,s.xyz.mean(axis=0),np.array([0.,10.]),max_step=.1)
    second,explicit=accept_trial(s.xyz,trial,local,metadata,s.xyz.mean(axis=0),np.array([0.,10.]),max_step=.1,rigid_residues=None)
    np.testing.assert_array_equal(first,second);assert details==explicit


def test_preserved_internal_floor_does_not_reject_rigid_translation():
    s=structure([[10.,0,0],[11.501,0,0]])
    metadata=GeometryMetadata.build(s,{('C','C'):1.5});local=local_geometry(s,{('C','C'):1.5},metadata=metadata)
    trial=s.xyz+np.array([0.,.02,0])
    accepted,_=accept_trial(s.xyz,trial,local,metadata,s.xyz.mean(axis=0),np.array([0.,10.]),rigid_residues=[0])
    np.testing.assert_allclose(accepted-s.xyz,np.tile([0,.02,0],(2,1)),atol=1e-12)


def test_wall_residue_acceptance_statistics_and_distance_invariance():
    radial,_,_,_,force,info,_,groups=wall_case()
    before=np.c_[radial,np.zeros((4,2))];after=before+force
    finish_wall_residues(info,info['rigid_residues'],info['rigid_residue_walls'],groups,before,after)
    assert info['wall_residue']['accepted_residues']==1 and info['wall_residue']['accepted_atoms']==2
    assert np.linalg.norm(after[0]-after[1])==pytest.approx(np.linalg.norm(before[0]-before[1]),abs=1e-9)


@pytest.mark.parametrize('backtrack',[False,True])
def test_undo_relation_and_wall_origin_diagnostics(backtrack):
    s=structure([[-1.504,0,0],[0,0,0],[1.504,0,0]],residues=[0,0,1])
    limits={('C','C'):1.5};metadata=GeometryMetadata.build(s,limits)
    local=local_geometry(s,limits,metadata=metadata);trial=s.xyz.copy();trial[[0,2],0]=[-1.3,1.3]
    _,info=accept_trial(s.xyz,trial,local,metadata,np.zeros(3),np.array([0.,10.]),max_step=.3,
                        backtrack=backtrack,wall_atoms=[0])
    assert info['undone_too_close_by_relation']=={'same_residue':1,'other_residue':1}
    assert info['per_element']['C']['undone_too_close_by_relation']=={'same_residue':1,'other_residue':1}
    assert info['wall_undone_too_close_by_relation']=={'same_residue':1,'other_residue':0}
    assert info['wall_undone_by_reason']['too_close']==1


def test_moving_atom_can_count_in_both_relation_sets():
    s=structure([[0.,0,0],[1.6,1.,0],[1.6,-1.,0]],residues=[0,0,1])
    metadata=GeometryMetadata.build(s,{('C','C'):1.5});local=local_geometry(s,{('C','C'):1.5},metadata=metadata)
    trial=s.xyz.copy();trial[0,0]=.6
    _,info=accept_trial(s.xyz,trial,local,metadata,s.xyz.mean(axis=0),np.array([0.,10.]),max_step=.6)
    assert info['undone_too_close_by_relation']=={'same_residue':1,'other_residue':1}


@pytest.mark.parametrize('flag,value',[('--repair-stuck-iterations','4'),('--repair-reset-limit','0'),
    ('--refine-wall-residue-step','.019'),('--refine-wall-residue-step','1.01')])
def test_invalid_cli_ranges_rejected_before_reading(flag,value,capsys):
    with pytest.raises(SystemExit) as error:main(['generate','--input','missing.pdb','--output-prefix','unused',flag,value])
    assert error.value.code==2
    assert 'Reading reference' not in capsys.readouterr().err


def test_help_and_unused_step_warning(capsys,tmp_path):
    text=parser().format_help()
    for flag in ('--repair-residue-reset','--repair-stuck-iterations','--repair-reset-limit','--refine-wall-residue-step'):
        assert flag in text
    assert 'wall-residue' in text
    assert main(['generate','--input','missing.pdb','--output-prefix',str(tmp_path/'unused'),'--refine-wall-residue-step','.5'])==1
    assert 'has no effect' in capsys.readouterr().err


def test_tiny_generate_records_recovery_and_wall_fields(tmp_path,small_pdb):
    prefix=tmp_path/'tiny'
    assert main(['generate','--input',str(small_pdb),'--output-prefix',str(prefix),'--seed','42',
        '--no-smoothing','--repair-residue-reset','off','--refine-wall-selection','wall-residue',
        '--refine-updates','2','--consolidate-iterations','0','--allow-approximate'])==2
    report=json.loads(prefix.with_suffix('.json').read_text());search_report=report['search']
    assert report['geometry']['mandatory_pass'] and report['connectivity_audit']['passed']
    assert search_report['repair_resets']['mode']=='off'
    assert search_report['repair_resets']['total_resets']==0
    assert search_report['refinement_settings']['wall_residue_step']==.5
    assert 'wall_residue' in search_report['push_history'][0]
    assert 'undone_too_close_by_relation' in search_report
    assert 'wall_undone_too_close_by_relation' in search_report
    assert all('resets_so_far' in item for item in search_report['repair_history'])


def test_reset_replaces_all_force_pieces_in_that_iteration(monkeypatch):
    from types import SimpleNamespace
    import mastercap.generate as generator
    reference=structure([[100,-.8,0],[100,.8,0]])
    current=structure([[100,-.8,0],[100,3.8,0]])
    edges=np.array([90.,110.]);profiles={'C':np.array([2.])/shell_volumes(edges)}
    analysis=SimpleNamespace(structure=reference,limits={('C','C'):1.},original=profiles)
    monkeypatch.setattr(generator,'initialize',lambda *a,**k:[])
    monkeypatch.setattr(generator,'pair_forces',lambda *a,**k:None)
    messages=[];checked=[];original_update=generator.update_positions
    def update(s,force,groups,mode,**kwargs):
        if messages and messages[-1].startswith('Repair reset at iteration 4:') and not checked:
            np.testing.assert_array_equal(force,np.zeros_like(force))
            before=s.xyz.copy();original_update(s,force,groups,mode,**kwargs)
            np.testing.assert_array_equal(before,s.xyz);checked.append(True)
        else:original_update(s,force,groups,mode,**kwargs)
    monkeypatch.setattr(generator,'update_positions',update)
    _,report=search(current,analysis,edges,profiles,np.zeros(3),np.random.default_rng(42),
                    repair_stuck_iterations=5,max_iterations=6,time_limit=5,progress_interval=1,
                    stage_progress=messages.append)
    assert checked and report['repair_resets']['total_resets']==1
    assert next(row for row in report['repair_history'] if row['iteration']==4)['resets_so_far']==1
