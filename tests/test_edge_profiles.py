"""Revision 08 coverage. Added without execution at the user's request."""
import numpy as np
import pytest

from mastercap.analysis import Target, held_out_target, target_on_edges
from mastercap.constraints import GeometryMetadata, local_geometry, profile_metrics, acceptance
from mastercap.diagnostics import edge_concentration
from mastercap.model import Atom, Structure, MastercapError
from mastercap.radial import offset_edges, shell_volumes
from mastercap.refinement import (apply_crossing_push, resolve_landing_depth, accept_trial,
                                  error_summary, errors_at_push, finish_push_metrics, candidate_key)
from mastercap.step import step_profiles


def structure(radial):
    return Structure([Atom('CA','C','GLY',i) for i in range(len(radial))],
                     np.column_stack([radial,np.zeros((len(radial),2))]))


def test_uniform_volume_edge_ratio_and_wall_exclusion():
    rng=np.random.default_rng(42)
    radial=(10**3+rng.random(200_000)*(20**3-10**3))**(1/3)
    d=edge_concentration(radial,np.arange(10.,21.))
    for ratio in d['ratio_within'].values():
        assert ratio==pytest.approx(1.,rel=.15)
    walls=edge_concentration(np.array([10.,20.]),np.array([10.,20.]))
    assert walls['fraction_within']['0.005']==0
    assert walls['expected_within']['0.005']==0
    assert walls['ratio_within']['0.005'] is None


def test_sheets_at_edges_have_large_ratio_and_histogram_accounts_for_tails():
    radial=np.repeat(np.arange(11.,20.)+.002,100)
    d=edge_concentration(radial,np.arange(10.,21.))
    assert d['ratio_within']['0.005']>50
    assert sum(d['band_histogram']['fractions'])+d['band_histogram']['fraction_outside_bands']==pytest.approx(1.)


def test_offset_edges_and_piecewise_target_conserve_counts():
    edges=np.array([10.,11.,12.,13.])
    np.testing.assert_array_equal(offset_edges(edges,.5),[10,10.5,11.5,12.5,13])
    target=Target(edges,{'C':np.array([.01,.02,.03]),'N':np.array([.04,.02,.01])},None)
    shifted,profiles=held_out_target(target)
    for e in profiles:
        assert np.sum(profiles[e]*shell_volumes(shifted))==pytest.approx(np.sum(target.profiles[e]*shell_volumes(edges)),rel=1e-9)
    assert profiles['C'][1]==pytest.approx((.01*(11**3-10.5**3)+.02*(11.5**3-11**3))/(11.5**3-10.5**3))


def test_continuous_step_conservation_and_not_generation_bin_interpolation():
    edges,profiles,density=step_profiles((10.,14.,2.),{'C':1000.,'N':300.},1.)
    target=Target(edges,profiles,None,{'generated_bounds':{'inner':10.,'outer':14.},'edge_width':2.,
                                      'density':{'plateau_by_element':{e:density[e] for e in profiles}}})
    shifted,held=held_out_target(target)
    approximated=target_on_edges(Target(edges,profiles,None),shifted)
    for e,count in [('C',1000),('N',300)]:
        assert np.sum(held[e]*shell_volumes(shifted))==pytest.approx(count,rel=1e-9)
    assert not np.allclose(held['C'],approximated['C'])


def test_exact_own_bin_sheets_are_detected_by_held_out_error():
    edges=np.array([10.,11.,12.,13.])
    target=Target(edges,{'C':np.full(3,200.)/shell_volumes(edges)},None)
    s=structure(np.repeat(edges[:-1]+.002,200))
    m=profile_metrics(s,np.zeros(3),edges,target.profiles,held_out=held_out_target(target))
    assert m['total_profile_error']==pytest.approx(0,abs=1e-12)
    assert m['held_out']['total_profile_error']>.1
    assert m['held_out']['edges_count']==5
    # Held-out error never changes acceptance or ranking.
    without={k:v for k,v in m.items() if k!='held_out'}
    assert candidate_key(m)==candidate_key(without)
    assert acceptance({'mandatory_pass':True},m,0.,0.)=='valid'


def test_uniform_volume_own_and_held_out_errors_are_comparable():
    rng=np.random.default_rng(2026);edges=np.arange(10.,21.)
    radial=(10**3+rng.random(200_000)*(20**3-10**3))**(1/3)
    density=200_000/sum(shell_volumes(edges))
    target=Target(edges,{'C':np.full(10,density)},None)
    metrics=profile_metrics(structure(radial),np.zeros(3),edges,target.profiles,held_out=held_out_target(target))
    assert metrics['total_profile_error']<.02
    assert metrics['held_out']['total_profile_error']<.02


def crossing(radial,edges,mode='random',seed=42,depth='auto'):
    profiles={'C':np.r_[0.,np.full(len(edges)-2,len(radial)/(len(edges)-2))]/shell_volumes(edges)}
    force=np.zeros((len(radial),3));rng=np.random.default_rng(seed)
    details,chosen=apply_crossing_push(force,radial,np.tile([1.,0,0],(len(radial),1)),{'C':np.arange(len(radial))},
                                      profiles,edges,.005,.05,1.,rng=rng,landing_mode=mode,landing_depth=depth)
    return force,details,chosen,rng


def test_random_landing_is_seeded_bounded_and_respects_partial_destination():
    radial=np.full(1000,1.99);edges=np.array([1.,2.,2.2])
    force,info,chosen,_=crossing(radial,edges,depth=.25)
    again,_,_,_=crossing(radial,edges,depth=.25)
    np.testing.assert_array_equal(force,again)
    depth=radial[chosen]+force[chosen,0]-2.
    assert np.all(depth>=.002-1e-12) and np.all(depth<=.1)
    assert info['landing']['crossing_moves']==len(chosen)
    assert info['landing']['mean_landing_depth']==pytest.approx(depth.mean())


def test_configured_depth_limit_and_actual_destination_cap_are_separate():
    radial=np.full(100,1.97);edges=np.array([1.,1.99,2.98]);force=np.zeros((100,3))
    details,chosen=apply_crossing_push(force,radial,np.tile([1.,0,0],(100,1)),{'C':np.arange(100)},
                                      {'C':np.array([0.,100.])/shell_volumes(edges)},edges,.005,.05,1.,
                                      rng=np.random.default_rng(42),landing_depth=.5,bin_width=1.)
    assert details['landing']['depth_max']==.5
    assert np.all(radial[chosen]+force[chosen,0]-1.99<=.495)


def test_legacy_moves_and_rng_sequence_match_v0112_fixture():
    radial=np.array([1.98,1.99]);force,info,chosen,rng=crossing(radial,np.array([1.,2.,3.]),mode='legacy')
    np.testing.assert_allclose(force[:,0],[.022,.012],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(chosen,[1,0])
    assert rng.random()==np.random.default_rng(42).random()
    assert info['landing']['mode']=='legacy'


def test_non_reaching_drift_has_no_landing_draws():
    force,info,_,rng=crossing(np.full(10,1.6),np.array([1.,2.,3.]))
    np.testing.assert_allclose(force[:,0],.02)
    assert info['landing']['crossing_moves']==0
    assert info['landing']['mean_landing_depth'] is None
    assert rng.random()==np.random.default_rng(42).random()


@pytest.mark.parametrize('mode',['independent','residue'])
def test_larger_trial_update_keeps_legacy_cap_and_rigid_projection(mode):
    from mastercap.generate import update_positions
    s=Structure([Atom('CA','C','GLY',0),Atom('C','C','GLY',0)],np.array([[1.,0,0],[2.6,0,0]]))
    groups=[np.arange(2)];original=s.xyz.copy()
    force=np.tile([.3,0,0],(2,1))
    update_positions(s,force.copy(),groups,mode,max_move=.3)
    np.testing.assert_allclose(s.xyz-original,force)
    assert np.linalg.norm(s.xyz[0]-s.xyz[1])==pytest.approx(1.6)
    legacy=Structure(s.atoms,original.copy())
    update_positions(legacy,force.copy(),groups,mode)
    np.testing.assert_allclose(legacy.xyz-original,np.tile([.2,0,0],(2,1)))


@pytest.mark.parametrize('value',['nan','inf','bad','0.001','0.501','-1'])
def test_invalid_landing_depth_names_bin_width(value):
    with pytest.raises(MastercapError,match='bin width 1'):
        resolve_landing_depth(value,1.)


def test_enlarged_crossing_trial_detects_previously_distant_neighbor():
    s=Structure([Atom('SG','S','CYS',i) for i in range(2)],np.array([[0.,0,0],[3.9,0,0]]))
    limits={('S','S'):3.65};metadata=GeometryMetadata.build(s,limits)
    local=local_geometry(s,limits,metadata=metadata)
    assert len(local['pairs'])==0
    trial=s.xyz.copy();trial[0,0]=.3
    accepted,report=accept_trial(s.xyz,trial,local,metadata,s.xyz.mean(axis=0),np.array([0.,10.]),max_step=.3)
    np.testing.assert_array_equal(accepted,s.xyz)
    assert report['undone_by_reason']['too_close']==1


def test_twenty_pushes_random_landing_avoids_legacy_sheets():
    edges=np.array([1.,2.,3.])
    initial=np.r_[(1+(np.arange(12000)+.5)/12000*7)**(1/3),
                  (8+(np.arange(8000)+.5)/8000*19)**(1/3)]
    profiles={'C':np.array([10000.,10000.])/shell_volumes(edges)}
    measured={}
    for mode in ('random','legacy'):
        radial=initial.copy();rng=np.random.default_rng(42)
        for _ in range(20):
            force=np.zeros((len(radial),3))
            apply_crossing_push(force,radial,np.tile([1.,0,0],(len(radial),1)),{'C':np.arange(len(radial))},
                                profiles,edges,.005,.05,1.,rng=rng,landing_mode=mode)
            radial+=force[:,0]
        measured[mode]=edge_concentration(radial,edges)['ratio_within']['0.005']
    assert measured['random']<2
    assert measured['legacy']>4


def test_push_held_out_values_follow_the_same_best_candidate():
    m={'elements':{'C':{'profile_error':.1,'count_difference':0}},'total_profile_error':.1,
       'held_out':{'total_profile_error':.2,'elements':{'C':.2}}}
    summary=error_summary(m);record={'profile_before':summary}
    finish_push_metrics(record,summary)
    assert record['held_out_total']==.2
    best=[{'iteration':3,'profile_updates':0,**summary}]
    checkpoint=errors_at_push(best,[{'iteration':4}])['1']
    assert checkpoint['held_out_total']==.2 and checkpoint['held_out_elements']=={'C':.2}


def test_legacy_report_validation_adds_both_measures(tmp_path,small_pdb):
    import json
    from mastercap.cli import main
    prefix=tmp_path/'old';report_path=prefix.with_suffix('.json')
    assert main(['generate','--input',str(small_pdb),'--output-prefix',str(prefix),
                 '--seed','42','--no-smoothing','--refine-updates','0','--allow-approximate',
                 '--refine-landing','legacy'])==2
    saved=json.loads(report_path.read_text())
    saved['mastercap_version']='0.1.12'
    saved['configuration'].pop('refine_landing')
    saved['configuration'].pop('refine_landing_depth')
    saved['metrics'].pop('held_out')
    saved['diagnostics'].pop('edge_concentration')
    report_path.write_text(json.dumps(saved))
    output=tmp_path/'validated'
    assert main(['validate','--input',str(small_pdb),'--pdb',str(prefix.with_suffix('.pdb')),
                 '--report',str(report_path),'--output-prefix',str(output)])==2
    result=json.loads(output.with_suffix('.json').read_text())
    assert result['metrics']['held_out']['offset_fraction']==.5
    assert 'edge_concentration' in result['diagnostics']


def test_landing_help_defaults_and_invalid_depth_before_input(capsys):
    from mastercap.cli import main,parser
    help_text=parser().format_help()
    assert '--refine-landing' in help_text and '--refine-landing-depth' in help_text
    assert 'PROFILE ERROR MEASURES' in help_text
    assert main(['generate','--input','missing.pdb','--output-prefix','unused',
                 '--refine-landing-depth','0.001'])==1
    output=capsys.readouterr().err
    assert 'bin width' in output and 'Reading reference' not in output
