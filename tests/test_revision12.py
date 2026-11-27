"""Synthetic tests for shared thickness changes; no capsid benchmarks."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest

from mastercap.analysis import Analysis, build_target, resolve_geometry, target_profiles
from mastercap.calibration import profile_settings, resampling_levels, calibrate, settings_from_generation, check_settings
from mastercap.cli import configured_target, rebuild_target, verify_geometry_scaling, parser, resolve_target_options, target_settings
from mastercap.model import Atom, Structure, MastercapError
from mastercap.radial import shell_volumes
from mastercap.splice import plan_splice, apply_splice, SplicePlan, _smooth
from mastercap.step import StepFit


@pytest.fixture
def synthetic(monkeypatch):
    edges=np.arange(100.,161.)
    r=(edges[:-1]+edges[1:])/2
    xyz=np.vstack([np.c_[r,np.zeros((len(r),2))],np.c_[-r,np.zeros((len(r),2))]])
    structure=Structure([Atom('CA','C','GLY',i) for i in range(len(xyz))],xyz,
                        provenance={'sha256':'synthetic','path':'synthetic.pdb','model':1})
    original={e:np.full(60,d) for e,d in [('C',.0001),('N',.00004),('O',.00004),('S',.000001)]}
    analysis=Analysis(structure,np.zeros(3),edges,original,copy.deepcopy(original),{}, {},[],0,nominal_bin_width=1.)
    def fit(radii,weights=None,edge_width=4.):
        return StepFit(105.,155.,edge_width,50.,105.-edge_width/2,155.+edge_width/2,{}, {},[0.,0.],1)
    monkeypatch.setattr('mastercap.step.fit_step',fit)
    return analysis,fit(None)


def options(**kwargs):
    args=parser().parse_args(['generate','--input','synthetic.pdb','--output-prefix','unused'])
    resolve_target_options(args)
    for k,v in kwargs.items():setattr(args,k,v)
    return args


def scaled(a,g,f=1.,step=False,**kwargs):
    inner,outer,block=resolve_geometry(a,step=step,edge_width=4.,curvature_factor=f,thickness_factor=g,**kwargs)
    plan=SplicePlan.from_report(block['splice'],a) if 'splice' in block else None
    target=build_target(a,inner,outer,step=step,edge_width=4.,splice=plan)
    target.geometry_scaling=block
    return target,block


@pytest.mark.parametrize('g',[.8,1.1,1.25])
def test_shared_absolute_delta_and_arm_ratios(synthetic,g):
    a,_=synthetic
    blocks=[scaled(a,g,step=step)[1] for step in (False,True)]
    assert blocks[0]['delta_bins']==blocks[1]['delta_bins']
    for b in blocks:
        assert b['delta_angstrom']==pytest.approx((b['generated']['thickness']-b['reference']['thickness']),abs=1e-9)
        assert b['thickness_ratio']==b['arm_thickness_ratio']
        assert b['thickness_factor']==g
        assert b['thickness_rule']=='shared_absolute_delta'
    assert blocks[0]['arm_thickness_ratio']!=blocks[1]['arm_thickness_ratio']


@pytest.mark.parametrize('delta,n',[(2.4,2),(2.6,3),(-2.4,-2),(-2.6,-3)])
def test_bin_rounding_and_effective_factor(synthetic,delta,n):
    a,_=synthetic
    _,b=scaled(a,1+delta/50)
    assert b['delta_bins']==n
    assert b['thickness_factor_effective']==1+n/50
    assert b['rounding']=='nearest whole reference bin'


def test_zero_bin_rounding_refusal_names_inputs(synthetic):
    with pytest.raises(MastercapError,match=r'G=.*Delta_req=.*w=.*minimum usable \|G - 1\|'):
        scaled(synthetic[0],1.001)


def test_fixed_score_avoids_spike_and_is_independent_of_run_sigma(synthetic):
    a,fit=synthetic;a.original['C'][30]=.01
    plans=[]
    for sigma in (0.,1.,2.):
        a.smoothing_sigma=sigma
        a.smoothed={e:_smooth(p,a.edges,sigma) for e,p in a.original.items()}
        plans.append(plan_splice(a,fit,5,sign=-1))
    assert len({(p.cut_bin,p.mismatch_total) for p in plans})==1
    p=plans[0]
    assert p.mismatch_total<1e-8
    assert not p.cut_bin-p.window_bins <= 30 < p.cut_bin+p.n_bins+p.window_bins


def gradient(a):
    a.original={e:(np.arange(60)+.1)*p[0] for e,p in a.original.items()}
    a.smoothed=copy.deepcopy(a.original)


def test_gradient_refused_or_explicitly_relaxed(synthetic):
    a,fit=synthetic;gradient(a)
    with pytest.raises(MastercapError,match='best i=.*candidates=.*median=.*knowingly'):
        plan_splice(a,fit,12,sign=-1)
    assert plan_splice(a,fit,12,sign=-1,max_mismatch=1).mismatch_total>.1


def test_flat_core_tie_break_and_no_candidates(synthetic):
    a,fit=synthetic
    p=plan_splice(a,fit,5,sign=-1)
    lo,hi=p.allowed_bins
    valid=range(lo+p.window_bins,hi-p.n_bins-p.window_bins+1)
    assert p.cut_bin==min(valid,key=lambda i:(abs(i+2.5-(lo+hi)/2),i))
    assert p.mismatch_total==0
    largest=hi-lo-2*p.window_bins
    for sign in (-1,1):
        with pytest.raises(MastercapError,match=f'n={largest}'):
            plan_splice(a,fit,largest+1,sign=sign)


@pytest.mark.parametrize('g,mode',[ (.8,'remove'),(1.2,'duplicate')])
def test_splice_arrays_walls_edges_and_report(synthetic,g,mode):
    a,_=synthetic
    # Distinct markers outside the scoring core expose any wall corruption.
    a.original['C'][:8]*=np.arange(1,9);a.original['C'][-8:]*=np.arange(8,0,-1)
    t,b=scaled(a,g);p=t.splice;i=p['cut_bin'];n=p['n_bins']
    lo,hi=p['allowed_bins']
    assert p['mode']==mode
    for e,orig in a.original.items():
        expected=np.r_[orig[:i],orig[i+n:]] if mode=='remove' else np.r_[orig[:i+n],orig[i:i+n],orig[i+n:]]
        np.testing.assert_array_equal(t.profiles[e],expected)
        np.testing.assert_array_equal(t.reference_on_edges[e],expected)
        np.testing.assert_array_equal(expected[:lo],orig[:lo])
        np.testing.assert_array_equal(expected[-(60-hi):],orig[hi:])
    assert len(t.edges)==61+(-n if mode=='remove' else n)
    np.testing.assert_array_equal(np.diff(t.edges),np.ones(len(t.edges)-1))
    assert t.edges[-1]==b['generated']['outer'] and t.edges[0]==b['generated']['inner']
    expected_keys={'mode','cut_bin','n_bins','window_bins','window_angstrom','score_sigma_angstrom','wall_keep_angstrom',
                   'allowed_reference_radii','band_reference_radii','seam_generated_radius','seam_definition','mismatch_total',
                   'mismatch_by_element','candidates','candidate_median_mismatch','threshold','reference_bins','generated_bins',
                   'expected_atoms_reference','expected_atoms_generated','note','candidate_mismatch'}
    assert expected_keys<=p.keys() and len(p['candidate_mismatch'])==p['candidates']
    assert p['expected_atoms_generated']==pytest.approx(sum(np.sum(x*shell_volumes(t.edges)) for x in t.profiles.values()))
    assert p['seam_generated_radius']==t.edges[i if mode=='remove' else i+n]


def test_post_splice_smoothing_preserves_counts(synthetic):
    a,_=synthetic;a.original['C'][25]*=2;a.smoothing_sigma=1.
    t,_=scaled(a,.8)
    assert not np.array_equal(t.profiles['C'],t.reference_on_edges['C'])
    for e,p in t.profiles.items():
        assert np.sum(p*shell_volumes(t.edges))==pytest.approx(np.sum(t.reference_on_edges[e]*shell_volumes(t.edges)),rel=1e-12)


def test_no_factors_and_curvature_only_keep_legacy_target(synthetic):
    a,_=synthetic;a.smoothing_sigma=1.
    a.smoothed={e:_smooth(p,a.edges,1.) for e,p in a.original.items()}
    unchanged=build_target(a)
    np.testing.assert_array_equal(unchanged.edges,a.edges)
    for e in a.original:np.testing.assert_array_equal(unchanged.profiles[e],a.smoothed[e])
    assert resolve_geometry(a,step=False,edge_width=4.)==(None,None,None)
    t,b=scaled(a,1.,1.5)
    edges,profiles=target_profiles(a,b['generated']['inner'],b['generated']['outer'])
    np.testing.assert_array_equal(t.edges,edges)
    for e in profiles:np.testing.assert_array_equal(t.profiles[e],profiles[e])
    assert b['thickness_method']=='shift' and 'splice' not in b


def test_curvature_and_thickness_compose_without_stretch(synthetic):
    a,_=synthetic
    plain,_=scaled(a,.8);curved,_=scaled(a,.8,1.5)
    for e in a.original:np.testing.assert_array_equal(plain.profiles[e],curved.profiles[e])
    np.testing.assert_allclose(curved.edges,plain.edges+(a.outer/1.5-a.outer),atol=1e-12)


def test_explicit_radii_still_stretch_and_warn(synthetic,caplog):
    a,_=synthetic
    args=options(inner_radius=110.,outer_radius=150.)
    t=configured_target(args,a)
    edges,profiles=target_profiles(a,110.,150.)
    np.testing.assert_array_equal(t.edges,edges)
    for e in profiles:np.testing.assert_array_equal(t.profiles[e],profiles[e])
    assert len([r for r in caplog.records if 'Explicit radii stretch' in r.message])==1
    assert target_settings(args,a,t.edges)['thickness_method']=='stretch'


def test_step_partner_feasibility_is_advisory(synthetic,caplog):
    a,_=synthetic
    _,b=scaled(a,.8,step=True)
    assert b['paired_layered_splice']['feasible']
    gradient(a)
    caplog.set_level('INFO')
    t=configured_target(options(step_profile=True,thickness_factor=.8),a)
    assert not t.geometry_scaling['paired_layered_splice']['feasible']
    assert 'reason' in t.geometry_scaling['paired_layered_splice']
    assert 'Paired layered splice is not feasible' in caplog.text


def saved_target(a,args,t):
    return {'configuration':vars(args).copy(),'analysis_options':{'step_profile':args.step_profile,'step_edge_width':4.,'width':1.},
            'geometry_scaling':copy.deepcopy(t.geometry_scaling),'target':{'edges':t.edges.tolist()}}


@pytest.mark.parametrize('step',[False,True])
def test_validate_rebuild_and_splice_tampering(synthetic,step):
    a,_=synthetic;args=options(step_profile=step,thickness_factor=.8)
    t=configured_target(args,a);saved=saved_target(a,args,t)
    rebuilt=rebuild_target(a,saved);verify_geometry_scaling(rebuilt,saved)
    np.testing.assert_array_equal(rebuilt.edges,t.edges)
    for e in t.profiles:np.testing.assert_array_equal(rebuilt.profiles[e],t.profiles[e])
    if not step:
        for key,value in [('cut_bin',0),('mode','duplicate'),('n_bins',1),('window_bins',99),('threshold',.9),('mismatch_total',.01)]:
            bad=copy.deepcopy(saved);bad['geometry_scaling']['splice'][key]=value
            with pytest.raises(MastercapError,match=f'Recomputed splice differs.*{key}'):
                verify_geometry_scaling(rebuild_target(a,bad),bad)
    bad=copy.deepcopy(saved);bad['geometry_scaling']['delta_bins']+=1
    with pytest.raises(MastercapError,match='Recomputed splice differs.*delta_bins'):verify_geometry_scaling(rebuilt,bad)
    bad=copy.deepcopy(saved);bad['target']['edges'][2]+=.001
    with pytest.raises(MastercapError,match='Recomputed splice differs.*edges'):verify_geometry_scaling(rebuilt,bad)


def test_legacy_v016_rebuild_warns_and_stretches(synthetic,caplog):
    a,_=synthetic;args=options(inner_radius=100.,outer_radius=150.)
    t=build_target(a,100.,150.)
    old={'generated':{'inner':100.,'outer':150.}}
    saved={'configuration':vars(args),'analysis_options':{'width':1.,'step_profile':False},'geometry_scaling':old}
    rebuilt=rebuild_target(a,saved);verify_geometry_scaling(rebuilt,saved)
    np.testing.assert_array_equal(rebuilt.edges,t.edges)
    assert 'Legacy v0.1.16 report' in caplog.text


def test_resized_counting_uses_target_totals_and_skips_undefined_levels(synthetic,monkeypatch):
    a,_=synthetic;t,_=scaled(a,.8)
    levels=resampling_levels(a,t.edges,t.profiles,replicates=3,seed=42,resized=True)
    for level in ('self','orientation','placement'):
        assert levels[level]=={'skipped':'not defined for resized layered targets'}
    # Match a direct multinomial replicate using target counts, not reference atoms.
    from mastercap.calibration import errors_from_counts
    rng=np.random.default_rng(42);v=shell_volumes(t.edges)
    counts={e:rng.multinomial(round(np.sum(p*v)),p*v/np.sum(p*v)) for e,p in t.profiles.items()}
    one=resampling_levels(a,t.edges,t.profiles,replicates=1,seed=42,resized=True)
    assert {e:s['mean'] for e,s in one['counting'].items()}==errors_from_counts(counts,t.edges,t.profiles)
    # At reference size this new flag preserves the exact prior RNG/results.
    assert resampling_levels(a,a.edges,a.smoothed,replicates=2)==resampling_levels(a,a.edges,a.smoothed,replicates=2,resized=False)
    settings=profile_settings(a,t.edges,1.,thickness_factor=.8)
    assert settings['resized'] and settings['thickness_method']=='splice'
    result=calibrate(a,t.edges,t.profiles,settings,replicates=2,floor_basis='counting')
    assert result['suggested']['basis']=='counting'


def test_sulfur_reference_resize_refused_and_unused_splice_options_warn_once(caplog):
    args=parser().parse_args(['generate','--input','unused','--output-prefix','unused',
                              '--thickness-factor','.8','--sulfur-residue-placement','reference'])
    with pytest.raises(MastercapError,match='requires the reference geometry'):resolve_target_options(args)
    for extra in ([],['--step-profile','--thickness-factor','.8']):
        caplog.clear()
        args=parser().parse_args(['generate','--input','unused','--output-prefix','unused',
                                 '--splice-window','1','--splice-wall-keep','3',*extra])
        resolve_target_options(args)
        assert len([r for r in caplog.records if 'has no effect without a layered thickness factor' in r.message or 'have no effect without a layered thickness factor' in r.message])==1


@pytest.mark.parametrize('flag,value',[('--splice-window','0'),('--splice-wall-keep','nan'),('--splice-max-mismatch','1.1')])
def test_invalid_splice_cli_options(flag,value):
    with pytest.raises(SystemExit):parser().parse_args(['calibrate','--input','unused',flag,value])


def test_search_metrics_uses_spliced_original_profiles(synthetic,monkeypatch):
    import mastercap.generate as generation
    a,_=synthetic
    a.original={'C':a.original['C']};a.smoothed={'C':a.smoothed['C']}
    t,_=scaled(a,.8)
    class Measured(Exception):pass
    def metric(candidate,center,edges,profiles,original,**kwargs):
        assert original is t.reference_on_edges
        assert all(len(p)==len(edges)-1 for p in original.values())
        raise Measured
    monkeypatch.setattr(generation,'profile_metrics',metric)
    monkeypatch.setattr(generation,'initialize',lambda *args,**kwargs:[])
    with pytest.raises(Measured):
        generation.search(a.structure,a,t.edges,t.profiles,a.center,np.random.default_rng(42),
                          max_iterations=0,original_profiles=t.reference_on_edges,target_definition=t)


def test_density_free_splice_refused(synthetic):
    a,fit=synthetic
    a.original={e:np.zeros_like(p) for e,p in a.original.items()}
    with pytest.raises(MastercapError,match='no density'):
        plan_splice(a,fit,4,sign=-1)


def test_resized_calibrate_cli_counting_and_output_files(synthetic,tmp_path,small_pdb,monkeypatch,capsys):
    import json
    import csv
    from mastercap.cli import main
    a,_=synthetic
    monkeypatch.setattr('mastercap.cli.analyze',lambda *args,**kw:a)
    prefix=tmp_path/'calib'
    flags=['calibrate','--input',str(small_pdb),'--output-prefix',str(prefix),
           '--thickness-factor','.8','--no-smoothing','--replicates','2','--csv','--plot']
    assert main(flags)==0
    result=json.loads((tmp_path/'calib_calibration.json').read_text())
    calibration=result['calibration']
    assert calibration['settings']['resized']
    assert calibration['settings']['splice_wall_keep']==6
    assert calibration['suggested']['basis']=='counting'
    assert calibration['levels']['self']['skipped']=='not defined for resized layered targets'
    rows=list(csv.DictReader((tmp_path/'calib_calibration.csv').open()))
    assert {row['source'] for row in rows}=={'counting'}
    assert next(int(row['n_atoms']) for row in rows if row['element']=='C')==round(calibration['target_counts']['C'])
    from matplotlib.image import imread
    assert imread(tmp_path/'calib_calibration.png').size>0
    assert main([*flags,'--overwrite','--floor-basis','orientation'])==1
    assert 'use --floor-basis counting' in capsys.readouterr().err


def test_resized_generated_assessment_has_no_orientation_gap(synthetic,tmp_path):
    from mastercap.pdbio import pdb_text
    a,_=synthetic
    a.original={'C':a.original['C']};a.smoothed={'C':a.smoothed['C']}
    t,_=scaled(a,.8)
    path=tmp_path/'generated.pdb';path.write_text(pdb_text(a.structure)[0])
    settings=profile_settings(a,t.edges,1.,thickness_factor=.8)
    report=calibrate(a,t.edges,t.profiles,settings,replicates=2,generated=[path],floor_basis='counting')
    assessed=report['generated'][0]
    assert assessed['multiples_orientation'] is None and assessed['gap_closed'] is None


def test_calibration_splice_settings_round_trip_and_mismatch(synthetic):
    a,_=synthetic;args=options(thickness_factor=.8)
    t=configured_target(args,a);saved=saved_target(a,args,t)
    saved.update(analysis=a.summary())
    saved['analysis_options'].update(sigma=0.,radius_method='extrema',percentiles=[.5,99.5])
    saved['target']['thickness_method']='splice'
    settings=target_settings(args,a,t.edges)
    restored=settings_from_generation(saved)
    check_settings(settings,restored)
    assert restored['resized'] and restored['thickness_method']=='splice'
    changed=dict(restored,splice_window=3.)
    with pytest.raises(MastercapError,match='splice window'):check_settings(settings,changed)
