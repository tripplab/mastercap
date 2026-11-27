"""Shared curvature shifts, pairing audits and legacy compatibility (synthetic only)."""
import copy
import numpy as np
import pytest

from test_revision12 import synthetic, options, saved_target
from mastercap.analysis import resolve_geometry, build_target
from mastercap.cli import configured_target, rebuild_target, verify_geometry_scaling, target_settings
from mastercap.calibration import check_settings, settings_from_generation
from mastercap.model import MastercapError
from mastercap.radial import shell_volumes
from mastercap.step import StepFit


def target(a,f,g,step=False,rule='shared_radial_shift'):
    return configured_target(options(curvature_factor=f,thickness_factor=g,step_profile=step,curvature_rule=rule),a)


@pytest.mark.parametrize('f',[.5,2/3,1.,1.5,2.])
@pytest.mark.parametrize('g',[.8,1.,1.25])
def test_step_bounds_and_targets_bit_identical_to_v017(synthetic,f,g):
    a,_=synthetic
    shared=target(a,f,g,True);legacy=target(a,f,g,True,'per_arm_outer')
    assert shared.geometry_scaling['generated']==legacy.geometry_scaling['generated']
    assert shared.step==legacy.step
    np.testing.assert_array_equal(shared.edges,legacy.edges)
    for e in shared.profiles:np.testing.assert_array_equal(shared.profiles[e],legacy.profiles[e])


@pytest.mark.parametrize('f',[.5,2/3,1.,1.5,2.])
def test_layered_and_step_share_radial_shift_and_paired_bounds(synthetic,f):
    a,fit=synthetic
    blocks=[target(a,f,.8,step).geometry_scaling for step in (False,True)]
    shift=fit.outer/f-fit.outer
    assert blocks[0]['generated']['outer']==pytest.approx(a.outer+shift,abs=1e-12)
    assert blocks[0]['radial_shift_angstrom']==pytest.approx(blocks[1]['radial_shift_angstrom'],abs=1e-12)
    assert blocks[0]['paired_bounds']==blocks[1]['paired_bounds']
    for b in blocks:
        assert b['paired_bounds'][b['arm']]=={k:b['generated'][k] for k in ('inner','outer')}
        assert b['curvature_rule']=='shared_radial_shift' and b['curvature_basis']=='step_fit_outer'
        assert b['r_step_outer_angstrom']==fit.outer
        assert b['outer_radius_ratio']==b['arm_outer_radius_ratio']
        assert b['arm_outer_radius_ratio']==b['reference']['outer']/b['generated']['outer']


@pytest.mark.parametrize('g',[.8,1.,1.25])
def test_unit_curvature_is_identical_to_legacy(synthetic,g):
    a,_=synthetic
    shared=target(a,1.,g);legacy=target(a,1.,g,rule='per_arm_outer')
    np.testing.assert_array_equal(shared.edges,legacy.edges)
    for e in shared.profiles:np.testing.assert_array_equal(shared.profiles[e],legacy.profiles[e])


def test_shift_only_and_shift_plus_splice(synthetic):
    a,fit=synthetic;shift=fit.outer/1.5-fit.outer
    shifted=target(a,1.5,1.)
    np.testing.assert_allclose(shifted.edges,a.edges+shift,rtol=0,atol=1e-9)
    for e in a.smoothed:np.testing.assert_array_equal(shifted.profiles[e],a.smoothed[e])
    plain=target(a,1.,.8);spliced=target(a,1.5,.8)
    np.testing.assert_allclose(spliced.edges,plain.edges+shift,rtol=0,atol=1e-9)
    for e in plain.profiles:np.testing.assert_array_equal(spliced.profiles[e],plain.profiles[e])


def test_target_pairing_mean_radius_audit(synthetic):
    a,_=synthetic
    # Match the exact reference step bins, including ramps. The layered outer
    # is the support edge (157); the fitted step outer is half height (155).
    base=build_target(a,step=True,bin_width=1.)
    a.edges=base.edges.copy();a.original=copy.deepcopy(base.profiles);a.smoothed=copy.deepcopy(base.profiles)
    pairs={rule:[target(a,1.5,1.,step,rule) for step in (False,True)]
           for rule in ('shared_radial_shift','per_arm_outer')}
    def mean(t):return t.geometry_scaling['target_mean_radius_angstrom']
    assert abs(mean(pairs['shared_radial_shift'][0])-mean(pairs['shared_radial_shift'][1]))<.05
    assert abs(mean(pairs['per_arm_outer'][0])-mean(pairs['per_arm_outer'][1]))>.5
    for pair in pairs.values():
        for t in pair:
            counts=sum(t.profiles.values())*shell_volumes(t.edges)
            assert t.geometry_scaling['target_expected_atoms']==float(counts.sum())
            assert mean(t)==float(np.sum((t.edges[1:]+t.edges[:-1])/2*counts)/counts.sum())


def test_invalid_partner_is_informational_and_selected_layered_refuses(synthetic,monkeypatch):
    a,_=synthetic
    a.edges=np.arange(80.,161.)
    a.original={e:np.full(80,p[0]) for e,p in a.original.items()};a.smoothed=copy.deepcopy(a.original)
    monkeypatch.setattr('mastercap.step.fit_step',lambda *args,**kw:StepFit(110.,130.,4.,20.,108.,132.,{}, {},[0.,0.],1))
    with pytest.raises(MastercapError,match=r'shift=.*R_step=.*analysis.outer='):
        target(a,4.,1.)
    t=target(a,4.,1.,True)
    assert 'error' in t.geometry_scaling['paired_bounds']['layered']
    assert t.geometry_scaling['paired_bounds']['step']=={'inner':12.5,'outer':32.5}


def test_invalid_step_partner_does_not_refuse_layered(synthetic):
    a,_=synthetic
    # Broaden layered bounds inward while retaining a small valid thickness.
    # With F=4 and G=1 step's inner is negative, but a thin layered arm fits.
    a.edges=np.arange(140.,161.)
    a.original={e:np.full(20,p[0]) for e,p in a.original.items()};a.smoothed=copy.deepcopy(a.original)
    t=target(a,4.,1.)
    assert 'error' in t.geometry_scaling['paired_bounds']['step']


@pytest.mark.parametrize('step',[False,True])
def test_new_report_rebuild_and_legacy_v017_report(synthetic,step,caplog):
    a,_=synthetic
    args=options(curvature_factor=1.5,thickness_factor=.8,step_profile=step)
    t=configured_target(args,a);saved=saved_target(a,args,t)
    rebuilt=rebuild_target(a,saved);verify_geometry_scaling(rebuilt,saved)
    np.testing.assert_array_equal(rebuilt.edges,t.edges)
    for key,value in [('radial_shift_angstrom',.5),('r_step_outer_angstrom',.5),('curvature_rule','invalid')]:
        bad=copy.deepcopy(saved)
        bad['geometry_scaling'][key]=(bad['geometry_scaling'][key]+value if key!='curvature_rule' else value)
        with pytest.raises(MastercapError,match=f'Recomputed geometry differs from geometry_scaling: {key}'):
            verify_geometry_scaling(rebuilt,bad)
    # Derived fields are audit information, not new validation constraints.
    derived=copy.deepcopy(saved)
    derived['geometry_scaling'].update(paired_bounds={},target_mean_radius_angstrom=-1,target_expected_atoms=-1)
    verify_geometry_scaling(rebuilt,derived)
    old_args=options(curvature_factor=1.5,thickness_factor=.8,step_profile=step,curvature_rule='per_arm_outer')
    old=target(a,1.5,.8,step,'per_arm_outer');old_saved=saved_target(a,old_args,old)
    old_saved['configuration'].update(inner_radius=old.geometry_scaling['generated']['inner'],outer_radius=old.geometry_scaling['generated']['outer'])
    old_saved['geometry_scaling'].pop('curvature_rule');old_saved['configuration'].pop('curvature_rule',None)
    caplog.clear()
    rebuilt=rebuild_target(a,old_saved);verify_geometry_scaling(rebuilt,old_saved)
    np.testing.assert_array_equal(rebuilt.edges,old.edges)
    assert len([r for r in caplog.records if 'Legacy v0.1.17 report' in r.message])==(0 if step else 1)


def test_calibration_curvature_rule_and_legacy_equivalence(synthetic):
    a,_=synthetic
    for step in (False,True):
        args=options(curvature_factor=1.5,step_profile=step)
        t=configured_target(args,a)
        settings=target_settings(args,a,t.edges)
        assert settings['curvature_rule']=='shared_radial_shift'
        old=dict(settings,curvature_rule='per_arm_outer')
        if step:check_settings(old,settings)
        else:
            with pytest.raises(MastercapError,match='curvature rule'):check_settings(old,settings)
        old.pop('curvature_rule')
        if step:check_settings(old,settings)
        else:
            with pytest.raises(MastercapError,match='curvature rule'):check_settings(old,settings)
        saved=saved_target(a,args,t)
        saved.update(analysis=a.summary())
        saved['analysis_options'].update(sigma=0.,radius_method='extrema',percentiles=[.5,99.5])
        assert settings_from_generation(saved)['curvature_rule']=='shared_radial_shift'
        saved['geometry_scaling'].pop('curvature_rule')
        assert settings_from_generation(saved)['curvature_rule']=='per_arm_outer'
    # With F=1 the rule label does not affect targets/settings compatibility.
    check_settings(dict(settings,curvature_factor=1,curvature_rule='per_arm_outer'),dict(settings,curvature_factor=1))


def test_invalid_curvature_rule_even_without_factors(synthetic):
    with pytest.raises(MastercapError,match='Invalid curvature_rule'):
        resolve_geometry(synthetic[0],step=False,curvature_rule='invalid')
