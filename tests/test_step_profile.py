"""Analytic step integrals, moment fits and shared workflow targets."""
import json
import numpy as np
import pytest
from scipy.integrate import quad
from mastercap.step import integral, shape, fit_step, step_profiles, validate_bounds
from mastercap.analysis import analyze, build_target, shell_volumes, target_profiles
from mastercap.model import MastercapError


@pytest.mark.parametrize('a,b,w', [(90,140,0),(90,140,4),(3,8,2),(100,104,4)])
def test_closed_form(a,b,w):
    for k in range(4):
        points=sorted(set([a-w/2,a+w/2,b-w/2,b+w/2]))
        value=sum(quad(lambda r: float(shape(r,a,b,w))*r**(2+k),l,u,epsabs=1e-12,epsrel=1e-13)[0]
                  for l,u in zip(points[:-1],points[1:]))
        assert integral(a,b,w,k)==pytest.approx(value,rel=1e-12)
    if w==0:
        assert integral(a,b,0,1)/integral(a,b,0)==pytest.approx(.75*(b**4-a**4)/(b**3-a**3))
        assert integral(a,b,0,2)/integral(a,b,0)==pytest.approx(.6*(b**5-a**5)/(b**3-a**3))


def sample_step(a,b,w,n=200_000):
    rng=np.random.default_rng(42)
    values=[]
    while sum(map(len,values))<n:
        r=rng.uniform(a-w/2,b+w/2,n*2)
        values.append(r[rng.random(len(r))<shape(r,a,b,w)*(r/(b+w/2))**2])
    return np.concatenate(values)[:n]


@pytest.mark.parametrize('w',[0,4])
def test_fit_recovery_and_exact_moments(w):
    r=sample_step(90,140,w)
    fit=fit_step(r,edge_width=w)
    assert fit.inner==pytest.approx(90,abs=.1)
    assert fit.outer==pytest.approx(140,abs=.1)
    np.testing.assert_allclose(list(fit.target_moments.values()),list(fit.model_moments.values()),rtol=1e-10)


@pytest.mark.parametrize('width',[1,4,3.7])
def test_count_conservation_and_constant_composition(width):
    counts={'C':10000,'S':50}
    edges,profiles,densities=step_profiles((90,140,4),counts,width)
    for e,n in counts.items():
        assert np.sum(profiles[e]*shell_volumes(edges))==pytest.approx(n,rel=1e-12)
        np.testing.assert_allclose(profiles[e]/sum(profiles.values()),n/sum(counts.values()),rtol=1e-12)
    assert densities['Total']==sum(densities[e] for e in counts)


@pytest.mark.parametrize('bounds',[(1,8,4),(3,5,4),(3,3,0)])
def test_invalid_bounds(bounds):
    with pytest.raises(MastercapError,match='Step bounds.*invalid.*need'):
        validate_bounds(*bounds)


def test_unrepresentable_fit_raises():
    for r in [np.ones(30)*100, np.array([99.9,100.1]),np.array([0.,1.])]:
        with pytest.raises(MastercapError,match='Step fit'):
            fit_step(r,edge_width=4)


def test_resize_and_default_reference_target(small_structure):
    # Wide enough for an exact hard fit; reference branch remains byte-identical.
    small_structure.xyz[1,0]=-1.6
    small_structure.xyz[2,0]=1.6
    analysis=analyze(small_structure,sigma=0)
    baseline=build_target(analysis)
    edges,profiles=target_profiles(analysis)
    np.testing.assert_array_equal(baseline.edges,edges)
    for e in profiles:np.testing.assert_array_equal(baseline.profiles[e],profiles[e])
    base=build_target(analysis,step=True,edge_width=0)
    resized=build_target(analysis,base.step['fit']['inner']/2,base.step['fit']['outer']/2,step=True,edge_width=0)
    assert resized.reference_on_edges is None
    assert resized.step['counts']['target']['C']==pytest.approx(.5)
    assert resized.step['density']['plateau_total']==pytest.approx(base.step['density']['plateau_total'])


@pytest.fixture
def step_pdb(tmp_path,small_structure):
    from mastercap.pdbio import pdb_text
    small_structure.xyz[1,0]=-1.6
    small_structure.xyz[2,0]=1.6
    path=tmp_path/'step_reference.pdb'
    path.write_text(pdb_text(small_structure)[0])
    return path


@pytest.mark.parametrize('options,message',[
    (['--step-edge-width','4'],'requires --step-profile'),
    (['--step-profile','--smoothing-sigma','2'],'does not apply'),
    (['--step-profile','--no-smoothing','--smoothing-sigma','2'],'does not apply'),
    (['--step-profile','--sulfur-residue-placement','reference'],'incompatible'),
])
def test_cli_interactions_before_input(options,message,capsys):
    from mastercap.cli import main
    assert main(['generate','--input','missing','--output-prefix','unused']+options)==1
    output=capsys.readouterr().err
    assert message in output and 'Reading reference' not in output


def test_calibration_basis_and_defaults(capsys):
    from mastercap.cli import main,parser,resolve_target_options
    assert main(['calibrate','--input','missing','--step-profile','--floor-basis','orientation'])==1
    assert 'orientation floors are not defined' in capsys.readouterr().err
    for step in (False,True):
        args=parser().parse_args(['calibrate','--input','missing']+(['--step-profile'] if step else []))
        resolve_target_options(args)
        assert args.floor_basis==('counting' if step else 'orientation')
        assert args.smoothing_sigma==(0 if step else 1)


def test_settings_model_mismatch_and_legacy():
    from mastercap.calibration import check_settings
    settings={'reference_hash':'x','model':1,'bin_width':4,'smoothing_sigma':0,
              'inner_radius':90,'outer_radius':140,'edges_count':14,'radius_method':'extrema','radius_percentiles':[.5,99.5]}
    check_settings(settings,{**settings,'profile_model':'reference','step_edge_width':None})
    step={**settings,'profile_model':'step','step_edge_width':4}
    for left,right in [(settings,step),(step,settings)]:
        with pytest.raises(MastercapError,match='profile model'):
            check_settings(left,right)


def test_workflow_and_recomputed_target(step_pdb,tmp_path,capsys):
    from mastercap.cli import main
    common=['--input',str(step_pdb),'--step-profile','--step-edge-width','0','--bin-width','.4']
    assert main(['analyze',*common,'--output-prefix',str(tmp_path/'fit'),'--csv','--plot'])==0
    fit=json.loads((tmp_path/'fit.json').read_text())
    assert fit['step']['moments']=='atom'
    assert main(['calibrate',*common,'--output-prefix',str(tmp_path/'calib'),'--replicates','3',
                 '--generated',str(step_pdb),'--compare',str(step_pdb),'--csv','--plot'])==0
    path=tmp_path/'calib_calibration.json'
    calib=json.loads(path.read_text())['calibration']
    assert calib['levels']['orientation']=={'skipped':'not defined for step targets'}
    assert calib['levels']['placement']=={'skipped':'not defined for step targets'}
    assert calib['suggested']['basis']=='counting' and not calib['suggested']['compare_used']
    assert set(calib['suggested']['rule_by_element'].values())=={'floor'}
    assert calib['generated'][0]['multiples_orientation'] is None
    assert calib['generated'][0]['gap_closed'] is None
    assert calib['generated'][0]['undefined_gap_reasons']['C'].startswith('step target')
    import csv
    rows=list(csv.DictReader((tmp_path/'calib_calibration.csv').open()))
    assert {r['source'] for r in rows}=={'self','counting','compare','generated'}
    assert next(r for r in rows if r['source']=='self')['label']=='reference vs step'
    code=main(['generate',*common,'--output-prefix',str(tmp_path/'particle'),'--seed','42','--refine-updates','0',
               '--tolerances-from',str(path),'--allow-approximate','--csv','--plot'])
    assert code in (0,2),capsys.readouterr().err
    report=tmp_path/'particle.json'
    saved=json.loads(report.read_text())
    assert 'original_profile_error' in saved['metrics']['elements']['C']
    validate=['validate','--input',str(step_pdb),'--pdb',str(tmp_path/'particle.pdb'),'--report',str(report)]
    assert main([*validate,'--output-prefix',str(tmp_path/'checked'),'--csv','--plot'])==code
    checked=json.loads((tmp_path/'checked.json').read_text())
    assert checked['target']['step']==saved['target']['step']
    saved['analysis_options']['step_edge_width']=.1
    report.write_text(json.dumps(saved))
    assert main([*validate,'--output-prefix',str(tmp_path/'tampered')])==1
    assert 'Recomputed step target differs' in capsys.readouterr().err


def test_resized_outputs_without_original(step_pdb,tmp_path):
    from mastercap.cli import main
    # Slight enlargement retains enough atoms for the tiny packing fixture.
    args=['generate','--input',str(step_pdb),'--output-prefix',str(tmp_path/'resized'),
          '--step-profile','--step-edge-width','0','--inner-radius','1.2','--outer-radius','2.8',
          '--bin-width','.4','--seed','42','--refine-updates','0','--allow-approximate','--csv','--plot']
    assert main(args)==2
    report=json.loads((tmp_path/'resized.json').read_text())
    assert report['target']['step']['generated_bounds']['resized']
    assert 'original_profile_error' not in report['metrics']['elements']['C']
    import csv
    rows=list(csv.DictReader((tmp_path/'resized_profiles.csv').open()))
    assert all(row['original_atoms_per_A3']=='' for row in rows)
    from matplotlib.image import imread
    assert imread(tmp_path/'resized_profiles.png').size>0


def test_api_forbids_reference_sulfur_for_step(small_structure):
    from mastercap.generate import search
    analysis=analyze(small_structure)
    with pytest.raises(MastercapError,match='incompatible'):
        search(small_structure,analysis,analysis.edges,analysis.original,analysis.center,np.random.default_rng(42),
               target_model='step',sulfur_residue_placement='reference')


def test_step_target_always_plotted_and_missing_original_is_omitted(monkeypatch):
    from matplotlib.axes import Axes
    from mastercap.reports import profile_png
    labels=[]
    plot=Axes.plot
    def capture(self,*args,**kwargs):
        labels.append(kwargs.get('label'))
        return plot(self,*args,**kwargs)
    monkeypatch.setattr(Axes,'plot',capture)
    edges,profiles,_=step_profiles((90,140,4),{'C':100},3.7)
    assert profile_png(edges,None,profiles,smoothing_enabled=False,target_label='Step target')
    assert labels.count('Step target')==2
    assert 'Smoothed target' not in labels and 'Original (mapped)' not in labels


def test_step_counting_uses_resized_material(step_pdb):
    from mastercap.pdbio import read_pdb
    from mastercap.calibration import calibrate,profile_settings
    from mastercap.reports import calibration_csv
    import csv,io
    analysis=analyze(read_pdb(step_pdb),sigma=0)
    target=build_target(analysis,1,3,step=True,edge_width=0,bin_width=.4)
    settings=profile_settings(analysis,target.edges,.4,model='step',step_edge_width=0)
    result=calibrate(analysis,target.edges,target.profiles,settings,replicates=5,floor_basis='counting')
    rows=list(csv.DictReader(io.StringIO(calibration_csv(result).decode())))
    count=next(r for r in rows if r['source']=='counting' and r['element']=='C')
    assert int(count['n_atoms'])==round(result['target_counts']['C'])
    assert int(count['n_atoms'])!=result['element_counts']['C']


def test_legacy_calibration_loads_and_models_reject_before_search(step_pdb,tmp_path,monkeypatch,capsys):
    import mastercap.cli as cli
    assert cli.main(['calibrate','--input',str(step_pdb),'--no-smoothing','--output-prefix',str(tmp_path/'reference'),
                     '--replicates','2'])==0
    path=tmp_path/'reference_calibration.json'
    saved=json.loads(path.read_text())
    for key in ('profile_model','step_edge_width'):saved['calibration']['settings'].pop(key)
    path.write_text(json.dumps(saved))
    common=['generate','--input',str(step_pdb),'--output-prefix',str(tmp_path/'out'),'--no-smoothing',
            '--tolerances-from',str(path),'--allow-approximate','--refine-updates','0','--seed','42']
    assert cli.main(common) in (0,2)
    monkeypatch.setattr(cli,'search',lambda *a,**k:pytest.fail('mismatched target must not start search'))
    assert cli.main([*common,'--output-prefix',str(tmp_path/'mismatch'),'--step-profile','--step-edge-width','0'])==1
    assert 'profile model' in capsys.readouterr().err
