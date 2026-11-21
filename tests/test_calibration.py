import json
from types import SimpleNamespace
import numpy as np
import pytest
from mastercap.analysis import analyze,target_profiles,shell_volumes
from mastercap.calibration import (profile_errors,residue_frame,rotated_coordinates,resampling_levels,
                                  statistics,suggest,round_up_three,comparison,calibrate,profile_settings)
from mastercap.constraints import profile_metrics
from mastercap.model import Atom,Structure
from mastercap.cli import main


def test_profile_errors_match_metrics_and_self_bias(small_structure):
    for sigma in (0,.8):
        analysis=analyze(small_structure,width=.4,sigma=sigma)
        edges,target=target_profiles(analysis)
        radius=np.linalg.norm(small_structure.xyz-analysis.center,axis=1)
        errors=profile_errors({'C':radius},edges,target)
        metrics=profile_metrics(small_structure,analysis.center,edges,target)
        assert errors['C']==metrics['elements']['C']['profile_error']
        assert errors['Total']==metrics['total_profile_error']
        assert errors['C']==0 if sigma==0 else errors['C']>0


def test_identity_rotates_about_own_centroid(small_structure):
    codes,centers,offsets=residue_frame(small_structure)
    xyz=rotated_coordinates(codes,centers,offsets,np.repeat(np.eye(3)[None],len(centers),axis=0))
    assert np.allclose(xyz,small_structure.xyz)
    rotation=np.repeat(np.array([[0,-1.,0],[1.,0,0],[0,0,1]])[None],len(centers),axis=0)
    moved=rotated_coordinates(codes,centers,offsets,rotation)
    for k,center in enumerate(centers):
        assert np.allclose(moved[codes==k].mean(axis=0),center)
    a=analyze(small_structure,width=.4,sigma=0)
    assert profile_errors({'C':np.linalg.norm(xyz-a.center,axis=1)},a.edges,a.smoothed)['C']==0


def test_counting_matches_multinomial_normal_approximation():
    from mastercap.calibration import errors_from_counts
    rng=np.random.default_rng(42);n=100000;k=10;edges=np.arange(1,k+2.)
    target={'C':np.full(k,n/k)/shell_volumes(edges)}
    values=[errors_from_counts({'C':rng.multinomial(n,np.full(k,1/k))},edges,target)['C'] for _ in range(1500)]
    expected=k*np.sqrt(2*n*(1/k)*(1-1/k)/np.pi)/n
    assert np.mean(values)==pytest.approx(expected,rel=.03)


def test_seed_reproducibility(small_structure):
    a=analyze(small_structure,width=.4,sigma=0)
    x=resampling_levels(a,a.edges,a.smoothed,20,42)
    assert x==resampling_levels(a,a.edges,a.smoothed,20,42)
    assert x!=resampling_levels(a,a.edges,a.smoothed,20,7)
    assert all(np.isfinite(v) for level in x.values() for stats in level.values() for v in stats.values())


def test_missing_comparison_element_and_count_normalized_shape():
    s=Structure([Atom('C','C','ALA',0)],np.array([[1.5,0,0]]))
    edges=np.array([1.,2.,3.]);target={e:np.array([2.,0])/shell_volumes(edges) for e in ('C','S')}
    with pytest.raises(ValueError,match='missing element S'):
        comparison(s,np.zeros(3),edges,target)
    target={'C':target['C']}
    report=comparison(s,np.zeros(3),edges,target)
    assert report['count_ratio']['C']==.5 and report['raw']['C']==.5 and report['shape']['C']==0


def test_suggestions_round_up_and_real_variation_sets_rule():
    levels={'orientation':{'C':statistics([.012345]),'Total':statistics([.012345])}}
    result=suggest(levels,[],multiple=2)
    assert result['profile_tolerance']['C']==.0247
    assert result['rule_by_element']['C']=='floor'
    result=suggest(levels,[{'shape':{'C':.071111}}])
    assert result['profile_tolerance']['C']==.0712 and result['rule_by_element']['C']=='real variation'
    for value in (.000123456,.018001,999.999,0):
        rounded,_=round_up_three(value);assert rounded>=value


def test_generated_reference_and_mismatched_sibling(tmp_path,small_pdb,small_structure):
    a=analyze(small_structure,width=.4,sigma=0)
    # The reader provenance supplies the actual source identity for settings.
    from mastercap.pdbio import read_pdb
    a=analyze(read_pdb(small_pdb),width=.4,sigma=0,compute_distances=False)
    edges,target=target_profiles(a);settings=profile_settings(a,edges,.4)
    report=calibrate(a,edges,target,settings,replicates=10,generated=[small_pdb])
    row=report['generated'][0]
    assert row['errors']['C']==0
    assert row['multiples_orientation']['C']==0
    assert row['gap_closed']['C'] is None  # Equal centroid radii need not create a placement gap.
    from mastercap.calibration import generated_assessment
    levels={level:{e:statistics([value]) for e in ('C','Total')} for level,value in [('counting',.1),('orientation',.2),('placement',.5)]}
    row=generated_assessment(small_pdb,a,edges,target,settings,levels)
    assert row['gap_closed']['C']>1
    assert row['gap_closed_clipped']['C']==1
    sibling=small_pdb.with_suffix('.json')
    sibling.write_text(json.dumps({'analysis_options':{'width':4,'sigma':0,'radius_method':'extrema','percentiles':[.5,99.5]},
      'configuration':{},'analysis':{'reference':a.structure.provenance},'target':{'edges':edges.tolist(),'center':[0,0,0]}}))
    with pytest.raises(ValueError,match='Calibration settings differ: bin width'):
        calibrate(a,edges,target,settings,replicates=2,generated=[small_pdb])


def test_cli_never_generates_and_protects_generation_prefix(tmp_path,small_pdb,monkeypatch):
    import mastercap.cli as cli
    monkeypatch.setattr(cli,'search',lambda *a,**k:pytest.fail('calibrate cannot generate'))
    prefix=tmp_path/'same'
    prefix.with_suffix('.json').write_text(json.dumps({'command':'generate'}))
    assert main(['calibrate','--input',str(small_pdb),'--output-prefix',str(prefix),'--overwrite'])==1
    prefix.with_suffix('.json').write_text('[]')
    assert main(['calibrate','--input',str(small_pdb),'--output-prefix',str(prefix),'--replicates','2'])==0
    assert prefix.with_suffix('.json').read_text()=='[]'
    out=tmp_path/'calib'
    assert main(['calibrate','--input',str(small_pdb),'--output-prefix',str(out),'--replicates','2'])==0
    assert sorted(p.name for p in tmp_path.glob('calib*'))==['calib_calibration.json']


def test_levels_order_on_element_correlated_shells():
    k=np.arange(400);z=1-2*(k+.5)/len(k);phi=k*np.pi*(3-np.sqrt(5))
    directions=np.column_stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z])
    atoms=[];xyz=[]
    for e,radius in [('C',10.),('N',20.)]:
        for d in directions:
            residue=len(atoms)//2
            for delta in (-.9,.9):
                atoms.append(Atom(e,e,'ALA',residue));xyz.append((radius+delta)*d)
    a=analyze(Structure(atoms,np.array(xyz)),width=.25,sigma=0,compute_distances=False)
    levels=resampling_levels(a,a.edges,a.smoothed,20,42)
    for e in ('C','N'):
        assert levels['counting'][e]['mean']<levels['orientation'][e]['mean']<levels['placement'][e]['mean']


@pytest.mark.parametrize('key,value,label',[
 ('reference_hash','other','reference hash'),('model',2,'model'),('bin_width',4,'bin width'),
 ('smoothing_sigma',2,'smoothing sigma'),('inner_radius',2,'inner radius'),('outer_radius',5,'outer radius')])
def test_settings_cannot_cross_reference_or_profile_definitions(key,value,label):
    from mastercap.calibration import check_settings
    settings={'reference_hash':'abc','model':1,'bin_width':1.,'smoothing_sigma':0.,'inner_radius':1.,'outer_radius':4.,
              'radius_method':'extrema','radius_percentiles':[.5,99.5],'edges_count':4}
    changed={**settings,key:value}
    with pytest.raises(ValueError,match='Calibration settings differ: '+label):check_settings(settings,changed)


def test_chain_label_profiles_are_scaled_to_full_counts():
    from mastercap.calibration import chain_spread
    xyz=np.array([[r*sign,0,0] for r in (1.,3.) for sign in (-1,1) for _ in range(500)])
    atoms=[Atom('C','C','ALA',i,{'chain':'A' if i<1000 else 'B'}) for i in range(2000)]
    a=analyze(Structure(atoms,xyz),width=1,sigma=0,compute_distances=False)
    spread=chain_spread(a,a.edges,a.smoothed)
    assert spread['available'] and not spread['used_for_suggestions']
    assert spread['pairwise'][0]['errors']['C']==pytest.approx(2)
    assert spread['labels']['A']['shape']['C']==pytest.approx(1)
    a.structure.atoms=[Atom(atom.name,atom.element,atom.resname,atom.residue) for atom in atoms]
    assert not chain_spread(a,a.edges,a.smoothed)['available']
