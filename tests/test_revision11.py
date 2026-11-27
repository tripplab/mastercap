"""Small geometry/provenance unit tests; no size pilot or benchmark runs."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mastercap.analysis import analyze, build_target, resolve_geometry
from mastercap.atomdata import encode_atom_data, load_atom_data
from mastercap.calibration import check_settings, settings_from_generation
from mastercap.cli import main, parser
from mastercap.model import Atom, Structure, MastercapError
from mastercap.pdbio import pdb_text
from mastercap.radial import shell_volumes
from mastercap.step import fit_step, integral
from mastercap.splice import SplicePlan


def reference(width=.5):
    r=np.linspace(10.,14.,12)
    xyz=np.vstack([np.c_[r,np.zeros((len(r),2))],np.c_[-r,np.zeros((len(r),2))]])
    s=Structure([Atom('CA','C','GLY',i//2) for i in range(len(xyz))],xyz)
    return analyze(s,width=width,sigma=0,compute_distances=False)


@pytest.mark.parametrize('step',[False,True])
def test_resolve_geometry_formulas_unity_and_no_factors(step):
    a=reference();width=.5
    original=fit_step(np.linalg.norm(a.structure.xyz-a.center,axis=1),edge_width=width) if step else a
    r0=original.outer;t0=original.outer-original.inner
    inner,outer,block=resolve_geometry(a,step=step,edge_width=width,curvature_factor=1.5,thickness_factor=.8,splice_wall_keep=.01,splice_window=.1,splice_max_mismatch=1)
    assert outer==pytest.approx(r0/1.5,abs=1e-12)
    delta=-round(.2*(fit_step(np.linalg.norm(a.structure.xyz-a.center,axis=1),edge_width=width).thickness)/.5)*.5
    assert outer-inner==pytest.approx(t0+delta,abs=1e-12)
    assert inner==pytest.approx(r0/1.5-t0-delta,abs=1e-12)
    assert block['basis']=='outer_radius' and block['outer_radius_ratio']==pytest.approx(1.5)
    assert block['mid_radius_ratio']==pytest.approx((original.inner+original.outer)/(inner+outer))
    if not step:
        assert 'layered_bin_width_angstrom' not in block
        assert block['delta_bins']==round(delta/.5)
    else:np.testing.assert_allclose(block['step_support'],[inner-width/2,outer+width/2])
    x,y,_=resolve_geometry(a,step=step,edge_width=width,curvature_factor=1.,thickness_factor=1.)
    np.testing.assert_allclose([x,y],[original.inner,original.outer],rtol=0,atol=1e-12)
    assert resolve_geometry(a,step=step,edge_width=width)==(None,None,None)


@pytest.mark.parametrize('f,g',[(4.,4.),(.1,1.),(1.,4.1),(np.inf,1.)])
@pytest.mark.parametrize('step',[False,True])
def test_invalid_geometry_factors_and_resulting_bounds(step,f,g):
    with pytest.raises(MastercapError):
        resolve_geometry(reference(),step=step,edge_width=.5,curvature_factor=f,thickness_factor=g)


def test_step_too_thin_and_layered_less_than_two_nominal_bins():
    with pytest.raises(MastercapError,match='f=1, G=0.25.*Step bounds'):
        resolve_geometry(reference(),step=True,edge_width=2.,curvature_factor=1.,thickness_factor=.25)
    with pytest.raises(MastercapError,match='nominal bin width'):
        resolve_geometry(reference(.75),step=False,edge_width=0,thickness_factor=.25)


def test_target_integrated_count_scaling_for_both_arms():
    a=reference()
    for step in (False,True):
        base=build_target(a,step=step,edge_width=.5,bin_width=.5)
        inner,outer,block=resolve_geometry(a,step=step,edge_width=.5,curvature_factor=1.5,thickness_factor=.8,splice_wall_keep=.01,splice_window=.1,splice_max_mismatch=1)
        plan=SplicePlan.from_report(block['splice'],a) if not step else None
        target=build_target(a,inner,outer,step=step,edge_width=.5,bin_width=.5,splice=plan)
        count=float(np.sum(target.profiles['C']*shell_volumes(target.edges)))
        if step:
            fit=base.step['fit']
            expected=len(a.structure.atoms)*integral(inner,outer,.5)/integral(fit['inner'],fit['outer'],.5)
        else:
            expected=float(np.sum(target.reference_on_edges['C']*shell_volumes(target.edges)))
            np.testing.assert_allclose(np.diff(target.edges),.5,rtol=1e-12)
        assert count==pytest.approx(expected,rel=1e-12)


@pytest.mark.parametrize('flags',[
    ['--curvature-factor','1','--inner-radius','1'],['--thickness-factor','1','--outer-radius','3'],
    ['--curvature-factor','.249'],['--thickness-factor','4.01']])
@pytest.mark.parametrize('command',['generate','calibrate'])
def test_factor_cli_rejects_exclusivity_and_range_before_input(command,flags,tmp_path,capsys):
    assert main([command,'--input','missing.pdb','--output-prefix',str(tmp_path/'unused'),*flags])==1
    text=capsys.readouterr().err
    assert 'mutually exclusive' in text or 'outside the supported range' in text
    assert 'Reading reference' not in text


@pytest.fixture
def atom_reports(tmp_path,small_pdb):
    reports={}
    for mode in ('inline','sidecar','omit'):
        prefix=tmp_path/mode
        assert main(['generate','--input',str(small_pdb),'--output-prefix',str(prefix),'--seed','42',
                     '--no-smoothing','--refine-updates','0','--report-atom-data',mode,'--allow-approximate'])==2
        path=prefix.with_suffix('.json');reports[mode]=(path,json.loads(path.read_text()))
    return small_pdb,reports


def validate_input(reference,path,prefix,*flags):
    return main(['validate','--input',str(reference),'--pdb',str(path.with_suffix('.pdb')),
                 '--report',str(path),'--output-prefix',str(prefix),*flags])


def test_sidecar_roundtrip_inline_content_and_no_pickle(atom_reports):
    _,reports=atom_reports
    inline_path,inline=reports['inline'];path,sidecar=reports['sidecar']
    for key in ('atom_mapping','residue_transforms'):assert key in inline and key not in sidecar
    assert 'initial_transforms' in inline['search'] and 'initial_transforms' not in sidecar['search']
    data=load_atom_data(sidecar,path)
    assert data.mapping_records()==inline['atom_mapping']
    assert data.transforms==inline['residue_transforms']
    assert data.initial==inline['search']['initial_transforms']
    assert inline_path.with_suffix('.pdb').read_bytes()==path.with_suffix('.pdb').read_bytes()
    npz=path.parent/sidecar['atom_data']['path']
    assert sidecar['atom_data']['sha256']==hashlib.sha256(npz.read_bytes()).hexdigest()
    with np.load(npz,allow_pickle=False) as arrays:
        assert arrays['format_version'].shape==() and int(arrays['format_version'])==1
        for key in ('reference_atom_index','synthetic_residue','id'):assert arrays[key].dtype==np.dtype('int32')
        assert arrays['rotation_row_vector'].shape==(2,3,3)
        assert arrays['translation'].dtype==np.dtype('float64')
        assert all(arrays[key].dtype.kind!='O' for key in arrays.files)
    assert sidecar['atom_data']['residue_transforms_summary']['max_rmsd']>=0


def test_optional_initial_fields_and_nullable_origin_roundtrip(tmp_path):
    mapping=[{'id':0,'synthetic_residue':0,'origin':{'reference_atom_index':1,'serial':None,'chain':'A'}}]
    initial=[{'rotation':np.eye(3).tolist(),'translation':[1.,2.,3.],
              'sulfur_placement':{'radius':10.2},'active':True}]
    payload=encode_atom_data(mapping,None,initial);path=tmp_path/'extra.npz';path.write_bytes(payload)
    data=load_atom_data({'atom_data':{'mode':'sidecar','path':path.name,'sha256':hashlib.sha256(payload).hexdigest(),'atoms':1}},tmp_path/'extra.json')
    assert data.mapping_records()==mapping and data.initial==initial and data.transforms is None


def test_sidecar_validation_and_validation_report_never_inlines(atom_reports,tmp_path):
    source,reports=atom_reports
    for mode,(path,saved) in reports.items():
        prefix=tmp_path/('checked_'+mode)
        assert validate_input(source,path,prefix)==2
        checked=json.loads(prefix.with_suffix('.json').read_text())
        assert 'atom_mapping' not in checked and 'residue_transforms' not in checked
        assert checked['atom_data']['mode']==mode
        assert checked['atom_data']['sha256_verified']==(mode=='sidecar')
        assert checked['geometry']['mandatory_pass']


def test_missing_sidecar_and_override_location(atom_reports,tmp_path,capsys):
    source,reports=atom_reports;path,saved=reports['sidecar']
    original=path.parent/saved['atom_data']['path'];relocated=tmp_path/'relocated.npz'
    original.rename(relocated)
    assert validate_input(source,path,tmp_path/'missing')==1
    text=capsys.readouterr().err
    assert str(original) in text and '--atom-data' in text
    assert validate_input(source,path,tmp_path/'found','--atom-data',str(relocated))==2
    assert json.loads((tmp_path/'found.json').read_text())['atom_data']['sha256_verified']


def test_sidecar_hash_and_reference_index_tampering(atom_reports,tmp_path,capsys):
    source,reports=atom_reports;path,saved=reports['sidecar'];npz=path.parent/saved['atom_data']['path']
    original=npz.read_bytes();npz.write_bytes(original+b'corrupt')
    assert validate_input(source,path,tmp_path/'hash')==1
    assert 'SHA-256 mismatch' in capsys.readouterr().err
    with np.load(io.BytesIO(original),allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    arrays['reference_atom_index'][0]=1000
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);npz.write_bytes(stream.getvalue())
    saved['atom_data']['sha256']=hashlib.sha256(stream.getvalue()).hexdigest();path.write_text(json.dumps(saved))
    assert validate_input(source,path,tmp_path/'index')==1
    assert 'reference_atom_index' in capsys.readouterr().err


def test_omit_signatures_warning_and_unavailable_rmsd(atom_reports,tmp_path,capsys):
    source,reports=atom_reports;path,saved=reports['omit']
    assert 'atom_mapping' not in saved and 'residue_transforms' not in saved
    assert 'initial_transforms' not in saved['search']
    assert not list(tmp_path.glob('omit*.npz'))
    capsys.readouterr()
    assert validate_input(source,path,tmp_path/'checked')==2
    assert 'provenance checked by residue signature only' in capsys.readouterr().err
    result=json.loads((tmp_path/'checked.json').read_text())
    assert result['diagnostics']['residue_drift']=={'median_angstrom':None,'p95_angstrom':None}
    pdb=path.with_suffix('.pdb');lines=pdb.read_text().splitlines()
    i=next(i for i,s in enumerate(lines) if s.startswith('ATOM'))
    lines[i]=lines[i][:12]+'ZZZZ'+lines[i][16:]
    pdb.write_text('\n'.join(lines)+'\n');saved['pdb_sha256']=hashlib.sha256(pdb.read_bytes()).hexdigest()
    path.write_text(json.dumps(saved))
    assert validate_input(source,path,tmp_path/'bad_signature')==1
    assert 'Residue signature' in capsys.readouterr().err


def test_old_inline_report_and_inline_priority(atom_reports,tmp_path):
    source,reports=atom_reports;path,saved=reports['inline']
    saved.pop('atom_data');saved['configuration'].pop('report_atom_data');saved['mastercap_version']='0.1.15'
    path.write_text(json.dumps(saved))
    assert validate_input(source,path,tmp_path/'old','--atom-data',str(tmp_path/'does_not_exist.npz'))==2
    assert json.loads((tmp_path/'old.json').read_text())['atom_data']['mode']=='inline'


def test_sidecar_overwrite_and_input_protection(tmp_path,small_pdb,capsys):
    prefix=tmp_path/'reserved';npz=tmp_path/'reserved.atomdata.npz';npz.write_bytes(b'keep')
    assert main(['generate','--input',str(small_pdb),'--output-prefix',str(prefix)])==1
    assert npz.read_bytes()==b'keep' and not prefix.with_suffix('.pdb').exists()
    assert main(['generate','--input',str(npz),'--output-prefix',str(prefix),'--overwrite'])==1
    assert 'replace an input file' in capsys.readouterr().err
    assert npz.read_bytes()==b'keep'


@pytest.mark.parametrize('step',[False,True])
def test_resolved_config_validation_and_tampered_scaling(tmp_path,small_structure,step,capsys,monkeypatch):
    # This four-atom CLI fixture cannot fit a 4 A ramp. Isolate its geometry
    # round-trip check from the separately tested reference fit.
    if not step:monkeypatch.setattr('mastercap.step.fit_step',lambda r,**kw:fit_step(r,edge_width=0))
    small_structure.xyz[1,0]=-1.6;small_structure.xyz[2,0]=1.6
    source=tmp_path/'ref.pdb';source.write_text(pdb_text(small_structure)[0]);prefix=tmp_path/'scaled'
    flags=['--step-profile','--step-edge-width','0'] if step else []
    assert main(['generate','--input',str(source),'--output-prefix',str(prefix),'--no-smoothing',
                 '--bin-width','.2','--curvature-factor','1','--thickness-factor','1','--refine-updates','0',
                 '--seed','42','--allow-approximate',*flags])==2
    path=prefix.with_suffix('.json');saved=json.loads(path.read_text())
    scaling=saved['geometry_scaling']['generated']
    assert saved['configuration']['inner_radius']==scaling['inner']
    assert saved['configuration']['outer_radius']==scaling['outer']
    assert validate_input(source,path,tmp_path/'validated')==2
    saved['geometry_scaling']['generated']['outer']+=.01;path.write_text(json.dumps(saved))
    assert validate_input(source,path,tmp_path/'tampered')==1
    assert 'geometry_scaling.generated' in capsys.readouterr().err


@pytest.mark.parametrize('factor_flag',['--curvature-factor','--thickness-factor'])
def test_calibration_single_factor_and_saved_settings(tmp_path,small_pdb,factor_flag,monkeypatch):
    monkeypatch.setattr('mastercap.step.fit_step',lambda r,**kw:SimpleNamespace(inner=.1,outer=2.5,thickness=2.4))
    prefix=tmp_path/'calib'
    assert main(['calibrate','--input',str(small_pdb),'--output-prefix',str(prefix),'--bin-width','.4',
                 '--no-smoothing',factor_flag,'1','--replicates','2'])==0
    result=json.loads((tmp_path/'calib_calibration.json').read_text());settings=result['calibration']['settings']
    assert settings['curvature_factor']==1 and settings['thickness_factor']==1
    assert settings['inner_radius']==result['geometry_scaling']['generated']['inner']
    # Compatibility is based on actual target geometry, not how it was specified.
    explicit=dict(settings,curvature_factor=None,thickness_factor=None)
    check_settings(settings,explicit)


def test_cli_help_lists_all_new_options():
    text=parser().format_help()
    for flag in ('--curvature-factor','--thickness-factor','--report-atom-data','--atom-data'):
        assert flag in text
    assert 'GEOMETRY FACTORS:' in text and 'REPORT FILES:' in text


@pytest.mark.parametrize('flags',[
    ['--curvature-factor','0'],['--thickness-factor','nan'],
    ['--report-atom-data','unknown'],
])
def test_invalid_geometry_and_layout_option_values(flags):
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(['generate','--input','unused.pdb','--output-prefix','unused',*flags])
    assert exc.value.code==2


@pytest.mark.parametrize('bad_array',[
    {'format_version':np.array(2,dtype=np.int32)},
    {'id':np.array([0.0])},
    {'unsafe':np.array([{'object':'not allowed'}],dtype=object)},
])
def test_invalid_or_pickled_sidecar_arrays_rejected(tmp_path,bad_array):
    path=tmp_path/'bad.npz'
    arrays={'format_version':np.array(1,dtype=np.int32),'id':np.array([0],dtype=np.int32),
            'synthetic_residue':np.array([0],dtype=np.int32),'reference_atom_index':np.array([0],dtype=np.int32),
            **bad_array}
    np.savez_compressed(path,**arrays)
    report={'atom_data':{'mode':'sidecar','path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'atoms':1}}
    with pytest.raises(MastercapError):load_atom_data(report,tmp_path/'bad.json')
