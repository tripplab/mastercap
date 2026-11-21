import hashlib
import json
import pytest
from mastercap.cli import main
from mastercap.constraints import acceptance
from mastercap.tolerances import profile_tolerance_spec,check_coverage


@pytest.mark.parametrize('value,expected',[('0.03',{'*':.03}),('c=.02,N=.04,O=.04,S=.15',{'C':.02,'N':.04,'O':.04,'S':.15}),('C=.02,N=.04,*=.05',{'C':.02,'N':.04,'*':.05})])
def test_forms(value,expected):
    assert profile_tolerance_spec(value)==expected


@pytest.mark.parametrize('value',['C=abc','C=.1,c=.2','*=.1,*=.2','X=.1','-.1','nan','inf','C=-1','C=inf','C=.1,0.03','','C=.1,'])
def test_invalid_precedes_input(value,capsys):
    with pytest.raises(SystemExit) as exc:
        main(['generate','--input','missing','--output-prefix','unused','--profile-tolerance',value])
    assert exc.value.code==2
    assert 'Invalid profile tolerance' in capsys.readouterr().err


def test_per_element_acceptance_and_missing_coverage():
    metrics={'elements':{e:{'profile_error':v,'count_difference':0,'target_count':100} for e,v in [('C',.02),('S',.14)]}}
    assert acceptance({'mandatory_pass':True},metrics,{'C':.03,'S':.1},0)=='approximate'
    assert metrics['elements']['C']['within_tolerance']
    assert not metrics['elements']['S']['within_tolerance']
    assert acceptance({'mandatory_pass':True},metrics,{'C':.03,'*':.15},0)=='valid'
    assert acceptance({'mandatory_pass':True},metrics,.15,0)=='valid'
    with pytest.raises(ValueError,match=r'element S; add S='):
        check_coverage({'C':.03},['C','S'])
    assert acceptance({'mandatory_pass':False},metrics,{'*':100},100)=='failed'


def test_coverage_failure_is_before_search(tmp_path,small_pdb,monkeypatch,capsys):
    import mastercap.cli as cli
    monkeypatch.setattr(cli,'search',lambda *a,**k:pytest.fail('search must not start'))
    assert main(['generate','--input',str(small_pdb),'--output-prefix',str(tmp_path/'missing'),
                 '--profile-tolerance','S=.1'])==1
    assert 'Profile tolerance has no value for element C' in capsys.readouterr().err


def test_generate_and_validate_saved_float_and_dict(tmp_path,small_pdb,capsys):
    prefix=tmp_path/'particle'
    assert main(['generate','--input',str(small_pdb),'--output-prefix',str(prefix),'--seed','42',
                 '--profile-tolerance','N=.3,*=2','--composition-tolerance','0','--refine-updates','0'])==0
    saved=json.loads(prefix.with_suffix('.json').read_text())
    assert saved['configuration']['profile_tolerance']=={'*':2.0}
    assert saved['configuration']['profile_tolerance_spec']=='N=.3,*=2'
    assert 'absent from the reference' in capsys.readouterr().err
    for value in (2.0,{'C':2.0}):
        saved['configuration']['profile_tolerance']=value
        prefix.with_suffix('.json').write_text(json.dumps(saved))
        out=tmp_path/('float' if isinstance(value,float) else 'dict')
        assert main(['validate','--input',str(small_pdb),'--pdb',str(prefix.with_suffix('.pdb')),
                     '--report',str(prefix.with_suffix('.json')),'--output-prefix',str(out)])==0
        report=json.loads(out.with_suffix('.json').read_text())
        assert report['metrics']['elements']['C']['tolerance']==2


def test_calibration_load_match_and_no_dependency_on_deleted_file(tmp_path,small_pdb):
    prefix=tmp_path/'calib'
    assert main(['calibrate','--input',str(small_pdb),'--output-prefix',str(prefix),'--replicates','4','--seed','42'])==0
    calibration=tmp_path/'calib_calibration.json';digest=hashlib.sha256(calibration.read_bytes()).hexdigest()
    output=tmp_path/'particle'
    code=main(['generate','--input',str(small_pdb),'--output-prefix',str(output),'--tolerances-from',str(calibration),
               '--seed','42','--allow-approximate','--refine-updates','0'])
    assert code in (0,2)
    report=json.loads(output.with_suffix('.json').read_text())
    assert report['configuration']['calibration_sha256']==digest
    assert isinstance(report['configuration']['profile_tolerance'],dict)
    calibration.unlink()
    assert main(['validate','--input',str(small_pdb),'--pdb',str(output.with_suffix('.pdb')),
                 '--report',str(output.with_suffix('.json')),'--output-prefix',str(tmp_path/'checked')])==code


def test_calibration_mismatch_and_conflicts(tmp_path,small_pdb,monkeypatch,capsys):
    import mastercap.cli as cli
    prefix=tmp_path/'calib'
    assert main(['calibrate','--input',str(small_pdb),'--output-prefix',str(prefix),'--bin-width','.5','--replicates','2'])==0
    path=tmp_path/'calib_calibration.json'
    monkeypatch.setattr(cli,'search',lambda *a,**k:pytest.fail('search must not start'))
    assert main(['generate','--input',str(small_pdb),'--output-prefix',str(tmp_path/'wrong'),'--tolerances-from',str(path)])==1
    assert 'Calibration settings differ: bin width' in capsys.readouterr().err
    for option,value in [('--profile-tolerance','.03'),('--composition-tolerance','0')]:
        with pytest.raises(SystemExit) as exc:
            main(['generate','--input','missing','--output-prefix','out','--tolerances-from',str(path),option,value])
        assert exc.value.code==2
