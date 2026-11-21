import csv
import io
import json
import numpy as np
import pytest
from mastercap.cli import main
from mastercap.reports import CALIBRATION_COLUMNS,calibration_png


def test_csv_and_png_include_sources_and_basis(tmp_path,small_pdb):
    import matplotlib.image
    prefix=tmp_path/'calib'
    assert main(['calibrate','--input',str(small_pdb),'--output-prefix',str(prefix),'--replicates','5',
                 '--generated',str(small_pdb),'--compare',str(small_pdb),'--floor-basis','counting','--csv','--plot'])==0
    data=json.loads((tmp_path/'calib_calibration.json').read_text())['calibration']
    assert data['generated'][0]['errors']['C']==data['levels']['self']['C']['mean']
    rows=list(csv.DictReader(io.StringIO((tmp_path/'calib_calibration.csv').read_text())))
    assert list(rows[0])==CALIBRATION_COLUMNS
    assert len(rows)==6*2
    assert {r['source'] for r in rows}=={'self','counting','orientation','placement','compare','generated'}
    assert sum(r['element']=='Total' for r in rows)==6
    for r in rows:
        if r['source'] in ('self','compare','generated'):
            assert all(r[name]=='' for name in ('sd','p05','p50','p95','min','max'))
        if r['suggested_tolerance']:
            assert r['source']=='counting' and r['element']=='C'
    png=tmp_path/'calib_calibration.png'
    assert png.stat().st_size>1000
    assert matplotlib.image.imread(png).shape[2]==4
    # A floor can be identically zero: log plots still render without inventing errors.
    for level in data['levels'].values():
        for stats in level.values():
            for key in stats:stats[key]=0
    data['suggested']['profile_tolerance']['C']=0
    assert len(calibration_png(data))>1000


@pytest.mark.parametrize('option,value',[('--replicates','0'),('--seed','-1'),('--floor-multiple','0'),
 ('--floor-multiple','nan'),('--floor-basis','invalid'),('--floor-statistic','invalid')])
def test_invalid_calibration_arguments_before_input(option,value,capsys):
    with pytest.raises(SystemExit):main(['calibrate','--input','missing',option,value])
    assert 'Reading reference' not in capsys.readouterr().err
