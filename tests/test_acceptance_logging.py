"""Acceptance messages must explain the actual policy and early stopping."""
import json

import pytest

from mastercap.cli import log_acceptance, main
from mastercap.constraints import acceptance


@pytest.mark.parametrize('tolerances,composition,status', [
    ({'C': .0753, 'N': .141, 'O': .113, 'S': .864}, 0, 'valid'),
    ({'*': .1}, 0, 'approximate'),
    (None, 0, 'uncalibrated'),
    ({'*': 1}, None, 'uncalibrated'),
])
def test_user_metrics_explain_each_effective_check(tolerances, composition, status, caplog):
    metrics = {'elements': {e: {'profile_error': error, 'count_difference': 0,
                               'target_count': 100}
                            for e, error in [('C', .0327), ('N', .1267), ('O', .0922), ('S', 0)]}}
    geometry = {'mandatory_pass': True}
    assert acceptance(geometry, metrics, tolerances, composition) == status
    with caplog.at_level('INFO', logger='mastercap'):
        log_acceptance(status, geometry, metrics, composition)
    output = caplog.text
    assert f'Acceptance status: {status}' in output
    assert 'Mandatory geometry: PASS' in output
    assert all(f'Acceptance {e}:' in output for e in metrics['elements'])
    if status == 'valid':
        assert 'error=0.1267 <= tolerance=0.141 PASS' in output
        assert 'absolute count difference=0 <= allowance=1 PASS' in output
        assert 'Total error is diagnostic' in output
    elif status == 'approximate':
        assert 'error=0.1267 > tolerance=0.1 FAIL' in output
    else:
        assert 'not configured' in output


def test_calibration_provenance_early_stop_and_saved_validation(tmp_path, small_pdb, capsys):
    calib = tmp_path/'calib'
    assert main(['calibrate', '--input', str(small_pdb), '--output-prefix', str(calib),
                 '--replicates', '2', '--plot']) == 0
    output = capsys.readouterr().err
    assert 'bars=p05-p95, dots=means' in output
    assert 'No generated structures supplied' in output
    calibration = tmp_path/'calib_calibration.json'
    saved = json.loads(calibration.read_text())
    # A deliberately permissive suggestion isolates the first-feasible stop policy.
    saved['calibration']['suggested']['profile_tolerance'] = {'C': 2}
    calibration.write_text(json.dumps(saved))
    prefix = tmp_path/'particle'
    assert main(['generate', '--input', str(small_pdb), '--output-prefix', str(prefix),
                 '--tolerances-from', str(calibration), '--seed', '42',
                 '--allow-approximate', '--refine-updates', '50']) == 0
    output = capsys.readouterr().err
    assert f'Loaded calibration tolerances: {calibration}' in output
    assert 'SHA-256' in output and 'settings matched' in output
    assert 'Effective profile tolerances: C=2' in output
    assert 'Effective composition tolerance: 0 relative' in output
    assert '0 of 50 maximum profile pushes used' in output
    assert 'Acceptance status: valid' in output
    report = json.loads(prefix.with_suffix('.json').read_text())
    assert report['search']['profile_updates'] == 0
    calibration.unlink()
    assert main(['validate', '--input', str(small_pdb), '--pdb', str(prefix.with_suffix('.pdb')),
                 '--report', str(prefix.with_suffix('.json')), '--output-prefix', str(tmp_path/'validation')]) == 0
    output = capsys.readouterr().err
    assert 'Using tolerances saved in generation report:' in output
    assert 'original calibration file is not reread' in output
    assert 'Loaded calibration tolerances:' not in output
    assert 'Acceptance C:' in output and 'Acceptance status: valid' in output


def test_zero_refinement_budget_does_not_claim_tolerance_success(tmp_path, small_pdb, capsys):
    assert main(['generate', '--input', str(small_pdb), '--output-prefix', str(tmp_path/'raw'),
                 '--seed', '42', '--allow-approximate', '--refine-updates', '0']) == 2
    output = capsys.readouterr().err
    assert 'because a refinement budget was set to 0; quality status=uncalibrated' in output
    assert 'all supplied profile/composition tolerances passed' not in output


@pytest.mark.parametrize('command', [[], ['generate'], ['calibrate'], ['validate']])
def test_help_explains_calibration_and_acceptance(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main(command + ['-h'])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    if not command or command == ['calibrate']:
        assert 'Reading the calibration PNG:' in output
        assert 'Hollow diamonds' in output and 'filled triangles' in output
        assert 'No particle is assessed unless supplied' in output
    if not command or command == ['generate']:
        assert 'Zero pushes can be valid' in output
        assert 'valid means these geometric/profile checks passed' in output
