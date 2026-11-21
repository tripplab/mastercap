import numpy as np
import pytest
from mastercap.analysis import (analyze, target_profiles, shell_volumes, distance_statistics,
                                apply_limit_overrides)
from mastercap.model import Atom, Structure, VDW


def test_smoothing_preserves_integrated_element_counts(small_structure):
    analysis = analyze(small_structure, width=0.2, sigma=0.4)
    volumes = shell_volumes(analysis.edges)
    for e in analysis.original:
        assert np.isclose(np.sum(analysis.original[e]*volumes), np.sum(analysis.smoothed[e]*volumes))


def test_resize_preserves_density_not_count(small_structure):
    analysis = analyze(small_structure, width=0.2)
    edges, profiles = target_profiles(analysis, analysis.inner*2, analysis.outer*2)
    assert np.array_equal(profiles["C"], analysis.smoothed["C"])
    assert np.isclose(np.sum(profiles["C"]*shell_volumes(edges)), 8*len(small_structure.atoms))


def test_pair_statistics_and_grouped_fallback():
    s = Structure([Atom("CA", "C", "ALA", 0), Atom("N", "N", "ALA", 0), Atom("O", "O", "ALA", 0)],
                  np.array([[0, 0, 0], [1.4, 0, 0], [0, 2, 0]]))
    limits, report, grouped = distance_statistics(s)
    assert all(v > 0 for v in limits.values())
    assert all(item["grouped_fallback"] for item in report.values())
    assert report["C-N"]["count"] == 2


def test_percentiles_exclude_and_report_atoms(small_structure):
    small_structure.xyz[3, 0] = 5.0
    analysis = analyze(small_structure, radius_method="percentile", percentiles=(0, 80))
    assert analysis.excluded == 1
    assert len(analysis.structure.atoms) == 3


def test_mixed_element_profiles_keep_separate_counts():
    elements = ["C", "N", "O", "S", "H", "C"]
    structure = Structure([Atom(e, e, "ALA", i//3) for i, e in enumerate(elements)],
                          np.array([[1,0,0], [0,2,0], [0,0,3], [-4,0,0], [0,-5,0], [0,0,-6.0]]))
    report = analyze(structure, width=0.4, sigma=0.8, min_samples=2)
    for e in set(elements):
        assert np.isclose(np.sum(report.smoothed[e]*shell_volumes(report.edges)), elements.count(e))


def sulfur_reference():
    return Structure([Atom(e, e, "CYS", i//2) for i, e in enumerate(["C", "S"]*6)],
                     np.array([[10*i+offset, 0, 0] for i in range(6) for offset in (0, 2.0)]))


def test_default_sulfur_override_changes_only_sulfur_spacing():
    structure = sulfur_reference()
    limits, samples, grouped = distance_statistics(structure, min_samples=2)
    result = analyze(structure, min_samples=2)
    assert result.limits[("S", "S")] == pytest.approx(limits[("C", "S")]/3.5*3.6)
    assert result.limits[("S", "S")] != limits[("S", "S")]
    for pair in limits:
        if pair != ("S", "S"):
            assert result.limits[pair] == limits[pair]
    assert result.neighbor_samples == samples
    assert result.distance_limit_overrides[0]["reason"] == "statistical spacing, not a steric limit"


def test_reference_and_numeric_overrides_are_recorded_exactly():
    limits = {("S", "S"): 14.0, ("C", "N"): 1.5}
    restored, records = apply_limit_overrides(limits, 0.4, ["S-S=reference", "N-C=1.4"])
    assert restored == {("S", "S"): 14.0, ("C", "N"): 1.4}
    assert {record["mode"] for record in records} == {"reference", "number"}


@pytest.mark.parametrize("specs", [["Q-S=steric"], ["S=steric"], ["S-S=0"], ["S-S=-1"],
                                   ["S-S=nan"], ["S-S=inf"], ["C-N=1", "N-C=2"]])
def test_bad_override_arguments_fail_before_input(specs):
    from mastercap.cli import main
    args = ["analyze", "--input", "missing.pdb", "--output-prefix", "out"]
    for spec in specs:
        args += ["--pair-limit-override", spec]
    with pytest.raises(SystemExit) as stopped:
        main(args)
    assert stopped.value.code == 2


def test_validate_reuses_saved_overrides_and_handles_legacy_reports(tmp_path, small_pdb, capsys):
    import json
    from mastercap.cli import main
    prefix = tmp_path/"saved"
    assert main(["generate", "--input", str(small_pdb), "--output-prefix", str(prefix), "--seed", "42",
                 "--pair-limit-override", "C-C=1.36", "--allow-approximate", "--refine-updates", "0"]) == 2
    report_path = prefix.with_suffix(".json")
    saved = json.loads(report_path.read_text())
    for legacy in (False, True):
        if legacy:
            prefix = tmp_path/'legacy-source'
            assert main(['generate','--input',str(small_pdb),'--output-prefix',str(prefix),
                         '--seed','42','--allow-approximate','--refine-updates','0',
                         '--pair-limit-override','S-S=reference']) == 2
            report_path = prefix.with_suffix('.json')
            saved = json.loads(report_path.read_text())
            saved["analysis"].pop("distance_limit_overrides")
            saved["analysis_options"].pop("pair_limit_overrides")
            saved["configuration"].pop("pair_limit_override")
            report_path.write_text(json.dumps(saved))
        output = tmp_path/("legacy" if legacy else "current")
        code = main(["validate", "--input", str(small_pdb), "--pdb", str(prefix.with_suffix(".pdb")),
                     "--report", str(report_path), "--output-prefix", str(output)])
        if not legacy:
            assert code == 2
            validated = json.loads(output.with_suffix(".json").read_text())
            assert validated["distance_limit_overrides"] == saved["analysis"]["distance_limit_overrides"]
        else:
            assert code == 2
            assert "Legacy report has no distance-limit override record" in capsys.readouterr().err


def test_generate_sulfur_then_validate_uses_saved_steric_not_cli_override(tmp_path):
    import json
    from mastercap.pdbio import pdb_text
    from mastercap.cli import main
    source = Structure([Atom('C' if i%2==0 else 'SG',e,'CYS',i//2) for i,e in enumerate(('C','S','C','S'))],
                       np.array([[-3.,0,0],[-1.,0,0],[1.,0,0],[3.,0,0]]))
    reference = tmp_path/'sulfur.pdb'; reference.write_text(pdb_text(source)[0])
    prefix = tmp_path/'generated'
    assert main(['generate','--input',str(reference),'--output-prefix',str(prefix),'--seed','42',
                 '--min-pair-samples','2','--allow-approximate','--refine-updates','0']) == 2
    saved = json.loads(prefix.with_suffix('.json').read_text())
    entry = next(r for r in saved['analysis']['distance_limit_overrides'] if r['pair']=='S-S')
    assert entry['new_value'] != entry['original_value']
    output = tmp_path/'verified'
    assert main(['validate','--input',str(reference),'--pdb',str(prefix.with_suffix('.pdb')),
                 '--report',str(prefix.with_suffix('.json')),'--output-prefix',str(output),
                 '--pair-limit-override','S-S=reference']) == 2
    validated = json.loads(output.with_suffix('.json').read_text())
    assert validated['distance_limit_overrides'] == saved['analysis']['distance_limit_overrides']


def test_element_derived_steric_and_pooled_guard():
    limits = {('C','C'):1.4, ('C','S'):2.709, ('N','S'):3.080, ('O','S'):3.467, ('S','S'):14.068}
    measured = {'-'.join(pair):{'grouped_fallback':False} for pair in limits}
    resolved, records = apply_limit_overrides(limits,.4,neighbor_samples=measured)
    assert resolved['S','S'] == pytest.approx(2.709/3.5*3.6,abs=1e-6)
    assert records[0]['source_pair'] == 'C-S'
    assert records[0]['ratio'] == pytest.approx(2.709/3.5)
    resolved, records = apply_limit_overrides(limits,.7,['S-S=pooled'],measured)
    assert resolved['S','S'] == pytest.approx(.7*3.6)
    assert records[0]['mode'] == 'pooled'
    with pytest.raises(ValueError,match='below the covalent floor'):
        apply_limit_overrides(limits,.4,['S-S=pooled'],measured)
    measured['C-S']['grouped_fallback'] = True
    resolved, records = apply_limit_overrides(limits,.4,neighbor_samples=measured)
    assert records[0]['source_pair'] == 'N-S'
    with pytest.raises(ValueError,match='numeric pair-limit override'):
        apply_limit_overrides({('S','S'):14.0},.4,neighbor_samples={})


def test_numeric_covalent_guard_precedes_input(capsys):
    from mastercap.cli import main
    with pytest.raises(SystemExit):
        main(['analyze','--input','missing','--output-prefix','unused','--pair-limit-override','S-S=1.371'])
    error = capsys.readouterr().err
    assert 'S-S override 1.371 A is below the covalent floor 1.85 A' in error
    assert 'Reading reference' not in error


def test_validate_old_steric_record_uses_numeric_value(tmp_path,capsys):
    import json
    from mastercap.cli import main
    from mastercap.pdbio import pdb_text
    source=Structure([Atom('C' if i%2==0 else 'SG',e,'CYS',i//2) for i,e in enumerate(('C','S','C','S'))],
                     np.array([[-3.,0,0],[-1.,0,0],[1.,0,0],[3.,0,0]]))
    reference=tmp_path/'reference.pdb';reference.write_text(pdb_text(source)[0])
    prefix=tmp_path/'legacy'
    assert main(['generate','--input',str(reference),'--output-prefix',str(prefix),'--seed','42',
                 '--min-pair-samples','2','--allow-approximate','--refine-updates','0']) == 2
    report_path=prefix.with_suffix('.json'); saved=json.loads(report_path.read_text())
    saved['mastercap_version']='0.1.7'
    entry=saved['analysis']['distance_limit_overrides'][0]
    entry.pop('ratio');entry.pop('source_pair');entry['new_value']=1.371
    report_path.write_text(json.dumps(saved))
    out=tmp_path/'checked'
    assert main(['validate','--input',str(reference),'--pdb',str(prefix.with_suffix('.pdb')),
                 '--report',str(report_path),'--output-prefix',str(out)]) == 2
    result=json.loads(out.with_suffix('.json').read_text())
    assert result['distance_limits']['S-S'] == 1.371
    assert result['distance_limit_overrides'] == saved['analysis']['distance_limit_overrides']
    assert 'Saved S-S override 1.371 A predates the covalent floor' in capsys.readouterr().err


def test_reference_mode_pair_remains_measured_steric_evidence():
    limits={('C','S'):2.709,('N','S'):3.080,('S','S'):14.068}
    measured={'-'.join(pair):{'grouped_fallback':False} for pair in limits}
    resolved,records=apply_limit_overrides(limits,.4,['C-S=reference'],measured)
    sulfur=next(r for r in records if r['pair']=='S-S')
    assert sulfur['source_pair']=='C-S'
    assert resolved['S','S']==pytest.approx(2.709/3.5*3.6)


def test_legacy_unsafe_coordinates_write_named_distance_audit_diagnostic(tmp_path,capsys):
    import hashlib
    import json
    from mastercap.cli import main
    from mastercap.pdbio import pdb_text,read_pdb
    source=Structure([Atom('C' if i%2==0 else 'SG',e,'CYS',i//2) for i,e in enumerate(('C','S','C','S'))],
                     np.array([[-3.,0,0],[-1.,0,0],[1.,0,0],[3.,0,0]]))
    reference=tmp_path/'reference.pdb';reference.write_text(pdb_text(source)[0])
    prefix=tmp_path/'old'
    assert main(['generate','--input',str(reference),'--output-prefix',str(prefix),'--seed','42',
                 '--min-pair-samples','2','--allow-approximate','--refine-updates','0']) == 2
    pdb=prefix.with_suffix('.pdb'); report=prefix.with_suffix('.json')
    saved=json.loads(report.read_text()); saved['mastercap_version']='0.1.7'
    entry=saved['analysis']['distance_limit_overrides'][0]
    entry.pop('ratio'); entry.pop('source_pair'); entry['new_value']=1.371
    particle=read_pdb(pdb,generated=True)
    particle.xyz[:]=[[-3.,0,0],[-1.,0,0],[2.39,0,0],[.39,0,0]]
    pdb.write_text(pdb_text(particle)[0]); saved['pdb_sha256']=hashlib.sha256(pdb.read_bytes()).hexdigest()
    report.write_text(json.dumps(saved)); original=report.read_bytes()
    output=tmp_path/'failed'
    assert main(['validate','--input',str(reference),'--pdb',str(pdb),'--report',str(report),
                 '--output-prefix',str(output)]) == 1
    diagnostic=json.loads((tmp_path/'failed.diagnostic.json').read_text())
    assert diagnostic['connectivity_audit']['passed']
    sulfur=diagnostic['pair_distance_audit']['pairs']['S-S']
    assert sulfur['minimum_distance_angstrom']==pytest.approx(1.39)
    assert sulfur['below_covalent_floor_count']==1
    assert 'serial 2 SG CYS' in diagnostic['error'] and 'serial 4 SG CYS' in diagnostic['error']
    assert report.read_bytes()==original
    assert not output.with_suffix('.json').exists()
