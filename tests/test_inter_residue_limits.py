import numpy as np
import pytest
from types import SimpleNamespace
from mastercap.model import Atom,Structure,covalent_floor
from mastercap.analysis import inter_residue_statistics
from mastercap.constraints import GeometryMetadata,local_geometry,pair_distance_audit,inter_residue_configuration
from mastercap.diagnostics import bond_keys,nearest_nonbonded
from mastercap.refinement import accept_trial


def fixture():
    # Residues 0/1 are peptide joined; residue 2 is an independent close contact.
    return Structure([Atom('C','C','GLY',0),Atom('N','N','GLY',1),Atom('CA','C','GLY',2)],
                     np.array([[0.,0,0],[1.33,0,0],[3.,0,0]]))


def test_statistics_exclude_peptides_and_disulfide_residue_pairs():
    s=fixture();limits,stats,_=inter_residue_statistics(s,1,1)
    assert limits['C','N']==pytest.approx(1.67)
    assert stats['C-N']['count']==2
    sulfur=Structure([Atom('SG','S','CYS',i) for i in range(3)],np.array([[0.,0,0],[2.05,0,0],[6.,0,0]]))
    values=nearest_nonbonded(sulfur,bond_keys(sulfur,disulfides=True))['S','S']
    assert values.min()==pytest.approx(3.95)
    assert inter_residue_statistics(sulfur,1,100)[1]['S-S']['grouped_fallback']


def test_effective_bounds_override_and_internal_exemption():
    s=Structure([Atom('CA','C','GLY',r) for r in [0,0,1]],np.array([[0.,0,0],[1.5,0,0],[4.,0,0]]))
    lower={('C','C'):1.2};raw={('C','C'):3.}
    metadata=GeometryMetadata.build(s,lower,enforce_covalent_floor=True,inter_residue_limits=raw,inter_residue_scale=.8)
    pairs=np.array([[0,1],[1,2]])
    np.testing.assert_allclose(metadata.pair_lower_bounds(pairs),[1.2,2.4])
    override=GeometryMetadata.build(s,lower,enforce_covalent_floor=True,inter_residue_limits=raw,overridden_pairs=['C-C'])
    np.testing.assert_allclose(override.pair_lower_bounds(pairs),[1.2,covalent_floor('C','C')])
    off=GeometryMetadata.build(s,lower,enforce_covalent_floor=True)
    np.testing.assert_array_equal(off.pair_lower_bounds(pairs),override.pair_lower_bounds(pairs))


def test_large_bound_reaches_geometry_acceptance_and_final_audit():
    s=Structure([Atom('CA','C','GLY',i) for i in range(2)],np.array([[0.,0,0],[5.,0,0]]))
    metadata=GeometryMetadata.build(s,{('C','C'):1.3},inter_residue_limits={('C','C'):6.})
    local=local_geometry(s,{('C','C'):1.3},metadata=metadata)
    assert local['violations'].sum()==1 # beyond the ordinary 3.4 A neighbor query
    audit=pair_distance_audit(s,{('C','C'):1.3},inter_residue_limits=metadata.effective_limits())
    assert audit['below_limit_count']==1 and not audit['passed']
    # Start feasible beyond the large floor; a move inside it must be rejected.
    s.xyz[1,0]=6.02
    local=local_geometry(s,{('C','C'):1.3},metadata=metadata)
    trial=s.xyz.copy();trial[1,0]=5.99
    accepted,details=accept_trial(s.xyz,trial,local,metadata,s.xyz.mean(axis=0),np.array([0.,10.]))
    np.testing.assert_array_equal(accepted,s.xyz)
    assert details['undone_by_reason']['too_close']>0


def test_no_cno_window_is_rejected_and_narrow_windows_warn():
    s=fixture();analysis=SimpleNamespace(structure=s,limits={('C','C'):1.3,('C','N'):1.3,('N','N'):1.3},
                                        inter_residue={('C','C'):5.,('C','N'):5.,('N','N'):5.},
                                        inter_scale=1.,distance_limit_overrides=[],inter_statistics={})
    with pytest.raises(ValueError,match='No positive C/N/O'):inter_residue_configuration(analysis)
    analysis.inter_scale=.67
    warnings=[];_,report=inter_residue_configuration(analysis,warn=warnings.append)
    assert warnings and report['windows']['C-N']<.05


def test_cli_active_limits_roundtrip_and_diagnostics(tmp_path,small_pdb):
    import json
    from mastercap.cli import main
    prefix=tmp_path/'inter'
    code=main(['generate','--input',str(small_pdb),'--output-prefix',str(prefix),'--seed','42','--refine-updates','0',
               '--allow-approximate','--inter-residue-limits','reference','--inter-residue-scale','.6'])
    assert code==2
    report=json.loads(prefix.with_suffix('.json').read_text())
    assert report['analysis']['inter_residue_limits']['mode']=='reference'
    assert report['inter_residue_limits']['scale']==.6
    assert report['diagnostics']['reference']['contacts']['overlap_per_atom']>0
    assert main(['validate','--input',str(small_pdb),'--pdb',str(prefix.with_suffix('.pdb')),'--report',str(prefix.with_suffix('.json')),
                 '--output-prefix',str(tmp_path/'checked'),'--mesh-element-sizes','.25,.5'])==2
    checked=json.loads((tmp_path/'checked.json').read_text())
    assert checked['inter_residue_limits']==report['inter_residue_limits']
    assert checked['metrics']==report['metrics']
    assert set(checked['diagnostics']['contacts']['near_gap_per_atom'])=={'0.25','0.5'}


@pytest.mark.parametrize('extra,message',[
    (['--inter-residue-scale','.6'],'requires --inter-residue-limits reference'),
    (['--refine-drift-step','.1'],'no larger than refine max step'),
])
def test_invalid_option_interactions_precede_input(extra,message,capsys):
    from mastercap.cli import main
    assert main(['generate','--input','missing','--output-prefix','unused',*extra])==1
    output=capsys.readouterr().err
    assert message in output and 'Reading reference coordinates' not in output


def test_off_retains_legacy_component_candidate_filter_at_large_margin():
    from mastercap.generate import connect_edges
    s=Structure([Atom('CA','C','GLY',i) for i in range(2)],np.array([[0.,0,0],[2.,0,0]]))
    limits={('C','C'):1.}
    off=GeometryMetadata.build(s,limits,enforce_covalent_floor=True)
    assert not off.inter_residue_enabled
    # Legacy proposals used the original element floor, even if a covalent guard
    # ultimately makes the large-margin geometry unachievable. Preserve that path.
    assert len(connect_edges(s.xyz,np.array([0,1]),metadata=off,margin=2.1))==1
    on=GeometryMetadata.build(s,limits,inter_residue_limits={('C','C'):1.4})
    assert len(connect_edges(s.xyz,np.array([0,1]),metadata=on,margin=2.1))==0
