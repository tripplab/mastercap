import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from mastercap.analysis import histogram, shell_volumes
from mastercap.generate import initialize, search, SulfurAnchors
from mastercap.model import Atom, Structure


def placement_case():
    elements = ['C','S','N','C','N','O','C','S','O','C','N','O']
    xyz = np.repeat([[10.,0,0],[0,12.,0],[-14.,0,0],[0,-16.,0]],3,axis=0)
    xyz += np.tile([[0,0,0],[.6,.2,.1],[-.4,.6,-.2]],(4,1))
    structure = Structure([Atom(e,e,'CYS' if i//3 in (0,2) else 'GLY',i//3) for i,e in enumerate(elements)], xyz)
    edges = np.array([6.,10,14,18,22])
    counts = {'C':[1,1,1,1], 'N':[1,1,0,1], 'O':[1,0,1,1], 'S':[0,1,1,0]}
    profiles = {e:np.array(v)/shell_volumes(edges) for e,v in counts.items()}
    return structure, edges, profiles, np.array([1.,2,3])


@pytest.mark.parametrize('distribution,xyz_hash,rng_hash', [
    ('regular','9c330e076e1aabc1931f2fd91f4adb3ff0090e6993538ffae0ddbc49f3bd0527','966245903fad24db5f93dd1db4320d901e7636289c84d29ca8fb82c6f4cc24c0'),
    ('random','67ebe2f5658cffb573c4f6fdcffab9d588ed8c2190543c7e961cfa4560111ac1','4299a7d9877c463678dae3033093d8f8306053e5ff1b38b79a908e31296e59ca')])
def test_off_is_bit_identical_to_v016_initializer(distribution, xyz_hash, rng_hash):
    structure, edges, profiles, center = placement_case()
    rng = np.random.default_rng(42)
    initialize(structure, None, edges, profiles, center, rng, 'independent', distribution,
               sulfur_residue_placement='off')
    assert hashlib.sha256(structure.xyz.tobytes()).hexdigest() == xyz_hash
    assert hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest() == rng_hash


@pytest.mark.parametrize('distribution', ['regular','random'])
@pytest.mark.parametrize('mode', ['independent','residue'])
def test_profile_anchors_exact_radii_histograms_and_rigid_shapes(distribution, mode):
    structure, edges, profiles, center = placement_case()
    original = structure.xyz.copy()
    anchors = SulfurAnchors()
    initialize(structure, None, edges, profiles, center, np.random.default_rng(42), mode, distribution,
               sulfur_residue_placement='profile', sulfur_anchors=anchors)
    radii = np.linalg.norm(structure.xyz[anchors.indices]-center, axis=1)
    assert np.allclose(radii, anchors.radii, atol=1e-6, rtol=0)
    assert np.array_equal(np.histogram(radii, edges)[0], [0,1,1,0])
    if distribution == 'regular':
        assert all(radius in (edges[:-1]+edges[1:])/2 for radius in anchors.radii)
    for group in range(4):
        ix = np.arange(3*group,3*group+3)
        assert np.allclose(np.linalg.norm(structure.xyz[ix]-structure.xyz[ix[0]],axis=1),
                           np.linalg.norm(original[ix]-original[ix[0]],axis=1))


def test_profile_is_reproducible_and_no_sulfur_is_exactly_off(caplog):
    outputs = []
    for _ in range(2):
        structure, edges, profiles, center = placement_case()
        initialize(structure, None, edges, profiles, center, np.random.default_rng(42), 'independent', 'regular',
                   sulfur_residue_placement='profile')
        outputs.append(structure.xyz.copy())
    assert np.array_equal(*outputs)
    structure, edges, profiles, center = placement_case()
    structure.atoms = [Atom(a.name,'O' if a.element=='S' else a.element,a.resname,a.residue) for a in structure.atoms]
    profiles['O'] += profiles.pop('S')
    rngs = [np.random.default_rng(42),np.random.default_rng(42)]
    outputs = []
    caplog.set_level('INFO')
    for setting, rng in zip(('off','profile'), rngs):
        candidate = structure.copy()
        initialize(candidate,None,edges,profiles,center,rng,'independent','regular', sulfur_residue_placement=setting)
        outputs.append(candidate.xyz)
    assert np.array_equal(*outputs)
    assert rngs[0].bit_generator.state == rngs[1].bit_generator.state
    assert 'no effect' in caplog.text


def test_anchor_spring_is_repair_only_and_arrays_stay_private():
    source = Structure([Atom(e,e,'ALA',i//2) for i,e in enumerate(('C','S','N','C'))],
                       np.array([[-2.4,0,0],[-.9,0,0],[.8,0,0],[2.4,0,0]]))
    edges = np.array([0.,2,4])
    profiles = histogram(source,np.zeros(3),edges,('C','N','S'))
    limits = {(a,b):.1 for a in ('C','N','S') for b in ('C','N','S') if a<=b}
    limits[('C','S')] = 1.59
    analysis = SimpleNamespace(limits=limits, original=profiles)
    def preset(structure, *args, sulfur_anchors, **kwargs):
        sulfur_anchors.indices = np.array([1])
        sulfur_anchors.radii = np.array([.8])
        sulfur_anchors.summary = {'mode':'profile','anchored_residue_count':1}
        return []
    with patch('mastercap.generate.initialize', side_effect=preset):
        output, report = search(source,analysis,edges,profiles,np.zeros(3),np.random.default_rng(42),
                                sulfur_residue_placement='profile',consolidate_iterations=0,
                                refine_updates=2,refine_patience=0,refine_selection='mismatch',
                                refine_strength=1e-10,refine_min_step=0,time_limit=30)
    assert output is not None
    assert 0 < report['sulfur_placement']['repair_anchor_updates'] == report['first_feasible_iteration']
    assert all(push['force_components']['sulfur_anchor']['max_move_angstrom']==0 for push in report['push_history'])
    assert 'indices' not in report['sulfur_placement']
    json.dumps(report)


def test_selection_is_by_element_and_multi_sulfur_residues_warn_once(caplog):
    structure, edges, profiles, center = placement_case()
    structure.atoms[:3] = [Atom(a.name, 'S' if i==0 else a.element, 'GLY', a.residue)
                           for i,a in enumerate(structure.atoms[:3])]
    anchors = SulfurAnchors()
    initialize(structure,None,edges,profiles,center,np.random.default_rng(42),'independent','regular',
               sulfur_residue_placement='profile',sulfur_anchors=anchors)
    assert anchors.summary['anchored_residue_count'] == 2
    assert anchors.summary['multiple_sulfur_residues'] == 1
    assert anchors.indices[0] == 0
    assert anchors.summary['residue_names']['GLY'] == 1
    assert caplog.text.count('multi-S residues') == 1


def test_profile_retains_each_seeded_residue_rotation():
    structure, edges, profiles, center = placement_case()
    outputs = []
    for mode in ('off','profile'):
        candidate = structure.copy()
        trace = initialize(candidate,None,edges,profiles,center,np.random.default_rng(42),
                           'independent','regular',sulfur_residue_placement=mode)
        outputs.append([t['rotation'] for t in trace])
    assert np.array_equal(*outputs)
