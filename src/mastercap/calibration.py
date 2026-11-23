"""Reference resampling baselines. No particle generation or geometry repair."""
from decimal import Decimal, ROUND_CEILING
from itertools import combinations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import __version__
from .analysis import shell_volumes
from .model import MastercapError
from .pdbio import read_pdb
from .tolerances import check_coverage


def errors_from_counts(counts, edges, target):
    volume=shell_volumes(edges)
    errors={}
    absolute=denominator=0.0
    for e,goal in target.items():
        n=float(np.sum(goal*volume))
        error=float(np.sum(np.abs(np.asarray(counts[e])/volume-goal)*volume))
        errors[e]=error/n if n else None
        absolute+=error;denominator+=n
    return {**errors,'Total':absolute/denominator if denominator else None}


def profile_errors(radii_by_element, edges, target):
    return errors_from_counts({e:np.histogram(radii_by_element.get(e,[]),bins=edges)[0] for e in target},edges,target)


def statistics(values):
    values=np.asarray(values,dtype=float)
    return {'mean':float(values.mean()),'sd':float(values.std(ddof=0)),
            **{name:float(np.percentile(values,p)) for name,p in (('p05',5),('p50',50),('p95',95))},
            'min':float(values.min()),'max':float(values.max())}


def summarize(samples):
    return {e:statistics([row[e] for row in samples]) for e in samples[0]}


def residue_frame(structure):
    _,codes=np.unique(structure.residues,return_inverse=True)
    n=int(codes.max())+1
    counts=np.bincount(codes,minlength=n)
    centroids=np.column_stack([np.bincount(codes,weights=structure.xyz[:,k],minlength=n)/counts for k in range(3)])
    return codes,centroids,structure.xyz-centroids[codes]


def rotated_coordinates(codes,centroids,offsets,rotations):
    return centroids[codes]+np.einsum('nij,nj->ni',rotations[codes],offsets)


def resampling_levels(analysis,edges,target,replicates=50,seed=42,progress=None,model="reference"):
    if not isinstance(replicates,int) or replicates<1 or seed<0:
        raise MastercapError('Replicates must be a positive integer and seed nonnegative')
    if model not in ('reference','step'):
        raise MastercapError('Profile model must be reference or step')
    rng=np.random.default_rng(seed)
    structure=analysis.structure
    elements=structure.elements
    indices={e:np.flatnonzero(elements==e) for e in target}
    def measure(xyz):
        radii=np.linalg.norm(xyz-analysis.center,axis=1)
        return profile_errors({e:radii[ix] for e,ix in indices.items()},edges,target)
    levels={'self':{e:statistics([v]) for e,v in measure(structure.xyz).items()}}
    if progress:progress('self',levels['self'])
    volume=shell_volumes(edges)
    probabilities={e:p*volume/np.sum(p*volume) for e,p in target.items()}
    counting=[]
    for _ in range(replicates):
        counts={e:rng.multinomial(int(round(np.sum(target[e]*volume))) if model=="step" else len(indices[e]), probabilities[e]) for e in target}
        counting.append(errors_from_counts(counts,edges,target))
    levels['counting']=summarize(counting)
    if progress:progress('counting',levels['counting'])
    if model == 'step':
        levels.update({name: {'skipped': 'not defined for step targets'} for name in ('orientation','placement')})
        return levels
    codes,centroids,offsets=residue_frame(structure)
    for level in ('orientation','placement'):
        samples=[]
        radius=np.linalg.norm(centroids-analysis.center,axis=1)
        direction=(centroids-analysis.center)/np.maximum(radius,1e-12)[:,None]
        direction[radius<1e-12]=[1,0,0]
        for _ in range(replicates):
            centers=centroids if level=='orientation' else analysis.center+direction*radius[rng.permutation(len(radius))][:,None]
            rotations=Rotation.random(len(centroids),random_state=rng).as_matrix()
            samples.append(measure(rotated_coordinates(codes,centers,offsets,rotations)))
        levels[level]=summarize(samples)
        if progress:progress(level,levels[level])
    return levels


def profile_settings(analysis,edges,bin_width,radius_method='extrema',percentiles=(.5,99.5), *, model='reference', step_edge_width=None):
    return {'reference_hash':analysis.structure.provenance['sha256'],
            'reference_filename':Path(analysis.structure.provenance['path']).name,
            'model':analysis.structure.provenance['model'],'bin_width':float(bin_width),
            'smoothing_sigma':analysis.smoothing_sigma,'inner_radius':float(edges[0]),'outer_radius':float(edges[-1]),
            'edges_count':len(edges),'radius_method':radius_method,'radius_percentiles':list(percentiles),
            'profile_model':model,'step_edge_width':step_edge_width if model=='step' else None}


def check_settings(saved,current):
    labels={'reference_hash':'reference hash','model':'model','bin_width':'bin width','smoothing_sigma':'smoothing sigma',
            'inner_radius':'inner radius','outer_radius':'outer radius','radius_method':'radius method',
            'radius_percentiles':'radius percentiles','edges_count':'edges count',
            'profile_model':'profile model','step_edge_width':'step edge width'}
    saved={'profile_model':'reference','step_edge_width':None,**saved}
    current={'profile_model':'reference','step_edge_width':None,**current}
    differences=[]
    for key,label in labels.items():
        if key not in saved:
            raise MastercapError(f'Calibration settings missing {label}')
        # Percentiles do not affect an extrema profile.
        if key=='radius_percentiles' and current['radius_method']=='extrema' and saved['radius_method']=='extrema':continue
        a,b=saved[key],current[key]
        same=(np.isclose(a,b,rtol=0,atol=1e-9) if key in ('inner_radius','outer_radius') else a==b)
        if not same:differences.append(f'{label} {a} vs {b}')
    if differences:raise MastercapError('Calibration settings differ: '+'; '.join(differences))


def settings_from_generation(saved):
    options=saved['analysis_options'];config=saved['configuration'];analysis=saved['analysis']
    return {'reference_hash':analysis['reference']['sha256'],'model':analysis['reference']['model'],
            'bin_width':options['width'],'smoothing_sigma':options['sigma'],
            'inner_radius':saved['target']['edges'][0],'outer_radius':saved['target']['edges'][-1],
            'edges_count':len(saved['target']['edges']),'radius_method':options['radius_method'],
            'radius_percentiles':options['percentiles'],
            'profile_model':'step' if options.get('step_profile',False) else 'reference',
            'step_edge_width':options.get('step_edge_width',4.0) if options.get('step_profile',False) else None}


def load_tolerances(path,current,elements,warning=None):
    data=Path(path).read_bytes()
    try:
        report=json.loads(data)
        if not isinstance(report,dict) or report.get('command')!='calibrate' or report.get('schema_version')!=1:
            raise MastercapError('Not a supported calibration report')
        calibration=report['calibration']
        check_settings(calibration['settings'],current)
        suggested=calibration['suggested']
        value=check_coverage(suggested['profile_tolerance'],elements,warning)
        composition=float(suggested['composition_tolerance'])
        if value is None or not np.isfinite(composition) or composition<0:
            raise MastercapError('Invalid suggested tolerances')
    except (KeyError,TypeError,json.JSONDecodeError) as exc:
        raise MastercapError(f'Invalid calibration report: {path}: {exc}') from exc
    return value,composition,hashlib.sha256(data).hexdigest()


def _counts(structure,center,edges,target):
    radial=np.linalg.norm(structure.xyz-center,axis=1)
    elements=structure.elements
    return {e:np.histogram(radial[elements==e],bins=edges)[0] for e in target}


def comparison(structure,center,edges,target):
    counts=_counts(structure,center,edges,target)
    volume=shell_volumes(edges)
    totals={e:float(np.sum(p*volume)) for e,p in target.items()}
    actual={e:int(np.count_nonzero(structure.elements==e)) for e in target}
    for e,n in actual.items():
        if not n:raise MastercapError(f'Comparison structure is missing element {e}')
        if not counts[e].sum():raise MastercapError(f'Comparison element {e} has no atoms within the reference radial bounds')
    shape={e:counts[e]*totals[e]/counts[e].sum() for e in target}
    return {'count_ratio':{e:actual[e]/totals[e] for e in target},
            'element_counts':actual,'in_range_counts':{e:int(c.sum()) for e,c in counts.items()},
            'raw':errors_from_counts(counts,edges,target),'shape':errors_from_counts(shape,edges,target)}


def chain_spread(analysis,edges,target):
    labels=np.array([a.origin.get('chain','') for a in analysis.structure.atoms])
    names,counts=np.unique(labels,return_counts=True)
    if len(names)<2 or '' in names or np.any(counts<1000):
        return {'available':False,'reason':'Need at least two nonempty chain labels, each with at least 1000 atoms'}
    volume=shell_volumes(edges)
    totals={e:float(np.sum(p*volume)) for e,p in target.items()}
    profiles={};errors={}
    for name in names:
        subset=analysis.structure.subset(labels==name)
        try:errors[str(name)]=comparison(subset,analysis.center,edges,target)
        except MastercapError as exc:return {'available':False,'reason':str(exc)}
        hist=_counts(subset,analysis.center,edges,target)
        profiles[str(name)]={e:hist[e]/hist[e].sum()*totals[e] for e in target}
    pairs=[]
    for a,b in combinations(profiles,2):
        goal={e:profiles[a][e]/volume for e in target}
        pairs.append({'labels':[a,b],'errors':errors_from_counts(profiles[b],edges,goal)})
    return {'available':True,'labels':errors,'pairwise':pairs,
            'spread':summarize([pair['errors'] for pair in pairs]),
            'used_for_suggestions':False}


def generated_assessment(path,analysis,edges,target,settings,levels):
    path=Path(path);particle=read_pdb(path,generated=True)
    sibling=Path(str(path)[:-3]) if path.suffix=='.gz' else path
    sibling=sibling.with_suffix('.json')
    center=analysis.center
    if sibling.exists():
        saved=json.loads(sibling.read_text())
        try:
            check_settings(settings_from_generation(saved),settings)
            center=np.asarray(saved['target']['center'],dtype=float)
            if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError('invalid center')
            if saved['pdb_sha256']!=particle.provenance['sha256']:
                raise MastercapError(f'Generated PDB hash differs from sibling report: {path}')
        except (KeyError,TypeError,ValueError) as exc:
            if isinstance(exc,MastercapError):raise
            raise MastercapError(f'Invalid sibling generation report {sibling}: {exc}') from exc
    counts=_counts(particle,center,edges,target)
    errors=errors_from_counts(counts,edges,target)
    step = settings.get('profile_model','reference') == 'step'
    multiples={level:{e:(error/levels[level][e]['mean'] if 'skipped' not in levels[level] and levels[level][e]['mean']>0 else None)
                      for e,error in errors.items()} for level in ('counting','orientation')}
    gap={};reasons={}
    for e,error in errors.items():
        if step:
            gap[e]=None
            reasons[e]='step target: no orientation/placement levels'
            continue
        placement,orientation=levels['placement'][e]['mean'],levels['orientation'][e]['mean']
        gap[e]=(placement-error)/(placement-orientation) if placement>orientation else None
        if gap[e] is None:reasons[e]='placement mean is not above orientation mean'
    return {'file':str(path),'sha256':particle.provenance['sha256'],'errors':errors,
            'sibling_report':str(sibling) if sibling.exists() else None,'center':center.tolist(),
            'element_counts':{e:int(np.count_nonzero(particle.elements==e)) for e in target},
            'warnings':particle.warnings,'multiples_counting':multiples['counting'],
            'multiples_orientation':None if step else multiples['orientation'],'gap_closed':None if step else gap,
            'gap_closed_clipped':None if step else {e:min(1,v) if v is not None else None for e,v in gap.items()},
            'undefined_gap_reasons':reasons,'zero_floor_means':[level+':'+e for level in multiples for e,v in multiples[level].items() if v is None and not (step and level=='orientation')]}


def round_up_three(value):
    if not np.isfinite(value) or value<0:raise MastercapError('Suggested tolerance is nonfinite; reduce the floor multiple')
    decimal=Decimal(str(value))
    if not value:return 0.0,'0'
    rounded=decimal.quantize(Decimal(1).scaleb(decimal.adjusted()-2),rounding=ROUND_CEILING)
    return float(rounded),format(rounded,'.3g')


def suggest(levels,compares,multiple=2.0,basis='orientation',statistic='p95', *, model='reference'):
    if not np.isfinite(multiple) or multiple<=0 or basis not in ('orientation','counting') or statistic not in ('p95','mean'):
        raise MastercapError('Invalid calibration floor suggestion settings')
    if model=='step' and basis!='counting':
        raise MastercapError('orientation floors are not defined for step targets; use --floor-basis counting')
    tolerances={};rules={};formatted={}
    for e,stats in levels[basis].items():
        if e=='Total':continue
        floor=multiple*stats[statistic]
        variation=max((row['shape'][e] for row in compares),default=0.0) if model!='step' else 0.0
        value=max(floor,variation)
        tolerances[e],formatted[e]=round_up_three(value)
        rules[e]='real variation' if variation>floor else 'floor'
    line='--profile-tolerance '+','.join(f'{e}={formatted[e]}' for e in sorted(formatted))+' --composition-tolerance 0'
    return {'profile_tolerance':tolerances,'composition_tolerance':0,'basis':basis,'statistic':statistic,
            'floor_multiple':multiple,'rule_by_element':rules,'paste_line':line,'compare_used':bool(compares) and model!='step'}


def calibrate(analysis,edges,target,settings,replicates=50,seed=42,compares=(),generated=(),
              floor_multiple=2.0,floor_basis='orientation',floor_statistic='p95',progress=None):
    model=settings.get('profile_model','reference')
    levels=resampling_levels(analysis,edges,target,replicates,seed,progress,model=model)
    comparisons=[]
    for path in compares:
        structure=read_pdb(path,model=settings['model'])
        row=comparison(structure,structure.xyz.mean(axis=0),edges,target)
        comparisons.append({'file':str(path),'sha256':structure.provenance['sha256'],'warnings':structure.warnings,**row})
    assessed=[generated_assessment(path,analysis,edges,target,settings,levels) for path in generated]
    return {'settings':{**settings,'replicates':replicates,'seed':seed,'mastercap_version':__version__,'sd_ddof':0},
            'element_counts':{e:int(np.count_nonzero(analysis.structure.elements==e)) for e in target},
            'target_counts':{e:float(np.sum(p*shell_volumes(edges))) for e,p in target.items()},
            'levels':levels,'compare':comparisons,'generated':assessed,'chain_type_spread':chain_spread(analysis,edges,target),
            'suggested':suggest(levels,comparisons,floor_multiple,floor_basis,floor_statistic,model=model)}
