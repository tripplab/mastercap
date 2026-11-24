"""Mesh-scale packing diagnostics. These measurements never alter acceptance."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.spatial import cKDTree
from .model import MastercapError
from .radial import shell_volumes

CHUNK = 2_000_000


def contact_zone(radial, target, zone=5.0, shell=0.25):
    """Atom-centre densities next to each support wall, with exact target integrals."""
    from .analysis import target_on_edges
    if not np.isfinite([zone, shell]).all() or min(zone, shell) <= 0:
        raise MastercapError('Contact-zone depth and shell width must be finite and positive')
    radial = np.asarray(radial)
    length = min(zone, target.edges[-1]-target.edges[0])
    if length <= 0:
        raise MastercapError('Contact zone requires a positive target support width')
    depths = np.r_[np.arange(0., length, shell), length]
    result = {}
    for wall in ('inner', 'outer'):
        edges = target.edges[0]+depths if wall == 'inner' else target.edges[-1]-depths[::-1]
        edges = np.clip(edges,target.edges[0],target.edges[-1])
        volumes = shell_volumes(edges)
        observed = np.histogram(radial, edges)[0]/volumes
        expected = sum(target_on_edges(target, edges).values())
        expected_count = float(np.sum(expected*volumes))
        ratios = [float(a/b) if b > 0 else None for a,b in zip(observed,expected)]
        near_depth = min(0.5, length)
        near_edges = np.array([target.edges[0], target.edges[0]+near_depth] if wall == 'inner'
                              else [target.edges[-1]-near_depth, target.edges[-1]])
        near_edges = np.clip(near_edges,target.edges[0],target.edges[-1])
        near_expected = float(np.sum(sum(target_on_edges(target, near_edges).values())*shell_volumes(near_edges)))
        near_observed = int(np.histogram(radial, near_edges)[0].sum())
        mids = (edges[:-1]+edges[1:])/2
        slope_ratio = None
        if len(mids) > 1:
            x = mids-mids.mean()
            denominator = float(np.dot(x,x))
            slope = float(np.dot(x,expected-expected.mean())/denominator)
            scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)/length
            if abs(slope) > 1e-12*scale:
                slope_ratio = float(np.dot(x,observed-observed.mean())/denominator/slope)
        result[wall] = {'shells_angstrom': edges.tolist(), 'observed': observed.tolist(),
            'expected': expected.tolist(), 'ratio': ratios,
            'zone_error': float(np.sum(np.abs(observed-expected)*volumes)/expected_count) if expected_count else None,
            'wall_excess_0p5': near_observed/near_expected if near_expected else None,
            'slope_ratio': slope_ratio}
    return result


def edge_concentration(radial, edges, thresholds=(0.002, 0.005, 0.01)):
    """Measure interior-edge sheets; support walls never count as edges here."""
    radial, edges = np.asarray(radial), np.asarray(edges)
    interior = edges[1:-1]
    widths = np.diff(edges)
    if not len(widths) or np.any(widths <= 0):
        raise MastercapError('Edge concentration requires increasing bin edges')
    if len(interior):
        pos = np.searchsorted(interior, radial)
        distance = np.minimum(np.abs(radial-interior[np.clip(pos, 0, len(interior)-1)]),
                              np.abs(radial-interior[np.clip(pos-1, 0, len(interior)-1)]))
    else:
        distance = np.full(len(radial), np.inf)
    counts = np.histogram(radial, edges)[0]
    sides = np.full(len(widths), 2)
    sides[0] -= 1
    sides[-1] -= 1
    fractions, expected, ratios = {}, {}, {}
    for t in thresholds:
        if not np.isfinite(t) or t <= 0:
            raise MastercapError('Edge thresholds must be finite and positive')
        key = str(t)
        fractions[key] = float(np.mean(distance < t)) if len(radial) else 0.
        expected[key] = float(np.sum(counts*np.minimum(1., sides*t/widths))/len(radial)) if len(radial) else 0.
        ratios[key] = fractions[key]/expected[key] if expected[key] else None
    bands = np.linspace(0, 0.5*np.median(widths), 11)
    histogram = np.histogram(distance, bands)[0]
    return {'fraction_within': fractions, 'expected_within': expected, 'ratio_within': ratios,
            'band_histogram': {'edges_angstrom': bands.tolist(),
                               'fractions': (histogram/max(1, len(radial))).tolist(),
                               'fraction_outside_bands': float(np.count_nonzero(distance > bands[-1])/max(1, len(radial)))}}


def mesh_sizes(value):
    try:
        values = [float(x) for x in (value.split(',') if isinstance(value,str) else value)]
        if not values or not np.isfinite(values).all() or min(values)<=0 or len(set(values))!=len(values):
            raise ValueError
        return tuple(sorted(values))
    except (TypeError,ValueError):
        raise MastercapError('Mesh element sizes must be distinct finite positive numbers, e.g. 0.5,1.0') from None


def residue_codes(structure):
    return np.unique(structure.residues,return_inverse=True)[1]


def bond_keys(structure, pairs=None, distances=None, *, disulfides=False):
    """Whole-residue exclusions by observed C--N geometry (and optionally S--S)."""
    codes=residue_codes(structure);nres=int(codes.max())+1
    names=np.array([a.name for a in structure.atoms]);elements=structure.elements
    if pairs is None:
        pairs=cKDTree(structure.xyz).query_pairs(2.3 if disulfides else 1.6,output_type='ndarray')
    found=[]
    for start in range(0,len(pairs),CHUNK):
        p=pairs[start:start+CHUNK]
        d=(np.linalg.norm(structure.xyz[p[:,0]]-structure.xyz[p[:,1]],axis=1)
           if distances is None else distances[start:start+CHUNK])
        a,b=p.T
        peptide=(((names[a]=='C') & (names[b]=='N')) | ((names[b]=='C') & (names[a]=='N'))) & (d<1.6)
        bonded=peptide
        if disulfides:bonded=bonded | ((elements[a]=='S') & (elements[b]=='S') & (d<2.3))
        bonded &= codes[a]!=codes[b]
        if bonded.any():
            residues=np.sort(codes[p[bonded]],axis=1)
            found.append(residues[:,0]*nres+residues[:,1])
    return np.unique(np.concatenate(found)) if found else np.empty(0,dtype=int)


def nonbonded_mask(codes, pairs, bonded):
    residues=np.sort(codes[pairs],axis=1)
    nres=int(codes.max())+1
    return (residues[:,0]!=residues[:,1]) & ~np.isin(residues[:,0]*nres+residues[:,1],bonded)


def nearest_nonbonded(structure, bonded):
    """Exact directed nearest observations, expanding k only for unresolved atoms."""
    xyz=structure.xyz;elements=structure.elements;codes=residue_codes(structure)
    nres=int(codes.max())+1
    groups={e:np.flatnonzero(elements==e) for e in sorted(set(elements))}
    trees={e:cKDTree(xyz[ix]) for e,ix in groups.items()}
    def directed(a,b):
        results=[];target=groups[b]
        for start in range(0,len(groups[a]),16384):
            pending=groups[a][start:start+16384];k=min(4,len(target))
            while len(pending):
                d,near=trees[b].query(xyz[pending],k=list(range(1,k+1)))
                left=codes[pending,None];right=codes[target[near]]
                keys=np.minimum(left,right)*nres+np.maximum(left,right)
                eligible=(left!=right) & ~np.isin(keys,bonded)
                distances=np.min(np.where(eligible,d,np.inf),axis=1)
                ok=np.isfinite(distances)
                results.append(distances[ok]);pending=pending[~ok]
                if k==len(target):break
                k=min(len(target),2*k)
        return np.concatenate(results) if results else np.array([])
    samples={}
    for a,b in combinations_with_replacement(groups,2):
        samples[a,b]=directed(a,b) if a==b else np.concatenate([directed(a,b),directed(b,a)])
    return samples


def layering_summary(density):
    if len(density)<9:return {'relative_sd':None,'min':None,'max':None,'shell_count':0}
    trend=np.convolve(density,np.ones(9)/9,mode='valid')
    observed=np.asarray(density)[4:-4]
    ratio=observed[trend>0]/trend[trend>0]
    return {'relative_sd':float(ratio.std()/ratio.mean()) if len(ratio) and ratio.mean() else None,
            'min':float(ratio.min()) if len(ratio) else None,'max':float(ratio.max()) if len(ratio) else None,
            'shell_count':len(ratio)}


def radial_layering(radial,edges):
    bins=np.arange(edges[0]+2,edges[-1]-2+1e-9,1.)
    density=np.histogram(radial,bins)[0]/shell_volumes(bins) if len(bins)>1 else []
    return layering_summary(density)


def particle_diagnostics(structure, center, edges, target, step_block=None,
                         element_sizes=(.5,1.0), reference_summary=None, *, transforms=None):
    sizes=mesh_sizes(element_sizes);n=len(structure.atoms);codes=residue_codes(structure)
    reach=2*structure.radii.max()+np.sqrt(3)*max(sizes)
    pairs=cKDTree(structure.xyz).query_pairs(reach,output_type='ndarray')
    bonded=bond_keys(structure,pairs)
    overlap=0;near={h:0 for h in sizes};close={2.:0,2.5:0}
    radii=structure.radii
    for start in range(0,len(pairs),CHUNK):
        p=pairs[start:start+CHUNK]
        d=np.linalg.norm(structure.xyz[p[:,0]]-structure.xyz[p[:,1]],axis=1)
        gap=d-radii[p].sum(axis=1)
        overlap+=np.count_nonzero(gap<0)
        for h in sizes:near[h]+=np.count_nonzero((gap>=0)&(gap<np.sqrt(3)*h))
        nonbonded=nonbonded_mask(codes,p,bonded)
        for threshold in close:close[threshold]+=np.count_nonzero(nonbonded & (d<threshold))
    del pairs
    samples=nearest_nonbonded(structure,bonded)
    nearest={'-'.join(pair):{'count':len(values),'p01':float(np.percentile(values,1)) if len(values) else None,
                            'p50':float(np.percentile(values,50)) if len(values) else None}
             for pair,values in samples.items()}
    radial=np.linalg.norm(structure.xyz-center,axis=1)
    total=sum(np.asarray(p) for p in target.values()) if target is not None else None
    def expected(lo,hi):
        if total is None:return None
        if step_block:
            from .step import integral
            a,b=(step_block['generated_bounds'][k] for k in ('inner','outer'))
            volume=4*np.pi*integral(a,b,step_block['edge_width'],lower=lo,upper=hi)
            return float(step_block['density']['plateau_total']*volume)
        lower=np.maximum(edges[:-1],lo);upper=np.minimum(edges[1:],hi)
        return float(np.sum(total*4*np.pi/3*np.maximum(upper**3-lower**3,0)))
    def expected_density(mid):
        if total is None:return None
        if step_block:
            from .step import shape
            a,b=(step_block['generated_bounds'][k] for k in ('inner','outer'))
            return float(step_block['density']['plateau_total']*shape(mid,a,b,step_block['edge_width']))
        ix=int(np.clip(np.searchsorted(edges,mid,side='right')-1,0,len(total)-1))
        return float(total[ix])
    walls={}
    for wall,sign in (('inner',1),('outer',-1)):
        bound=edges[0] if sign==1 else edges[-1]
        counts={str(d):int(np.count_nonzero(np.abs(radial-bound)<d)) for d in (.01,.1,.5)}
        density=[];goal=[]
        for k in range(8):
            lo,hi=sorted([bound+sign*.5*k,bound+sign*.5*(k+1)])
            lo,hi=max(lo,edges[0]),min(hi,edges[-1])
            valid=hi>lo
            upper=(radial<=hi) if hi==edges[-1] else (radial<hi)
            density.append(float(np.count_nonzero((radial>=lo)&upper)/(4*np.pi/3*(hi**3-lo**3))) if valid else None)
            goal.append(expected_density((lo+hi)/2) if valid else None)
        lo,hi=sorted([bound,bound+sign*.1])
        lo01,hi01=sorted([bound,bound+sign*.01])
        walls[wall]={'radius_angstrom':float(bound),'atoms_within':counts,'density_0p5':density,'expected_0p5':goal,
                     'excess_ratio':density[0]/goal[0] if goal[0] else None,'expected_within_0p1':expected(lo,hi),'expected_within_0p01':expected(lo01,hi01)}
    def shell_density(lo,hi):
        if hi-lo<1:return np.array([]),np.array([])
        bins=np.arange(lo,hi+1e-9,1.)
        return np.histogram(radial,bins)[0]/shell_volumes(bins),(bins[:-1]+bins[1:])/2
    layering=radial_layering(radial,edges)
    if step_block:
        density,mids=shell_density(edges[0]+6,edges[-1]-6)
        goal=np.array([expected_density(mid) for mid in mids]);valid=goal>0
        ratio=density[valid]/goal[valid]
        layering['plateau_relative_sd']=float(ratio.std()/ratio.mean()) if len(ratio) and ratio.mean() else None
    drift=[r['rigid_fit_rmsd_angstrom'] for r in transforms] if transforms is not None else []
    result={'contacts':{'overlap_per_atom':float(2*overlap/n),
                       'near_gap_per_atom':{str(h):float(2*count/n) for h,count in near.items()},
                       'gap_threshold_angstrom':{str(h):float(np.sqrt(3)*h) for h in sizes}},
            'inter_residue':{'peptide_bonded_residue_pairs':len(bonded),'peptide_bonds_per_residue':len(bonded)/len(np.unique(codes)),
                             'close_contacts_per_atom':{str(t):float(2*c/n) for t,c in close.items()},'nearest_nonbonded':nearest},
            'residue_drift':{'median_angstrom':float(np.median(drift)) if drift else None,
                             'p95_angstrom':float(np.percentile(drift,95)) if drift else None},
            'walls':walls,'layering':layering}
    result['edge_concentration'] = edge_concentration(radial, edges)
    if target is not None:
        from .analysis import Target
        result['contact_zone'] = contact_zone(radial, Target(edges, target, None, step_block))
    else:
        result['contact_zone'] = None
    if reference_summary is not None:result['reference']=reference_summary
    return result
