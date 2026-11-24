"""Bounded residue-template recovery during pre-feasible geometric repair."""
import copy
import numpy as np

from .model import MastercapError


def rigid_fit(source, target):
    """Proper Kabsch fit using row-vector coordinates; preserve report convention."""
    source, target = np.asarray(source), np.asarray(target)
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source-a).T@(target-b))
    rotation = u@np.diag([1, 1, np.linalg.det(u@vt)])@vt
    fitted = (source-a)@rotation+b
    translation = b-a@rotation
    rmsd = float(np.sqrt(np.mean(np.sum((target-fitted)**2,axis=1))))
    return fitted, rotation, translation, rmsd


def support_shift(xyz, center, edges, spare=.01):
    """Smallest whole-template shift along its centroid radius; None if impossible."""
    offsets = xyz-center
    radial = np.linalg.norm(offsets,axis=1)
    if np.all((radial >= edges[0]) & (radial <= edges[-1])):
        return xyz.copy(), 0.
    inner, outer = edges[0]+spare, edges[-1]-spare
    if inner > outer:return None
    centroid = offsets.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-12:return None
    direction = centroid/norm
    projection = offsets@direction
    perpendicular = np.maximum(0.,np.sum(offsets**2,axis=1)-projection**2)
    if np.any(perpendicular > outer**2):return None
    radius = np.sqrt(np.maximum(0.,outer**2-perpendicular))
    lo, hi = float(np.max(-projection-radius)), float(np.min(-projection+radius))
    if lo > hi:return None
    inner_radius = np.sqrt(np.maximum(0.,inner**2-perpendicular))
    candidates = np.r_[np.clip(0.,lo,hi),lo,hi,(-projection-inner_radius),(-projection+inner_radius)]
    candidates = candidates[(candidates >= lo) & (candidates <= hi)]
    for shift in candidates[np.argsort(np.abs(candidates),kind='stable')]:
        fitted = xyz+shift*direction
        r = np.linalg.norm(fitted-center,axis=1)
        if np.all((r >= inner-1e-9) & (r <= outer+1e-9)):
            return fitted, float(shift)
    return None


class RepairRecovery:
    def __init__(self, metadata, mode='on', stuck_iterations=50, reset_limit=5):
        if mode not in ('on','off'):
            raise MastercapError('Repair residue reset must be on or off')
        if (isinstance(stuck_iterations,bool) or not isinstance(stuck_iterations,(int,np.integer))
                or stuck_iterations < 5 or isinstance(reset_limit,bool)
                or not isinstance(reset_limit,(int,np.integer)) or reset_limit < 1):
            raise MastercapError('Repair stuck iterations must be an integer >= 5; reset limit an integer >= 1')
        self.metadata = metadata
        size = max(metadata.groups,default=-1)+1
        self.disconnected_streak = np.zeros(size,dtype=int)
        self.clash_streak = np.zeros(size,dtype=int)
        self.resets = np.zeros(size,dtype=int)
        self.cap = max(1,int(.001*len(metadata.groups)))
        self.exhausted = set()
        self.report = {'mode':mode,'stuck_iterations':int(stuck_iterations),'reset_limit':int(reset_limit),
                       'events':[],'total_resets':0,'skipped_support':0,'reset_exhausted':[]}

    def apply(self, iteration, structure, local, reference, center, edges, rng, anchored_indices=(), progress=None):
        mask = np.zeros(len(structure.atoms),dtype=bool)
        if self.report['mode'] == 'off':return mask
        metadata = self.metadata
        disconnected = np.asarray(local['disconnected_residues'],dtype=int)
        active = np.zeros(len(self.resets),bool);active[disconnected] = True
        self.disconnected_streak[:] = np.where(active,self.disconnected_streak+1,0)
        pairs = local['pairs'][local['violations']]
        pair_residues = metadata.residues[pairs]
        active[:] = False
        active[np.unique(pair_residues)] = True
        self.clash_streak[:] = np.where(active,self.clash_streak+1,0)
        anchored_indices = np.asarray(anchored_indices,dtype=int)
        anchored = set(metadata.residues[anchored_indices].tolist())
        threshold = self.report['stuck_iterations']
        stuck = disconnected[self.disconnected_streak[disconnected] >= threshold]
        candidates = [(-int(self.disconnected_streak[r]),0,int(r),None) for r in stuck]
        persistent = np.max(self.clash_streak[pair_residues],axis=1) >= threshold
        for ra,rb in np.unique(np.sort(pair_residues[persistent],axis=1),axis=0):
            ra,rb = int(ra),int(rb)
            streak = max(self.clash_streak[ra],self.clash_streak[rb])
            if (ra in anchored) != (rb in anchored):
                move = rb if ra in anchored else ra
            else:
                move = min((ra,rb),key=lambda r:(len(metadata.groups[r]),-r))
            partner = rb if move == ra else ra
            candidates.append((-int(streak),1,move,partner))
        moved = set()
        for negative_streak, kind, residue, partner in sorted(set(candidates)):
            if len(moved) >= self.cap:break
            if residue in moved:continue
            if self.resets[residue] >= self.report['reset_limit']:
                self.exhausted.add(residue);continue
            group = metadata.groups[residue]
            original = structure.xyz[group].copy()
            if reference is None:
                raise MastercapError('A reference structure is required when a repair residue reset fires')
            try:
                source = np.array([reference.xyz[structure.atoms[i].origin['reference_atom_index']] for i in group])
            except (KeyError,IndexError) as exc:
                raise MastercapError(f'Repair reset residue {residue} lacks valid reference atom provenance') from exc
            fitted, _, _, rmsd = rigid_fit(source,original)
            state = None
            if kind:
                state = copy.deepcopy(rng.bit_generator.state)
                direction = original.mean(axis=0)-center
                direction /= max(np.linalg.norm(direction),1e-12)
                if not np.any(direction):direction = np.array([1.,0,0])
                basis = np.eye(3)[np.argmin(np.abs(direction))]
                tangent = np.cross(direction,basis);tangent /= np.linalg.norm(tangent)
                angle = rng.uniform(0,2*np.pi)
                fitted += .5*(np.cos(angle)*tangent+np.sin(angle)*np.cross(direction,tangent))
            anchors = np.flatnonzero(np.isin(group,anchored_indices))
            before_r = np.linalg.norm(original[anchors]-center,axis=1)
            radial_shift = 0.
            if len(anchors):
                index = anchors[0];offset = fitted[index]-center
                r = np.linalg.norm(offset)
                direction = offset/r if r > 1e-12 else np.array([1.,0,0])
                amount = before_r[0]-r
                fitted += amount*direction;radial_shift += abs(float(amount))
            supported = support_shift(fitted,center,edges)
            if supported is not None:
                fitted, amount = supported
                radial_shift += abs(amount)
            if (supported is None or (len(anchors) and
                    not np.allclose(np.linalg.norm(fitted[anchors]-center,axis=1),before_r,rtol=0,atol=1e-6))):
                self.report['skipped_support'] += 1
                if state is not None:rng.bit_generator.state = state
                continue
            structure.xyz[group] = fitted
            mask[group] = True;moved.add(residue)
            self.resets[residue] += 1
            if kind:
                self.clash_streak[residue] = self.clash_streak[partner] = 0
            else:self.disconnected_streak[residue] = 0
            if self.resets[residue] >= self.report['reset_limit']:self.exhausted.add(residue)
            event = {'iteration':int(iteration),'residue':residue,'resname':structure.atoms[group[0]].resname,
                     'kind':'unjam' if kind else 'disconnected','streak':-negative_streak,
                     'fit_rmsd_angstrom':rmsd,'radial_shift_angstrom':radial_shift,
                     'tangential_shift_angstrom':.5 if kind else 0.,'anchored':residue in anchored,
                     'partner_residue':partner}
            self.report['total_resets'] += 1
            if len(self.report['events']) < 200:self.report['events'].append(event)
            if progress:progress(f"Repair reset at iteration {iteration}: residue {residue} ({event['resname']}), kind {event['kind']}, streak {event['streak']}")
        self.report['reset_exhausted'] = sorted(self.exhausted)
        return mask
