"""Versioned, pickle-free atom provenance sidecars and legacy inline loading."""
from dataclasses import dataclass, field
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np

from .model import MastercapError


def _column(values):
    valid=[v for v in values if v is not None]
    encoded=False
    if valid and all(isinstance(v,(bool,np.bool_)) for v in valid):
        array=np.array([False if v is None else v for v in values],dtype=bool)
    elif all(isinstance(v,(int,np.integer)) and not isinstance(v,(bool,np.bool_)) for v in valid):
        if any(not np.iinfo(np.int32).min <= int(v) <= np.iinfo(np.int32).max for v in valid):
            raise MastercapError('Atom-data integer exceeds int32 capacity')
        array=np.array([0 if v is None else v for v in values],dtype=np.int32)
    elif valid and all(isinstance(v,(int,float,np.number)) for v in valid):
        array=np.array([0 if v is None else v for v in values],dtype=np.float64)
    elif all(isinstance(v,str) for v in valid):
        array=np.array(['' if v is None else v for v in values],dtype=str)
    else:
        array=np.array([json.dumps(v,allow_nan=False) for v in values],dtype=str);encoded=True
    return array,encoded


def _write_table(arrays,rows,prefix,skip=(),aliases=None):
    aliases=aliases or {}
    keys=sorted(set().union(*(row.keys() for row in rows))-set(skip)) if rows else []
    arrays[prefix+'keys']=np.asarray(keys,dtype=str)
    for key in keys:
        name=aliases.get(key,prefix+key)
        array,encoded=_column([row.get(key) for row in rows])
        arrays[name]=array
        if encoded:arrays[name+'__json']=np.array(True)
        if any(row.get(key) is None for row in rows):arrays[name+'__none']=np.array([row.get(key) is None for row in rows])
        if any(key not in row for row in rows):arrays[name+'__present']=np.array([key in row for row in rows])


def _read_table(arrays,prefix,n,aliases=None):
    aliases=aliases or {};rows=[{} for _ in range(n)]
    for key in arrays.get(prefix+'keys',np.array([],dtype=str)).tolist():
        name=aliases.get(key,prefix+key);column=arrays[name]
        if column.shape != (n,):raise MastercapError(f'Invalid atom-data column {name}')
        for i,value in enumerate(column.tolist()):
            if name+'__present' in arrays and not arrays[name+'__present'][i]:continue
            if name+'__none' in arrays and arrays[name+'__none'][i]:value=None
            elif bool(arrays.get(name+'__json',False)):value=json.loads(value)
            rows[i][key]=value
    return rows


def encode_atom_data(mapping,transforms,initial,rigid_initial=None):
    arrays={'format_version':np.array(1,dtype=np.int32)}
    _write_table(arrays,mapping,'atom_',skip=('origin',),aliases={'id':'id','synthetic_residue':'synthetic_residue'})
    _write_table(arrays,[row['origin'] for row in mapping],'origin_',aliases={'reference_atom_index':'reference_atom_index'})
    for key in ('id','synthetic_residue','reference_atom_index'):
        arrays.setdefault(key,np.empty(0,dtype=np.int32))
    arrays['transforms_present']=np.array(transforms is not None)
    rows=transforms or []
    arrays.update(residue=np.array([r['residue'] for r in rows],dtype=np.int32),
                  rotation_row_vector=np.array([r['rotation_row_vector'] for r in rows],dtype=np.float64).reshape(-1,3,3),
                  translation=np.array([r['translation'] for r in rows],dtype=np.float64).reshape(-1,3),
                  rigid_fit_rmsd_angstrom=np.array([r['rigid_fit_rmsd_angstrom'] for r in rows],dtype=np.float64))
    _write_table(arrays,rows,'transform_extra_',skip=('residue','rotation_row_vector','translation','rigid_fit_rmsd_angstrom'))
    arrays['initial_present']=np.array(initial is not None)
    rows=initial or []
    arrays['initial_rotation']=np.array([r['rotation'] for r in rows],dtype=np.float64).reshape(-1,3,3)
    arrays['initial_translation']=np.array([r['translation'] for r in rows],dtype=np.float64).reshape(-1,3)
    _write_table(arrays,rows,'initial_extra_',skip=('rotation','translation'))
    if rigid_initial is not None:
        arrays['rigid_initial_transforms_json']=np.array([json.dumps(row,allow_nan=False) for row in rigid_initial],dtype=str)
    stream=io.BytesIO();np.savez_compressed(stream,**arrays)
    return stream.getvalue()


@dataclass
class AtomData:
    mapping: dict | None
    source: dict
    arrays: dict = field(default_factory=dict)
    inline_mapping: list | None = None
    inline_transforms: list | None = None
    inline_initial: list | None = None

    def mapping_records(self):
        if self.inline_mapping is not None:return self.inline_mapping
        if self.mapping is None:return None
        n=len(self.mapping['id'])
        rows=_read_table(self.arrays,'atom_',n,{'id':'id','synthetic_residue':'synthetic_residue'})
        origins=_read_table(self.arrays,'origin_',n,{'reference_atom_index':'reference_atom_index'})
        for row,origin in zip(rows,origins):row['origin']=origin
        return rows

    @property
    def transforms(self):
        if self.inline_mapping is not None:return self.inline_transforms
        if not bool(self.arrays.get('transforms_present',False)):return None
        n=len(self.arrays['residue']);rows=_read_table(self.arrays,'transform_extra_',n)
        for i,row in enumerate(rows):
            row.update(residue=int(self.arrays['residue'][i]),rotation_row_vector=self.arrays['rotation_row_vector'][i].tolist(),
                       translation=self.arrays['translation'][i].tolist(),rigid_fit_rmsd_angstrom=float(self.arrays['rigid_fit_rmsd_angstrom'][i]))
        return rows

    @property
    def initial(self):
        if self.inline_mapping is not None:return self.inline_initial
        if not bool(self.arrays.get('initial_present',False)):return None
        n=len(self.arrays['initial_rotation']);rows=_read_table(self.arrays,'initial_extra_',n)
        for i,row in enumerate(rows):row.update(rotation=self.arrays['initial_rotation'][i].tolist(),translation=self.arrays['initial_translation'][i].tolist())
        return rows


def store_atom_data(report,mode,sidecar):
    """Keep inline objects unchanged, or remove them and return staged NPZ bytes."""
    mapping=report.get('atom_mapping',[]);transforms=report.get('residue_transforms')
    initial=report.get('search',{}).get('initial_transforms')
    rmsd=np.array([r['rigid_fit_rmsd_angstrom'] for r in transforms or []])
    block={'mode':mode,'path':None,'sha256':None,'atoms':len(mapping),'residues':len(transforms or []),
           'residue_transforms_summary':{'median_rmsd':float(np.median(rmsd)) if len(rmsd) else None,
             'p95_rmsd':float(np.percentile(rmsd,95)) if len(rmsd) else None,'max_rmsd':float(rmsd.max()) if len(rmsd) else None},
           'initial_transforms':mode if initial is not None and mode != 'omit' else 'omitted'}
    files={}
    if mode=='sidecar':
        data=encode_atom_data(mapping,transforms,initial,report.get('rigid_search',{}).get('initial_transforms'))
        block.update(path=Path(sidecar).name,sha256=hashlib.sha256(data).hexdigest())
        files[Path(sidecar)]=data
    elif mode not in ('inline','omit'):raise MastercapError('Report atom data must be sidecar, inline or omit')
    if mode!='inline':
        report.pop('atom_mapping',None);report.pop('residue_transforms',None)
        report.get('search',{}).pop('initial_transforms',None)
        report.get('rigid_search',{}).pop('initial_transforms',None)
    report['atom_data']=block
    return files


def load_atom_data(report,report_path,override=None):
    report_path=Path(report_path);block=report.get('atom_data',{})
    if 'atom_mapping' in report:
        rows=report['atom_mapping']
        try:
            mapping={k:np.asarray([row['origin'][k] if k=='reference_atom_index' else row[k] for row in rows])
                     for k in ('id','synthetic_residue','reference_atom_index')}
            if any(a.dtype.kind not in 'iu' for a in mapping.values()):raise MastercapError('Inline atom mapping indices must be integers')
        except (KeyError,TypeError,ValueError) as exc:raise MastercapError(f'Invalid inline atom mapping: {exc}') from exc
        return AtomData(mapping,{'mode':'inline','path':str(report_path.resolve()),'sha256':None,'sha256_verified':False},
                        inline_mapping=rows,inline_transforms=report.get('residue_transforms'),
                        inline_initial=report.get('search',{}).get('initial_transforms'))
    path=Path(override) if override is not None else report_path.parent/block['path'] if block.get('path') else None
    if path is None and block.get('mode')=='omit':
        return AtomData(None,{'mode':'omit','path':None,'sha256':None,'sha256_verified':False})
    if path is None or not path.is_file():
        raise MastercapError(f'Missing atom-data sidecar: {path or "no path in report"}; provide --atom-data PATH')
    try:
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest();expected=block.get('sha256')
        if expected and digest != expected:raise MastercapError(f'Atom-data SHA-256 mismatch: {path}')
        if block.get('mode')=='sidecar' and not expected:raise MastercapError('Sidecar report has no atom-data SHA-256')
        with np.load(io.BytesIO(data),allow_pickle=False) as archive:
            arrays={key:archive[key] for key in archive.files}
        version=arrays['format_version']
        if version.shape!=() or version.dtype.kind not in 'iu' or int(version)!=1:raise MastercapError('Unsupported atom-data format_version')
        mapping={key:arrays[key] for key in ('id','synthetic_residue','reference_atom_index')}
        n=len(mapping['id'])
        if any(a.shape!=(n,) or a.dtype.kind not in 'iu' or np.any(a<0) for a in mapping.values()):
            raise MastercapError('Invalid atom-data mapping arrays')
        if block.get('atoms',n)!=n:raise MastercapError('Atom-data count differs from report')
        for present,keys in [('transforms_present',('residue','rotation_row_vector','translation','rigid_fit_rmsd_angstrom')),
                             ('initial_present',('initial_rotation','initial_translation'))]:
            if bool(arrays.get(present,False)):
                count=len(arrays[keys[0]])
                for key in keys:
                    shape=(count,3,3) if 'rotation' in key else (count,3) if 'translation' in key else (count,)
                    if arrays[key].shape!=shape or not np.isfinite(arrays[key]).all():
                        raise MastercapError(f'Invalid atom-data array {key}')
        return AtomData(mapping,{'mode':'sidecar','path':str(path.resolve()),'sha256':digest,
                                'sha256_verified':bool(expected)},arrays=arrays)
    except (OSError,ValueError,TypeError,KeyError,EOFError,zipfile.BadZipFile) as exc:
        if isinstance(exc,MastercapError):raise
        raise MastercapError(f'Invalid atom-data sidecar {path}: {exc}') from exc


def verify_provenance(data,particle,reference):
    if data.mapping is None:
        def signatures(structure):
            groups={}
            for atom in structure.atoms:groups.setdefault(atom.residue,[]).append(atom)
            return [(rows[0].resname,tuple(sorted((a.name,a.element) for a in rows))) for rows in groups.values()]
        allowed=set(signatures(reference))
        for i,signature in enumerate(signatures(particle)):
            if signature not in allowed:raise MastercapError(f'Residue signature does not match reference at residue {i}')
        return
    mapping=data.mapping
    if len(mapping['id'])!=len(particle.atoms):raise MastercapError('Atom mapping length does not match PDB')
    for i,atom in enumerate(particle.atoms):
        if mapping['id'][i]!=i or mapping['synthetic_residue'][i]!=atom.residue:
            raise MastercapError('PDB residue grouping does not match report')
        index=int(mapping['reference_atom_index'][i])
        if not 0 <= index < len(reference.atoms):raise MastercapError('Invalid reference_atom_index in atom data')
        origin=reference.atoms[index]
        if (atom.name,atom.element,atom.resname)!=(origin.name,origin.element,origin.resname):
            raise MastercapError('PDB atom identity does not match reference provenance')
