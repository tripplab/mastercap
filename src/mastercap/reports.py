"""Reports and staged output files."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .model import MastercapError


def encode_json(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def preflight(paths, overwrite=False, protected=()):
    resolved = [Path(p).resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise MastercapError("Output destinations overlap")
    protected = {Path(p).resolve() for p in protected}
    for path in resolved:
        if path in protected:
            raise MastercapError(f"Output would replace an input file: {path}")
        if path.exists() and not overwrite:
            raise MastercapError(f"Output exists; use --overwrite explicitly: {path}")


def publish(files, overwrite=False, protected=()):
    preflight(files, overwrite, protected)
    staged = []
    backups = {}
    promoted = []
    try:
        for path, data in files.items():
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data if isinstance(data, bytes) else data.encode())
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((temporary, path))
        # Per-file atomic replacement. All content is prepared before promotion.
        for temporary, path in staged:
            if overwrite:
                if path.exists():
                    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as placeholder:
                        backup = Path(placeholder.name)
                    backup.unlink()
                    os.link(path, backup)
                    backups[path] = backup
                os.replace(temporary, path)
            else:
                # Atomic no-clobber even when another process creates the file
                # after preflight. Both files are on the same filesystem.
                os.link(temporary, path)
                temporary.unlink()
            promoted.append(path)
    except BaseException:
        for path in reversed(promoted):
            if path in backups:
                os.replace(backups[path], path)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def profile_csv(edges, original, smoothed, generated=None):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["element", "inner_angstrom", "outer_angstrom", "original_atoms_per_A3",
                     "target_atoms_per_A3", "generated_atoms_per_A3"])
    for element in sorted(smoothed):
        for k in range(len(edges)-1):
            writer.writerow([element, edges[k], edges[k+1], original[element][k], smoothed[element][k],
                             generated[element][k] if generated else ""])
    return stream.getvalue().encode()


def profile_png(edges, original, smoothed, generated=None, *, smoothing_enabled=True):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(9, 2.5*(len(smoothed)+1)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(len(smoothed)+1, 1))
    mid = (np.asarray(edges[:-1])+np.asarray(edges[1:]))/2
    names = sorted(smoothed)+["Total"]
    for axis, name in zip(axes, names):
        def values(profiles):
            return sum(np.asarray(v) for v in profiles.values()) if name == "Total" else profiles[name]
        axis.plot(mid, values(original), label="Original (mapped)", alpha=0.65, color="C0")
        if smoothing_enabled:
            axis.plot(mid, values(smoothed), label="Smoothed target", color="C1")
        if generated:
            axis.plot(mid, values(generated), label="Generated", color="C2")
        axis.set(title=name, xlabel="Radius (angstrom)", ylabel="Atoms / angstrom³")
        axis.legend()
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=150)
    return stream.getvalue()


CALIBRATION_COLUMNS=['source','label','element','n_atoms','mean','sd','p05','p50','p95',
                     'min','max','multiple_of_counting','multiple_of_orientation','gap_closed','suggested_tolerance']


def calibration_csv(calibration):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=CALIBRATION_COLUMNS)
    writer.writeheader()
    names=sorted(calibration['element_counts'])+['Total']
    def row(source,label,e,values):
        n=sum(calibration['element_counts'].values()) if e=='Total' else calibration['element_counts'][e]
        return {'source':source,'label':label,'element':e,'n_atoms':n,**values}
    for source,values in calibration['levels'].items():
        for e in names:
            stats={'mean':values[e]['mean']} if source=='self' else values[e]
            record=row(source,'',e,stats)
            if source==calibration['suggested']['basis'] and e!='Total':
                record['suggested_tolerance']=calibration['suggested']['profile_tolerance'][e]
            writer.writerow(record)
    for sample in calibration['compare']:
        for e in names:
            # CSV real variation uses normalized shape; JSON also retains raw errors.
            record=row('compare',Path(sample['file']).name,e,{'mean':sample['shape'][e]})
            record['n_atoms']=sum(sample['element_counts'].values()) if e=='Total' else sample['element_counts'][e]
            writer.writerow(record)
    for sample in calibration['generated']:
        for e in names:
            record=row('generated',Path(sample['file']).name,e,{'mean':sample['errors'][e],
                 'multiple_of_counting':sample['multiples_counting'][e],
                 'multiple_of_orientation':sample['multiples_orientation'][e],'gap_closed':sample['gap_closed'][e]})
            record['n_atoms']=sum(sample['element_counts'].values()) if e=='Total' else sample['element_counts'][e]
            writer.writerow(record)
    return stream.getvalue().encode()


def calibration_png(calibration):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    names=sorted(calibration['element_counts'])+['Total']
    figure=Figure(figsize=(9,1.8*len(names)+.6),layout='constrained')
    FigureCanvasAgg(figure)
    axes=np.atleast_1d(figure.subplots(len(names),1))
    settings=calibration['settings']
    figure.suptitle(f"{settings['reference_filename']} | bins <= {settings['bin_width']:g} A | "
                   f"sigma {settings['smoothing_sigma']:g} A | {settings['replicates']} replicates")
    for axis,e in zip(axes,names):
        values=[calibration['levels'][level][e][k] for level in ('counting','orientation','placement') for k in ('p05','p95','mean')]
        values += [s['shape'][e] for s in calibration['compare']]+[s['errors'][e] for s in calibration['generated']]
        tolerance=calibration['suggested']['profile_tolerance'].get(e)
        positive=[v for v in values+([tolerance] if tolerance is not None else []) if v>0]
        zero=max(min(positive)/10,1e-12) if positive else 1e-6
        def plot_value(v):return max(v,zero)
        for k,level in enumerate(('counting','orientation','placement')):
            stats=calibration['levels'][level][e]
            axis.plot([plot_value(stats['p05']),plot_value(stats['p95'])],[k,k],linewidth=5,color=f'C{k}',solid_capstyle='round')
            axis.plot(plot_value(stats['mean']),k,'o',color=f'C{k}')
        for sample in calibration['compare']:
            axis.plot(plot_value(sample['shape'][e]),3,marker='D',markerfacecolor='none',color='C3')
        generated=calibration['generated']
        for sample in generated:
            value=plot_value(sample['errors'][e]);axis.plot(value,4,'^',color='C4')
            if len(generated)<=5:axis.annotate(Path(sample['file']).stem,(value,4),xytext=(3,3),textcoords='offset points',fontsize=7,rotation=15)
        if len(generated)>5:axis.plot([],[],'^',color='C4',label='Generated structures')
        if tolerance is not None:
            axis.axvline(plot_value(tolerance),linestyle='--',color='black',label=f'Tolerance {tolerance:.3g}')
        axis.set(xscale='log',title=e,xlabel='Normalized integrated profile error',
                 yticks=range(5),yticklabels=['Counting','Orientation','Placement','Real variation (shape)','Generated'],ylim=(-.6,4.9))
        if any(v<=0 for v in values) or tolerance==0:
            axis.text(.99,.98,f'Zero values shown at {zero:.1g}',transform=axis.transAxes,ha='right',va='top',fontsize=7)
        if axis.get_legend_handles_labels()[0]:axis.legend(fontsize=7,loc='lower right')
    stream=io.BytesIO();figure.savefig(stream,format='png',dpi=150)
    return stream.getvalue()
