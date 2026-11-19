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


def profile_png(edges, original, smoothed, generated=None):
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
        axis.plot(mid, values(original), label="Original (mapped)", alpha=0.65)
        axis.plot(mid, values(smoothed), label="Smoothed target")
        if generated:
            axis.plot(mid, values(generated), label="Generated")
        axis.set(title=name, xlabel="Radius (angstrom)", ylabel="Atoms / angstrom³")
        axis.legend()
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=150)
    return stream.getvalue()
