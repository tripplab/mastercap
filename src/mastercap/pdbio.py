"""Fixed-column PDB input and TER-segmented output."""
import gzip
import hashlib
import json
import logging
import re
import ssl
import string
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np

from .model import Atom, Structure, MastercapError, PROTEIN, VDW
from . import __version__

LOG = logging.getLogger("mastercap")


def content(path):
    path = Path(path)
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    try:
        return data, data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise MastercapError("Coordinate input must be ASCII PDB content") from exc


def download(identifier, cache_dir, refresh=False, *, no_check_certificate=False):
    if not re.fullmatch(r"[A-Za-z0-9]{4}", identifier):
        raise MastercapError("A VIPERdb identifier must contain four alphanumeric characters")
    identifier = identifier.lower()
    directory = Path(cache_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{identifier}_full.vdb.gz"
    receipt = directory / f"{identifier}_full.vdb.gz.download.json"
    url = f"https://viperdb.org/resources/OLIGOMERS/{identifier}_full.vdb.gz"
    cached_tls = None
    if target.exists() and receipt.exists():
        try:
            prior = json.loads(receipt.read_text())
            if prior.get("sha256") == hashlib.sha256(target.read_bytes()).hexdigest():
                cached_tls = prior.get("tls_certificate_verified")
        except (OSError, ValueError, AttributeError):
            pass  # Legacy/changed cache content has unknown TLS provenance.
    cache_reused = not refresh and target.exists() and not (cached_tls is False and not no_check_certificate)
    if not cache_reused:
        LOG.info("Downloading expanded coordinates from %s", url)
        temporary = None
        try:
            options = {}
            if no_check_certificate:
                LOG.warning("TLS certificate verification disabled for this VIPERdb download (--no-check-certificate)")
                options["context"] = ssl._create_unverified_context()
            with urlopen(Request(url, headers={"User-Agent": f"mastercap/{__version__}"}), timeout=60, **options) as response:
                data = response.read()
            text = gzip.decompress(data).decode("ascii")
            if not any(line.startswith("ATOM  ") for line in text.splitlines()):
                raise MastercapError("Downloaded file contains no ATOM records")
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            temporary.replace(target)
            cached_tls = not no_check_certificate
            receipt.write_text(json.dumps({"sha256": hashlib.sha256(data).hexdigest(),
                                           "tls_certificate_verified": cached_tls}))
        except Exception as exc:
            raise MastercapError(f"VIPERdb download failed for {url}: {exc}") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    else:
        LOG.info("Using cached VIPERdb coordinates: %s", target)
    content(target)  # Also verify existing cached gzip files.
    return target, {"url": url, "cache_path": str(target),
                    "cache_reused": cache_reused, "tls_certificate_verified": cached_tls,
                    "tls_verification_requested": not no_check_certificate,
                    "acquired_utc": datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat()}


def infer_element(name, resname):
    # Protein CA is alpha carbon, never calcium. Hydrogen digits are prefixes.
    clean = name.strip().lstrip("0123456789").upper()
    if resname == "MSE" and clean == "SE":
        return "SE"
    if resname == "SEC" and clean == "SE":
        return "SE"
    if clean and clean[0] in "HCNOSP":
        return clean[0]
    raise MastercapError(f"Cannot infer element for {resname}:{name!r}")


def read_pdb(path, model=1, generated=False):
    data, text = content(path)
    atoms, xyz, warnings = [], [], []
    current_model, segment = 1, 0
    previous_key, residue = None, -1
    explicit_models = False
    seen_models = set()
    excluded_hetatm = excluded_nonprotein = 0
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.ljust(80)
        record = line[:6].strip()
        if record == "MODEL":
            if generated:
                raise MastercapError("Generated PDB must not contain MODEL blocks")
            explicit_models = True
            try:
                current_model = int(line[10:14])
            except ValueError as exc:
                raise MastercapError(f"Invalid MODEL at line {number}") from exc
            seen_models.add(current_model)
            segment += 1
            previous_key = None
            continue
        if record == "ENDMDL":
            current_model = None
            continue
        if current_model != model:
            continue
        if record == "TER":
            segment += 1
            previous_key = None
            continue
        if record == "HETATM":
            excluded_hetatm += 1
            continue
        if record != "ATOM":
            continue
        resname = line[17:20].strip().upper()
        if not generated and resname not in PROTEIN:
            excluded_nonprotein += 1
            continue
        if line[16] != " ":
            raise MastercapError(f"Alternate position at line {number}: {line[12:20].strip()}")
        try:
            position = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
        except ValueError as exc:
            raise MastercapError(f"Invalid coordinates at line {number}") from exc
        if not np.isfinite(position).all():
            raise MastercapError(f"Nonfinite coordinates at line {number}")
        name = line[12:16].strip()
        element = line[76:78].strip().upper()
        if not element:
            element = infer_element(name, resname)
            warnings.append(f"Inferred {element} at line {number} ({resname}:{name})")
        if element not in VDW:
            raise MastercapError(f"No fixed vdW radius for {element!r} at line {number}")
        key = (segment, line[21], line[22:26], line[26], resname)
        if key != previous_key:
            residue += 1
            previous_key = key
        origin = {"line": number, "model": model, "segment": segment,
                  "chain": line[21].strip(), "residue_number": line[22:26].strip(),
                  "insertion_code": line[26].strip(), "residue_name": resname}
        atoms.append(Atom(name, element, resname, residue, origin))
        xyz.append(position)
    if explicit_models and model not in seen_models:
        raise MastercapError(f"Model {model} does not exist")
    if not explicit_models and model != 1:
        raise MastercapError("An input without MODEL records only has model 1")
    if not atoms:
        raise MastercapError("No accepted protein ATOM records in selected model")
    if excluded_hetatm:
        warnings.append(f"Ignored {excluded_hetatm} HETATM records")
    if excluded_nonprotein:
        warnings.append(f"Ignored {excluded_nonprotein} nonprotein or unrecognized ATOM records")
    return Structure(atoms, np.asarray(xyz), warnings,
                     {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                      "model": model, "radius_table": "Bondi 1964, doi:10.1021/j100785a001"})


def pdb_text(structure, segment_atom_limit=99998, segment_residue_limit=9999):
    if not 1 <= segment_atom_limit <= 99998 or not 1 <= segment_residue_limit <= 9999:
        raise MastercapError("Invalid PDB segment limits")
    rounded = np.round(structure.xyz, 3)
    if not np.isfinite(rounded).all() or any(len(f"{x:8.3f}") != 8 for x in rounded.flat):
        raise MastercapError("Coordinates exceed the PDB 8.3 field representation")
    lines = ["REMARK 900 MASTERCAP GEOMETRIC MODEL; NOT A FORCE-FIELD-READY PROTEIN",
             "REMARK 900 TER SEGMENTS MAY REUSE SERIALS AND CHAIN IDS; SEE JSON ORDER"]
    chains = string.ascii_uppercase + string.ascii_lowercase + string.digits
    segment = serial = resid = 0
    mapping = []
    groups = []
    for index, atom in enumerate(structure.atoms):
        if not groups or atom.residue != structure.atoms[groups[-1][-1]].residue:
            groups.append([])
        groups[-1].append(index)
    def terminate():
        lines.append(f"TER   {serial + 1:5d}      {last_resname:>3s} {chains[segment % len(chains)]}{resid:4d}")
    last_resname = "UNK"
    for indices in groups:
        if len(indices) > segment_atom_limit:
            raise MastercapError("A residue exceeds the PDB segment atom capacity")
        if serial and (serial + len(indices) > segment_atom_limit or resid == segment_residue_limit):
            terminate()
            segment += 1
            serial = resid = 0
        resid += 1
        for index in indices:
            atom = structure.atoms[index]
            serial += 1
            name = atom.name
            if len(name) > 4 or len(atom.resname) > 3:
                raise MastercapError("Atom or residue name exceeds PDB field width")
            aligned_name = f" {name:<3}" if len(atom.element) == 1 and len(name) < 4 else f"{name:<4}"
            x, y, z = rounded[index]
            lines.append(f"ATOM  {serial:5d} {aligned_name} {atom.resname:>3s} {chains[segment % len(chains)]}{resid:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {atom.element:>2s}  ")
            mapping.append({"id": index, "segment": segment, "serial": serial,
                            "chain": chains[segment % len(chains)], "residue_number": resid,
                            "synthetic_residue": atom.residue, "origin": atom.origin})
        last_resname = structure.atoms[indices[-1]].resname
    if serial:
        terminate()
    lines.append("END")
    return "\n".join(lines) + "\n", mapping
