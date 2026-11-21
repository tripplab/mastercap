"""Profile tolerance syntax and coverage, independent of CLI and search."""
import math
from .model import MastercapError, VDW


def profile_tolerance_spec(spec):
    message=f"Invalid profile tolerance {spec!r}; use NUMBER or ELEMENT=NUMBER[,ELEMENT=NUMBER...][,*=NUMBER]"
    try:
        parts=spec.split(',')
        if len(parts)==1 and '=' not in spec:
            result={'*':float(spec)}
        else:
            result={}
            for part in parts:
                key,value=part.split('=')
                key=key.strip().upper()
                if key not in VDW and key!='*' or key in result:
                    raise ValueError
                result[key]=float(value)
        if not result or any(not math.isfinite(v) or v<0 for v in result.values()):
            raise ValueError
        return result
    except (ValueError,TypeError,AttributeError):
        raise MastercapError(message) from None


def normalize_tolerances(value):
    if value is None:
        return None
    if isinstance(value,dict):
        return profile_tolerance_spec(','.join(f'{e}={v}' for e,v in value.items()))
    return profile_tolerance_spec(str(value))


def check_coverage(value,elements,warning=None):
    elements=set(elements)
    normalized=normalize_tolerances(value)
    if normalized is None:
        return None
    for e in sorted(elements):
        if e not in normalized and '*' not in normalized:
            raise MastercapError(f'Profile tolerance has no value for element {e}; add {e}=... or *=...')
    for e in sorted(set(normalized)-set(elements)-{'*'}):
        if warning:
            warning(f'Profile tolerance element {e} is absent from the reference; ignoring it')
        normalized.pop(e)
    return normalized
