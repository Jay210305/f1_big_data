"""Canonical data schema for the F1 telemetry dataset.

Provisional manifest (Phase 1). It encodes structural facts already confirmed
by direct inspection; `explore.py` re-verifies presence across 2021-2025 and
this file is finalized from those results.

Key structural facts:
  * Telemetry is nested:  {"tel": { ...per-sample arrays..., "dataKey": <str> }}
    -> `dataKey` is a FILE-LEVEL scalar (not a per-sample array).
  * Per-driver `laptimes.json` is a flat dict-of-lists.
  * Session-level `session_laptimes.json` is a flat dict-of-lists that ALSO
    embeds weather columns (wAT..wWS), driver/team, and delta/pit info.
  * `weather.json`, `corners.json`, `rcm.json` are flat dicts-of-lists.
  * `drivers.json` wraps an array of objects under the key "drivers".

Nullable values are represented by the string "None" (and occasionally ""),
not JSON null — handled by the normalizers below.
"""
from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Envelope / structure constants
# ---------------------------------------------------------------------------
TELEMETRY_ENVELOPE = "tel"

# Per-sample channels inside the "tel" object (arrays, equal length).
TELEMETRY_COLUMNS: list[str] = [
    "time", "rpm", "speed", "gear", "throttle", "brake", "drs",
    "distance", "rel_distance", "DriverAhead", "DistanceToDriverAhead",
    "acc_x", "acc_y", "acc_z", "x", "y", "z",
]

# File-level scalar fields in the "tel" object (constants, not arrays).
TELEMETRY_SCALARS: list[str] = ["dataKey"]

# Per-driver laptimes.json.
LAPTIMES_COLUMNS: list[str] = [
    "time", "lap", "compound", "stint", "s1", "s2", "s3",
    "ms1", "ms2", "ms3", "life", "pos", "status", "pb",
]

# Session-level session_laptimes.json (superset incl. weather + pit deltas).
SESSION_LAPTIMES_COLUMNS: list[str] = [
    "time", "lap", "compound", "stint", "s1", "s2", "s3",
    "ms1", "ms2", "ms3", "life", "pos", "status", "pb",
    "sesT", "drv", "dNum", "pout", "pin", "s1T", "s2T", "s3T",
    "vi1", "vi2", "vfl", "vst", "fresh", "team",
    "lST", "lSD", "del", "delR", "ff1G", "iacc",
    "wT", "wAT", "wH", "wP", "wR", "wTT", "wWD", "wWS",
]

# weather.json channels.
WEATHER_COLUMNS: list[str] = ["wT", "wAT", "wH", "wP", "wR", "wTT", "wWD", "wWS"]

# corners.json channels.
CORNERS_COLUMNS: list[str] = ["CornerNumber", "X", "Y", "Angle", "Distance", "Rotation"]

# rcm.json (Race Control Messages) channels.
RCM_COLUMNS: list[str] = ["time", "cat", "msg", "status", "flag", "scope",
                          "sector", "dNum", "lap"]

# drivers.json object keys (wrapped under "drivers").
DRIVERS_COLUMNS: list[str] = ["driver", "team", "dn", "fn", "ln", "tc", "url"]

# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------
NULL_TOKENS: frozenset[str] = frozenset({"", "None", "none", "null", "N/A", "nan"})


def _is_null(x: Any) -> bool:
    return x is None or (isinstance(x, str) and x.strip() in NULL_TOKENS)


def to_float(x: Any) -> Optional[float]:
    if _is_null(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def to_int(x: Any) -> Optional[int]:
    f = to_float(x)
    return None if f is None else int(f)


def to_str(x: Any) -> Optional[str]:
    if _is_null(x):
        return None
    return str(x)


# Dtype intent per channel (used by Phase 2 ETL to build typed tables).
# "float"/"int"/"str" map to the normalizers above.
DTYPE_MAP: dict[str, str] = {
    # telemetry
    "time": "float", "rpm": "int", "speed": "float", "gear": "int",
    "throttle": "float", "brake": "float", "drs": "int",
    "distance": "float", "rel_distance": "float",
    "DriverAhead": "str", "DistanceToDriverAhead": "float",
    "acc_x": "float", "acc_y": "float", "acc_z": "float",
    "x": "float", "y": "float", "z": "float",
    # laptimes
    "lap": "int", "compound": "str", "stint": "int",
    "s1": "float", "s2": "float", "s3": "float",
    "life": "int", "pos": "int", "status": "str", "pb": "bool",
    # session_laptimes extra
    "sesT": "str", "drv": "str", "dNum": "int", "pout": "int", "pin": "int",
    "s1T": "float", "s2T": "float", "s3T": "float",
    "vi1": "float", "vi2": "float", "vfl": "float", "vst": "float",
    "fresh": "bool", "team": "str", "lST": "float", "lSD": "float",
    "del": "float", "delR": "float", "ff1G": "float", "iacc": "float",
    # weather
    "wT": "float", "wAT": "float", "wH": "float", "wP": "float",
    "wR": "bool", "wTT": "float", "wWD": "int", "wWS": "float",
    # corners
    "CornerNumber": "int", "Angle": "float", "Distance": "float", "Rotation": "float",
    # rcm
    "cat": "str", "msg": "str", "flag": "str", "scope": "str", "sector": "int",
}


def dtype_of(field: str) -> str:
    """Return the intended dtype for a channel, defaulting to 'float'."""
    return DTYPE_MAP.get(field, "float")
