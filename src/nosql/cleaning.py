"""BSON-safe coercion using the lakehouse's canonical normalizers."""
import math
from collections import Counter
from datetime import datetime
from src.ingest.schema import to_float, to_int, to_str, DTYPE_MAP

INTS = {"lap", "life", "stint", "cluster", "_pos", "is_sc_lap", "n_laps", "n_drivers",
        "n_sims", "target_rul", "cliff_within_3", "actual_stops", "rec_stops"}
FLOATS = {"laptime", "fuel_corr", "pace_above_best", "delta", "degradation_slope"}
BOOLS = {"had_cliff", "stop_match", "pb", "fresh", "wR"}
STRINGS = {"_id", "season", "gp", "session", "driver", "compound", "session_type", "scenario"}


def clean_document(document, counts=None):
    counts = counts if counts is not None else Counter()
    result = {}
    for key, value in document.items():
        original = value
        if isinstance(value, dict):
            value = clean_document(value, counts)
        elif isinstance(value, (list, tuple)):
            value = [clean_document({"value": v}, counts)["value"] for v in value]
        elif isinstance(value, datetime):
            pass
        elif isinstance(value, float) and not math.isfinite(value):
            value = None
            counts["nonfinite_to_null"] += 1
        elif key in STRINGS:
            value = to_str(value)
        elif key in BOOLS:
            value = None if to_str(value) is None else (
                value.strip().lower() in {"true", "1", "yes"} if isinstance(value, str) else bool(value))
        elif key in INTS or DTYPE_MAP.get(key) == "int":
            number = to_float(value)
            value = to_int(number) if number is not None and math.isfinite(number) else None
        elif key in FLOATS or DTYPE_MAP.get(key) == "float":
            value = to_float(value)
            if value is not None and not math.isfinite(value):
                value = None
                counts["nonfinite_to_null"] += 1
        elif isinstance(value, str):
            value = to_str(value)
        if type(value) is not type(original):
            counts[f"{type(original).__name__}_to_{type(value).__name__}"] += 1
        result[key] = value
    return result
