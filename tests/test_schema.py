"""Unit tests for the canonical schema normalizers (pure functions, no data)."""
from src.ingest import schema as S


def test_null_tokens():
    for tok in ["", "None", "none", "null", "N/A", "nan"]:
        assert S._is_null(tok)
    assert S._is_null(None)
    assert not S._is_null("SOFT")
    assert not S._is_null(0.0)


def test_to_float():
    assert S.to_float("12.5") == 12.5
    assert S.to_float(3) == 3.0
    assert S.to_float("None") is None
    assert S.to_float("") is None
    assert S.to_float("garbage") is None
    assert S.to_float(None) is None


def test_to_int():
    assert S.to_int("7") == 7
    assert S.to_int(7.9) == 7
    assert S.to_int("None") is None
    assert S.to_int("x") is None


def test_to_str():
    assert S.to_str("SOFT") == "SOFT"
    assert S.to_str(12) == "12"
    assert S.to_str("None") is None
    assert S.to_str(None) is None


def test_dtype_of():
    assert S.dtype_of("speed") == "float"
    assert S.dtype_of("lap") == "int"
    assert S.dtype_of("compound") == "str"
    assert S.dtype_of("pb") == "bool"
    assert S.dtype_of("wR") == "bool"
    # unknown channels default to float
    assert S.dtype_of("does_not_exist") == "float"


def test_telemetry_columns_present():
    for c in ["time", "rpm", "speed", "gear", "throttle", "brake", "drs",
              "distance", "DistanceToDriverAhead", "acc_x", "acc_y", "acc_z"]:
        assert c in S.TELEMETRY_COLUMNS
