"""Unit tests for the ETL pure helpers (no dataset required)."""
import polars as pl

from src.ingest import etl as E


def test_slug():
    assert E._slug("São Paulo Grand Prix") == "S_o_Paulo_Grand_Prix"
    assert E._slug("Race") == "Race"
    assert E._slug("!!!") == "x"
    assert E._slug("") == "x"


def test_lap_number():
    assert E._lap_number("12_tel.json") == 12
    assert E._lap_number("0_tel.json") == 0
    assert E._lap_number("lap_tel.json") is None
    assert E._lap_number("12.json") is None


def test_null_lit_produces_null_typed_column():
    expr = E._null_lit("speed")
    df = pl.DataFrame({"x": [1]}).select(expr)
    assert df.columns == ["speed"]
    assert df["speed"].dtype == pl.Float64
    assert df["speed"][0] is None


def test_cast_expr():
    expr = E._cast_expr("speed")
    df = pl.DataFrame({"speed": ["12.5", "None"]}).select(expr)
    assert df["speed"].dtype == pl.Float64
    assert df["speed"][0] == 12.5
    assert df["speed"][1] is None


def test_normalize_telemetry_shape():
    inner = {"speed": [1.0, 2.0, 3.0], "rpm": [10, 20, 30]}
    meta = {"season": "2025", "gp": "Test", "session": "Race",
            "driver": "VER", "lap": 3}
    df = E._normalize_telemetry(inner, meta)
    assert df.height == 3
    # all canonical columns present + partition keys
    for c in ["season", "gp", "session", "driver", "lap"]:
        assert c in df.columns
    assert df["lap"][0] == 3


def test_record_error_structured():
    stats = {"errors": 0, "error_log": []}
    E._record_error(stats, ValueError("boom"), "load_telemetry",
                    {"driver": "VER"})
    assert stats["errors"] == 1
    assert stats["error_log"][0]["type"] == "ValueError"
    assert stats["error_log"][0]["message"] == "boom"
    assert stats["error_log"][0]["stage"] == "load_telemetry"
