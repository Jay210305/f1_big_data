"""Task 8.H.4 — Bronze schema-drift & data-quality validation.

Verifies the full Bronze telemetry lakehouse (seasons 2021-2025) for:
  1. Column availability — which canonical telemetry channels are present
     per season (schema drift detection).
  2. Type checks — each column's physical dtype vs its intended dtype
     (from schema.DTYPE_MAP).
  3. Unexpected null rates — per-column null fraction, with core columns
     expected to be near-complete and known-drift columns tracked
     separately.

    python -m src.ingest.verify_schema
"""
from __future__ import annotations

import json

import duckdb

from . import config as C
from .schema import DTYPE_MAP, TELEMETRY_COLUMNS

# Core channels that must exist in every season with negligible nulls.
CORE_COLUMNS = ["time", "rpm", "speed", "gear", "throttle", "brake",
                "drs", "distance", "x", "y", "z"]
# Channels known to drift across seasons (added/missing by year).
DRIFT_COLUMNS = ["rel_distance", "DriverAhead", "DistanceToDriverAhead",
                 "acc_x", "acc_y", "acc_z"]

NULL_RATE_WARN = 0.05   # flag core columns above this null fraction

# DuckDB DESCRIBE type names -> intended dtype (from schema.DTYPE_MAP).
DB_TO_INTENT = {
    "DOUBLE": "float", "FLOAT": "float",
    "INTEGER": "int", "INT32": "int", "INT64": "int", "BIGINT": "int",
    "SMALLINT": "int", "TINYINT": "int", "HUGEINT": "int",
    "BOOLEAN": "bool", "BOOL": "bool",
    "VARCHAR": "str", "STRING": "str", "LARGE_STRING": "str",
}


def _glob(season: str) -> str:
    return str(C.BRONZE_DIR / f"season={season}" / "**" / "*.parquet") \
        .replace("\\", "/")


def _season_analysis(con: duckdb.DuckDBPyConnection, season: str) -> dict:
    glob = _glob(season)
    # physical schema via DESCRIBE on a zero-row read (cheap)
    cols = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{glob}', hive_partitioning=1)"
    ).fetchall()
    name_to_type = {c[0]: str(c[1]).upper() for c in cols}
    present = set(name_to_type)

    missing = [c for c in TELEMETRY_COLUMNS if c not in present]
    type_ok = {}
    type_bad = {}
    for c in TELEMETRY_COLUMNS:
        if c not in present:
            continue
        intent = DTYPE_MAP.get(c, "float")
        actual = DB_TO_INTENT.get(name_to_type[c].split("(")[0], "str")
        if actual == intent:
            type_ok[c] = actual
        else:
            type_bad[c] = f"expected {intent}, got {actual}"

    null_rates = {}
    null_sql = " UNION ALL ".join(
        f"SELECT '{c}' AS col, "
        f"avg(({c} IS NULL)::int) AS null_frac "
        f"FROM read_parquet('{glob}', hive_partitioning=1)"
        for c in present & set(TELEMETRY_COLUMNS))
    if null_sql:
        for col, frac in con.execute(null_sql).fetchall():
            null_rates[col] = round(float(frac), 4)

    row_count = con.execute(
        f"SELECT count(*) FROM read_parquet('{glob}', hive_partitioning=1)"
    ).fetchone()[0]

    warnings = []
    for c in CORE_COLUMNS:
        if c in null_rates and null_rates[c] > NULL_RATE_WARN:
            warnings.append(f"{c} null_rate={null_rates[c]:.3f}")

    return {
        "season": season,
        "rows": row_count,
        "n_files": None,  # filled by caller
        "columns_present": len(present),
        "missing_canonical": missing,
        "type_mismatches": type_bad,
        "null_rates": null_rates,
        "warnings": warnings,
    }


def main() -> None:
    con = duckdb.connect()
    print("=" * 78)
    print("Bronze schema-drift & data-quality validation (2021-2025)")
    print("=" * 78)

    results = {}
    for season in C.YEARS:
        res = _season_analysis(con, season)
        n_files = sum(1 for _ in (C.BRONZE_DIR / f"season={season}")
                      .rglob("*.parquet"))
        res["n_files"] = n_files
        results[season] = res
        print(f"\nseason={season}  rows={res['rows']:,}  files={n_files}")
        if res["missing_canonical"]:
            print(f"  MISSING columns: {res['missing_canonical']}")
        else:
            print("  all canonical columns present")
        if res["type_mismatches"]:
            print("  TYPE mismatches:")
            for c, why in res["type_mismatches"].items():
                print(f"    {c}: {why}")
        else:
            print("  all dtypes match intent")
        if res["warnings"]:
            print("  HIGH null rate (core):")
            for w in res["warnings"]:
                print(f"    {w}")
        else:
            print("  core-column null rates within tolerance")

    # drift summary across seasons
    print("\n" + "=" * 78)
    print("Null-rate matrix (rows = canonical channel, cols = season):")
    all_cols = sorted({c for r in results.values() for c in r["null_rates"]})
    header = "  {:<22s}".format("channel") + "".join(
        f"{s:>8s}" for s in C.YEARS)
    print(header)
    for c in all_cols:
        row = "  {:<22s}".format(c)
        for s in C.YEARS:
            v = results[s]["null_rates"].get(c)
            if v is None:
                row += f"{'—':>8s}"
            else:
                row += f"{v:>8.3f}"
        print(row)

    out = C.BRONZE_DIR.parent / "phase2_schema_verify.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved -> {out}")

    n_missing = sum(len(r["missing_canonical"]) for r in results.values())
    n_type = sum(len(r["type_mismatches"]) for r in results.values())
    n_warn = sum(len(r["warnings"]) for r in results.values())
    print("=" * 78)
    print(f"SUMMARY: {n_missing} missing columns, {n_type} type mismatches, "
          f"{n_warn} null-rate warnings across {len(results)} seasons.")
    print("=" * 78)


if __name__ == "__main__":
    main()
