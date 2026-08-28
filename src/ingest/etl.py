"""Phase 2 — Streaming ETL: raw JSON -> Bronze/Silver Parquet lakehouse.

Memory-bounded: processes one session at a time (one worker per session),
parses each lap file individually, and flushes a single Parquet file per
(year, Grand Prix, session). Never holds more than one session in memory.

Layout produced (Hive-partitioned by season):
    data/bronze/telemetry/season=YYYY/<gp>__<session>.parquet
    data/silver/stint_features/season=YYYY/<gp>__<session>.parquet
    data/silver/race_events/weather/season=YYYY/<gp>__<session>.parquet
    data/silver/race_events/corners/season=YYYY/<gp>__<session>.parquet
    data/silver/race_events/rcm/season=YYYY/<gp>__<session>.parquet

CLI:
    python src/ingest/etl.py                       # all years, all sessions
    python src/ingest/etl.py --years 2025 --sessions Race --max-sessions 2
    python src/ingest/etl.py --no-bronze           # silver only (faster)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from multiprocessing import Pool, get_context
from pathlib import Path

import polars as pl

from . import config as C
from .schema import TELEMETRY_COLUMNS, dtype_of

_NULL_TYPES = {"float": pl.Float64, "int": pl.Int32,
               "bool": pl.Boolean, "str": pl.Utf8}


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return s or "x"


def _null_lit(field: str) -> pl.Expr:
    t = _NULL_TYPES.get(dtype_of(field), pl.Float64)
    return pl.lit(None).cast(t).alias(field)


def _cast_expr(field: str) -> pl.Expr:
    dt = dtype_of(field)
    if dt == "float":
        return pl.col(field).cast(pl.Float64, strict=False)
    if dt == "int":
        return pl.col(field).cast(pl.Int32, strict=False)
    if dt == "bool":
        return pl.col(field).cast(pl.Boolean, strict=False)
    return pl.col(field).cast(pl.Utf8, strict=False)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_telemetry(inner: dict, meta: dict) -> pl.DataFrame:
    arrays = {k: v for k, v in inner.items() if isinstance(v, list)}
    if not arrays:
        return pl.DataFrame()
    min_len = min(len(v) for v in arrays.values())
    arrays = {k: v[:min_len] for k, v in arrays.items()}
    df = pl.DataFrame(arrays, strict=False)
    for c in df.columns:
        df = df.with_columns(_cast_expr(c))
    for c in TELEMETRY_COLUMNS:
        if c not in df.columns:
            df = df.with_columns(_null_lit(c))
    df = df.with_columns([
        pl.lit(meta["season"]).cast(pl.Utf8).alias("season"),
        pl.lit(meta["gp"]).cast(pl.Utf8).alias("gp"),
        pl.lit(meta["session"]).cast(pl.Utf8).alias("session"),
        pl.lit(meta["driver"]).cast(pl.Utf8).alias("driver"),
        pl.lit(int(meta["lap"])).cast(pl.Int32).alias("lap"),
    ])
    return df.select(TELEMETRY_COLUMNS + ["season", "gp", "session", "driver", "lap"])


def _silver_exprs(cols: list[str], meta: dict) -> list[pl.Expr]:
    has = set(cols)
    E: list[pl.Expr] = [
        pl.lit(meta["season"]).cast(pl.Utf8).alias("season"),
        pl.lit(meta["gp"]).cast(pl.Utf8).alias("gp"),
        pl.lit(meta["session"]).cast(pl.Utf8).alias("session"),
        pl.lit(meta["driver"]).cast(pl.Utf8).alias("driver"),
        pl.lit(int(meta["lap"])).cast(pl.Int32).alias("lap"),
        pl.len().cast(pl.Int32).alias("n_samples"),
    ]

    def num(c, fn, alias, t=pl.Float64):
        E.append(fn(pl.col(c)).cast(t).alias(alias) if c in has
                 else pl.lit(None).cast(t).alias(alias))

    def frac(c, pred, alias):
        if c in has:
            E.append(pred(pl.col(c)).fill_null(False).mean().cast(pl.Float64).alias(alias))
        else:
            E.append(pl.lit(None).cast(pl.Float64).alias(alias))

    num("speed", lambda c: c.mean(), "speed_mean")
    num("speed", lambda c: c.max(), "speed_max")
    num("speed", lambda c: c.min(), "speed_min")
    num("speed", lambda c: c.std(), "speed_std")
    num("throttle", lambda c: c.mean(), "throttle_mean")
    frac("throttle", lambda c: c > 0, "throttle_active_frac")
    frac("brake", lambda c: c > 0, "brake_frac")
    frac("drs", lambda c: c > 0, "drs_frac")
    num("acc_x", lambda c: c.abs().mean(), "lat_g_mean")
    num("acc_y", lambda c: c.abs().mean(), "long_g_mean")
    num("acc_z", lambda c: c.abs().mean(), "vert_g_mean")
    frac("DistanceToDriverAhead", lambda c: c < 10, "dirty_air_frac")
    num("distance", lambda c: c.max(), "lap_distance")
    return E


def _load_laptimes(session_path: Path) -> pl.DataFrame | None:
    sl = session_path / "session_laptimes.json"
    if sl.exists():
        try:
            d = _load_json(sl)
        except Exception:
            d = None
        if d:
            cols = {k: v for k, v in d.items() if isinstance(v, list)}
            if not cols:
                return None
            min_len = min(len(v) for v in cols.values())
            cols = {k: v[:min_len] for k, v in cols.items()}
            df = pl.DataFrame(cols, strict=False)
            rename = {}
            if "drv" in df.columns:
                rename["drv"] = "driver"
            if "time" in df.columns:
                rename["time"] = "laptime"
            if rename:
                df = df.rename(rename)
            keep = ["driver", "lap", "laptime", "compound", "stint",
                    "life", "s1", "s2", "s3"]
            df = df.select([c for c in keep if c in df.columns])
            df = df.with_columns([pl.col(c).cast(pl.Int32, strict=False)
                                  for c in ["lap", "stint", "life"] if c in df.columns])
            df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False)
                                  for c in ["laptime", "s1", "s2", "s3"] if c in df.columns])
            if "driver" in df.columns:
                df = df.with_columns(pl.col("driver").cast(pl.Utf8))
            return df
    rows = []
    for d in session_path.iterdir():
        if not d.is_dir():
            continue
        lf = d / "laptimes.json"
        if not lf.exists():
            continue
        try:
            dd = _load_json(lf)
        except Exception:
            continue
        n = len(dd.get("lap", []))
        for i in range(n):
            rows.append({
                "driver": d.name,
                "lap": dd["lap"][i],
                "laptime": dd.get("time", [None] * n)[i],
                "compound": dd.get("compound", [None] * n)[i],
                "stint": dd.get("stint", [None] * n)[i],
                "life": dd.get("life", [None] * n)[i],
                "s1": dd.get("s1", [None] * n)[i],
                "s2": dd.get("s2", [None] * n)[i],
                "s3": dd.get("s3", [None] * n)[i],
            })
    if not rows:
        return None
    df = pl.DataFrame(rows, strict=False, infer_schema_length=len(rows))
    df = df.with_columns([pl.col(c).cast(pl.Int32, strict=False)
                          for c in ["lap", "stint", "life"]])
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False)
                          for c in ["laptime", "s1", "s2", "s3"]])
    df = df.with_columns(pl.col("driver").cast(pl.Utf8))
    return df


def _events_df(payload: dict, meta: dict) -> pl.DataFrame:
    cols = {k: v for k, v in payload.items() if isinstance(v, list)}
    if not cols:
        return pl.DataFrame()
    min_len = min(len(v) for v in cols.values())
    cols = {k: v[:min_len] for k, v in cols.items()}
    df = pl.DataFrame(cols, strict=False)
    for c in df.columns:
        df = df.with_columns(_cast_expr(c))
    df = df.with_columns([
        pl.lit(meta["season"]).cast(pl.Utf8).alias("season"),
        pl.lit(meta["gp"]).cast(pl.Utf8).alias("gp"),
        pl.lit(meta["session"]).cast(pl.Utf8).alias("session"),
    ])
    return df


def _lap_number(name: str) -> int | None:
    m = re.match(r"^(\d+)_tel\.json$", name)
    return int(m.group(1)) if m else None


def process_session(task: tuple) -> dict:
    year, gp, session, spath = task
    meta_base = {"season": year, "gp": gp, "session": session}
    slug = f"{_slug(gp)}__{_slug(session)}"
    stats = {"season": year, "gp": gp, "session": session,
             "laps": 0, "bronze_rows": 0, "silver_rows": 0,
             "weather": 0, "corners": 0, "rcm": 0, "errors": 0}

    bronze_dfs: list[pl.DataFrame] = []
    silver_rows: list[pl.DataFrame] = []

    for d in sorted(p for p in spath.iterdir() if p.is_dir()):
        driver = d.name
        tel_files = []
        for f in d.iterdir():
            if f.name.endswith("_tel.json"):
                lap = _lap_number(f.name)
                if lap is not None:
                    tel_files.append((lap, f))
        tel_files.sort()
        for lap, f in tel_files:
            try:
                data = _load_json(f)
                inner = data.get("tel", data)
            except Exception:
                stats["errors"] += 1
                continue
            if not isinstance(inner, dict) or not any(isinstance(v, list) for v in inner.values()):
                continue
            meta = {**meta_base, "driver": driver, "lap": lap}
            try:
                bdf = _normalize_telemetry(inner, meta)
                if bdf.height:
                    bronze_dfs.append(bdf)
                    stats["bronze_rows"] += bdf.height
                    silver_rows.append(bdf.select(_silver_exprs(bdf.columns, meta)))
                    stats["laps"] += 1
            except Exception:
                stats["errors"] += 1

    if bronze_dfs and C.WRITE_BRONZE:
        try:
            bdir = C.BRONZE_DIR / f"season={year}"
            bdir.mkdir(parents=True, exist_ok=True)
            pl.concat(bronze_dfs).write_parquet(
                bdir / f"{slug}.parquet", compression=C.COMPRESSION)
        except Exception:
            stats["errors"] += 1

    if silver_rows:
        try:
            sdf = pl.concat(silver_rows)
            ldf = _load_laptimes(spath)
            if ldf is not None and {"driver", "lap"}.issubset(sdf.columns) \
                    and {"driver", "lap"}.issubset(ldf.columns):
                sdf = sdf.join(ldf, on=["driver", "lap"], how="left")
            sdir = C.SILVER_DIR / "stint_features" / f"season={year}"
            sdir.mkdir(parents=True, exist_ok=True)
            sdf.write_parquet(sdir / f"{slug}.parquet", compression=C.COMPRESSION)
            stats["silver_rows"] = sdf.height
        except Exception:
            stats["errors"] += 1

    for fname, key, sub in [
        ("weather.json", "weather", "weather"),
        ("corners.json", "corners", "corners"),
        ("rcm.json", "rcm", "rcm"),
    ]:
        fpath = spath / fname
        if not fpath.exists():
            continue
        try:
            edf = _events_df(_load_json(fpath), meta_base)
            if edf.height:
                edir = C.SILVER_DIR / "race_events" / sub / f"season={year}"
                edir.mkdir(parents=True, exist_ok=True)
                edf.write_parquet(edir / f"{slug}.parquet", compression=C.COMPRESSION)
                stats[key] = edf.height
        except Exception:
            stats["errors"] += 1

    return stats


def _iter_sessions(years, sessions):
    for year in years:
        ydir = C.ROOT / year
        if not ydir.is_dir():
            continue
        for gp in sorted(ydir.iterdir()):
            if not gp.is_dir() or gp.name in C.NON_GP:
                continue
            for session in sorted(gp.iterdir()):
                if not session.is_dir():
                    continue
                if sessions and session.name not in sessions:
                    continue
                yield year, gp.name, session.name, session


FAILED_SESSIONS = [
    ("2023", "São Paulo Grand Prix", "Sprint Shootout"),
    ("2024", "Pre-Season Testing", "Practice 1"),
    ("2024", "Pre-Season Testing", "Practice 3"),
    ("2025", "Pre-Season Testing", "Practice 1"),
    ("2025", "Pre-Season Testing", "Practice 3"),
    ("2025", "Pre-Season Testing", "Practice 2"),
]


def _iter_failed():
    for year, gp, session in FAILED_SESSIONS:
        spath = C.ROOT / year / gp / session
        if spath.is_dir():
            yield year, gp, session, spath


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 ETL: JSON -> Parquet lakehouse")
    ap.add_argument("--years", nargs="*", default=C.YEARS)
    ap.add_argument("--sessions", type=str, default=None,
                    help="comma-separated session names, e.g. 'Race,Sprint Qualifying'")
    ap.add_argument("--no-bronze", action="store_true", help="skip raw telemetry (silver only)")
    ap.add_argument("--max-sessions", type=int, default=None, help="cap # sessions (quick test)")
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    ap.add_argument("--fix-failed", action="store_true",
                    help="re-ingest only the 6 sessions that errored in the full run")
    args = ap.parse_args()

    C.WRITE_BRONZE = not args.no_bronze
    C.SESSIONS = [s.strip() for s in args.sessions.split(",")] if args.sessions else None

    if args.fix_failed:
        tasks = list(_iter_failed())
    else:
        tasks = list(_iter_sessions(args.years, C.SESSIONS))
    if args.max_sessions:
        tasks = tasks[:args.max_sessions]
    print(f"Phase 2 ETL  |  sessions={len(tasks)}  workers={args.workers}  "
          f"bronze={C.WRITE_BRONZE}  compression={C.COMPRESSION}")
    print(f"  years={args.years}  sessions={args.sessions or 'ALL'}")
    print("-" * 70)

    t0 = time.time()
    ctx = get_context("spawn")
    agg = {"laps": 0, "bronze_rows": 0, "silver_rows": 0,
           "weather": 0, "corners": 0, "rcm": 0, "errors": 0, "sessions_done": 0}
    with ctx.Pool(args.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(process_session, tasks), 1):
            for k in ["laps", "bronze_rows", "silver_rows", "weather",
                      "corners", "rcm", "errors"]:
                agg[k] += st.get(k, 0)
            agg["sessions_done"] = i
            print(f"[{i:>4}/{len(tasks)}] {st['season']} {st['gp']} "
                  f"{st['session']}  laps={st['laps']} bronze={st['bronze_rows']} "
                  f"silver={st['silver_rows']} err={st['errors']}")
            if i % 20 == 0:
                el = time.time() - t0
                print(f"   ... {el:.0f}s elapsed, "
                      f"{agg['bronze_rows']:,} bronze rows so far")

    el = time.time() - t0
    print("-" * 70)
    print(f"DONE in {el:.0f}s")
    print(f"  sessions : {agg['sessions_done']}")
    print(f"  laps     : {agg['laps']:,}")
    print(f"  bronze   : {agg['bronze_rows']:,} rows")
    print(f"  silver   : {agg['silver_rows']:,} rows")
    print(f"  weather  : {agg['weather']:,}")
    print(f"  corners  : {agg['corners']:,}")
    print(f"  rcm      : {agg['rcm']:,}")
    print(f"  errors   : {agg['errors']:,}")

    report = {**agg, "elapsed_sec": el, "years": args.years,
              "sessions_filter": args.sessions, "bronze": C.WRITE_BRONZE}
    C.BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.ROOT / "data" / "bronze" / "phase2_etl_report.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report: data/bronze/phase2_etl_report.json")


if __name__ == "__main__":
    main()
