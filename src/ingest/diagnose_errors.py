"""Diagnose the 6 ETL errors from the full Phase 2 run.

Re-runs ONLY the failing sessions in the main process with full tracebacks
per sub-step (bronze, silver, weather, corners, rcm) so we can see the real
exception instead of the swallowed `err=1`.
"""
from __future__ import annotations

import traceback
from pathlib import Path

import polars as pl

from src.ingest import etl
from src.ingest.config import ROOT

TARGETS = [
    ("2023", "São Paulo Grand Prix", "Sprint Shootout"),
    ("2024", "Pre-Season Testing", "Practice 1"),
    ("2024", "Pre-Season Testing", "Practice 3"),
    ("2025", "Pre-Season Testing", "Practice 1"),
    ("2025", "Pre-Season Testing", "Practice 3"),
    ("2025", "Pre-Season Testing", "Practice 2"),
]


def diag_one(year: str, gp: str, session: str) -> None:
    spath = ROOT / year / gp / session
    print("=" * 80)
    print(f"DIAG: {year} {gp} {session}")
    print(f"  path exists: {spath.exists()}")
    if not spath.exists():
        print("  -> session folder missing")
        return
    meta_base = {"season": year, "gp": gp, "session": session}

    # --- step A: parse telemetry (mirror process_session) ------------------
    silver_rows: list[pl.DataFrame] = []
    n_laps = 0
    for d in sorted(p for p in spath.iterdir() if p.is_dir()):
        for f in d.iterdir():
            lap = etl._lap_number(f.name)
            if lap is None:
                continue
            try:
                data = etl._load_json(f)
                inner = data.get("tel", data)
                bdf = etl._normalize_telemetry(
                    inner, {**meta_base, "driver": d.name, "lap": lap})
                if bdf.height:
                    silver_rows.append(
                        bdf.select(etl._silver_exprs(bdf.columns,
                                    {**meta_base, "driver": d.name, "lap": lap})))
                    n_laps += 1
            except Exception:
                print(f"  [step A ERROR] lap file {f}")
                traceback.print_exc()
    print(f"  step A (telemetry parse): laps={n_laps} silver_rows={len(silver_rows)}")

    # --- step B: silver concat + laptimes join + write ---------------------
    try:
        sdf = pl.concat(silver_rows)
        print(f"  step B concat ok: {sdf.shape}  cols={sdf.columns}")
    except Exception:
        print("  [step B concat ERROR]")
        traceback.print_exc()
        sdf = None

    if sdf is not None:
        try:
            ldf = etl._load_laptimes(spath)
            if ldf is None:
                print("  step B laptimes: None (no session_laptimes / driver laptimes)")
            else:
                print(f"  step B laptimes: {ldf.shape}  cols={ldf.columns}")
        except Exception:
            print("  [step B laptimes ERROR]")
            traceback.print_exc()
            ldf = None
        try:
            if ldf is not None and {"driver", "lap"}.issubset(sdf.columns) \
                    and {"driver", "lap"}.issubset(ldf.columns):
                joined = sdf.join(ldf, on=["driver", "lap"], how="left")
                print(f"  step B join ok: {joined.shape}")
        except Exception:
            print("  [step B join ERROR]")
            traceback.print_exc()

    # --- step C: race_events (weather / corners / rcm) ---------------------
    for fname, sub in [("weather.json", "weather"),
                       ("corners.json", "corners"),
                       ("rcm.json", "rcm")]:
        fpath = spath / fname
        if not fpath.exists():
            print(f"  step C {fname}: missing")
            continue
        try:
            payload = etl._load_json(fpath)
            edf = etl._events_df(payload, meta_base)
            print(f"  step C {fname}: rows={edf.height} cols={edf.columns}")
        except Exception:
            print(f"  [step C ERROR] {fname}")
            traceback.print_exc()


def main() -> None:
    for year, gp, session in TARGETS:
        diag_one(year, gp, session)
    print("\n" + "=" * 80)
    print("diagnosis complete")


if __name__ == "__main__":
    main()
