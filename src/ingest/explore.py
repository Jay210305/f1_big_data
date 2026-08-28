"""Phase 1 — Data exploration & schema validation (bounded sample).

Walks the 2021-2025 tree (directory listings only — no file contents opened
except a bounded sample) and reports:

  1. Per-year structure: # GPs, session types, driver dirs, file counts.
  2. JSON envelope (is telemetry nested under "tel"?).
  3. Field presence per year for every file type (telemetry, laptimes,
     session_laptimes, weather, corners, rcm, drivers).
  4. Detected value type + a sample value per field.
  5. Null-token ("None" string) prevalence.

Run (venv active, from repo root):
    python src/ingest/explore.py

It also writes a machine-readable summary to `data/bronze/phase1_schema_report.json`.

Sampling caps are hard-coded so this never reads a full season into memory:
  - up to 2 GPs x 2 drivers x 2 laps per session-type per year (telemetry)
  - up to 15 driver laptimes per year
  - up to 12 of each session-level JSON per year
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YEARS = ["2021", "2022", "2023", "2024", "2025"]

# Season-level folders that are NOT Grands Prix.
NON_GP = {".git", ".github", ".devcontainer", "cache", "cache_joblib",
          "cache_preseason", "fastf1_cache", "__pycache__"}

SESSION_LEVEL_FILES = {"weather.json", "corners.json", "rcm.json",
                       "session_laptimes.json", "drivers.json"}

NULL_TOKENS = {"", "None", "none", "null", "N/A", "nan"}


def list_sessions(year_dir: Path):
    """Yield (gp_name, session_name, session_path) for a season folder."""
    for gp in sorted(year_dir.iterdir()):
        if not gp.is_dir() or gp.name in NON_GP:
            continue
        for session in sorted(gp.iterdir()):
            if session.is_dir():
                yield gp.name, session.name, session


def type_of(v):
    """Return a short type label + example for a JSON value."""
    if isinstance(v, bool):
        return "bool", str(v)
    if isinstance(v, int):
        return "int", str(v)
    if isinstance(v, float):
        return "float", str(v)
    if isinstance(v, str):
        return "str", repr(v[:30])
    if isinstance(v, list):
        return "list", f"len={len(v)}"
    if isinstance(v, dict):
        return "dict", f"keys={list(v.keys())[:6]}"
    return type(v).__name__, repr(v)


def first_element(v):
    """First non-null element of a list (for type inference)."""
    if not isinstance(v, list):
        return None
    for x in v:
        if x is None or (isinstance(x, str) and x in NULL_TOKENS):
            continue
        return x
    return None


def null_count(v):
    """Count null-ish entries in a list."""
    if not isinstance(v, list):
        return 0
    return sum(1 for x in v if x is None or (isinstance(x, str) and x in NULL_TOKENS))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------
# field -> set of years it appeared in, per file type
presence = defaultdict(lambda: defaultdict(set))
# field -> (type_label, example), per file type
field_type = defaultdict(dict)
# file-type -> field -> total null-token count across sampled files
null_counts = defaultdict(lambda: defaultdict(int))
# file-type -> total sampled list-lengths (to report sample sizes)
sample_sizes = defaultdict(int)
# structural counters
gp_per_year: dict[str, set] = defaultdict(set)
session_per_year: dict[str, Counter] = defaultdict(Counter)
driver_dirs_per_year: dict[str, int] = defaultdict(int)
file_counts: dict[str, Counter] = defaultdict(Counter)
envelope_ok: dict[str, int] = defaultdict(int)  # year -> tel files with "tel" key


def record_fields(ftype: str, obj: dict, year: str) -> None:
    """Record field presence/type/null info for a flat dict-of-lists."""
    for k, v in obj.items():
        presence[ftype][k].add(year)
        t, ex = type_of(v)
        if k not in field_type[ftype]:
            field_type[ftype][k] = (t, ex)
        null_counts[ftype][k] += null_count(v)
    sample_sizes[ftype] += 1


def record_telemetry(ftype: str, data: dict, year: str, path: Path) -> None:
    """Record telemetry, unwrapping the {"tel": {...}} envelope if present."""
    inner = data.get("tel", data)
    if "tel" in data:
        envelope_ok[year] += 1
    record_fields(ftype, inner, year)
    # dataKey is a scalar, handled by type_of above (not a list)


def count_files(session_path: Path) -> None:
    """Count files per session and driver dirs (listing names only)."""
    for entry in session_path.iterdir():
        if entry.is_dir():
            # driver dir
            tel = sum(1 for f in entry.iterdir() if f.name.endswith("_tel.json"))
            file_counts["telemetry"][entry.name] += tel
            if tel:
                driver_dirs_per_year[entry.name] += 0  # noop placeholder
            if (entry / "laptimes.json").exists():
                file_counts["laptimes"][entry.name] += 1
        elif entry.name in SESSION_LEVEL_FILES:
            file_counts[entry.name][entry.name] += 1


def main() -> None:
    print("=" * 78)
    print("Phase 1 — Data exploration & schema validation")
    print(f"Root: {ROOT}")
    print("=" * 78)

    # --- structure walk (directory listings) -------------------------------
    tel_samples: dict[str, Counter] = defaultdict(Counter)       # year -> session_type -> n
    lap_samples: dict[str, Counter] = defaultdict(Counter)
    sfile_samples: dict[str, Counter] = defaultdict(Counter)

    for year in YEARS:
        ydir = ROOT / year
        if not ydir.is_dir():
            print(f"[WARN] {year} not found at {ydir}")
            continue
        print(f"\n### {year}  ({ydir})")

        gp_set = set()
        driver_total = 0
        for gp, session_name, spath in list_sessions(ydir):
            gp_set.add(gp)
            session_per_year[year][session_name] += 1
            for entry in spath.iterdir():
                if entry.is_dir():
                    driver_total += 1

        gp_per_year[year] = gp_set
        driver_dirs_per_year[year] = driver_total
        print(f"  Grands Prix: {len(gp_set)}")
        print(f"  Driver dirs: {driver_total}")
        print(f"  Session types: {dict(session_per_year[year])}")

    # --- full file counts (single pass, names only) -------------------------
    print("\n" + "=" * 78)
    print("File counts (from directory listings, contents not opened)")
    print("=" * 78)
    tot_tel = 0
    tot_lap = 0
    for year in YEARS:
        ydir = ROOT / year
        if not ydir.is_dir():
            continue
        tel = lap = 0
        for _, _, spath in list_sessions(ydir):
            for entry in spath.iterdir():
                if entry.is_dir():
                    tel += sum(1 for f in entry.iterdir() if f.name.endswith("_tel.json"))
                    if (entry / "laptimes.json").exists():
                        lap += 1
        tot_tel += tel
        tot_lap += lap
        print(f"  {year}: telemetry={tel:,}  laptimes={lap:,}")
    print(f"  TOTAL: telemetry={tot_tel:,}  laptimes={tot_lap:,}")

    # --- bounded schema sampling -------------------------------------------
    print("\n" + "=" * 78)
    print("Schema sampling (bounded)")
    print("=" * 78)
    for year in YEARS:
        ydir = ROOT / year
        if not ydir.is_dir():
            continue
        # group sessions by type so Race / Quali / Sprint all get sampled
        sessions_by_type: dict[str, list[Path]] = defaultdict(list)
        for _, sname, spath in list_sessions(ydir):
            sessions_by_type[sname].append(spath)

        for stype, sessions in sorted(sessions_by_type.items()):
            for spath in sessions[:2]:  # up to 2 GPs per session type
                drivers = [d for d in spath.iterdir() if d.is_dir()][:2]

                # telemetry: up to 2 laps per driver
                for d in drivers:
                    laps = sorted(f for f in d.iterdir() if f.name.endswith("_tel.json"))[:2]
                    for lp in laps:
                        if tel_samples[year][stype] >= 4:
                            break
                        try:
                            data = load_json(lp)
                            record_telemetry("telemetry", data, year, lp)
                            tel_samples[year][stype] += 1
                        except Exception as exc:  # noqa: BLE001
                            print(f"  [ERR] {lp}: {exc}")

                # driver laptimes: 1 per driver
                for d in drivers:
                    if lap_samples[year][stype] >= 4:
                        break
                    lf = d / "laptimes.json"
                    if lf.exists():
                        try:
                            record_fields("laptimes", load_json(lf), year)
                            lap_samples[year][stype] += 1
                        except Exception as exc:  # noqa: BLE001
                            print(f"  [ERR] {lf}: {exc}")

            # session-level JSONs: 1 per session (bounded)
            for sf in SESSION_LEVEL_FILES:
                fpath = spath / sf
                if fpath.exists() and sfile_samples[year][sf] < 12:
                    try:
                        data = load_json(fpath)
                        if sf == "drivers.json":
                            # unwrap list-of-objects
                            objs = data.get("drivers", [])
                            merged: dict = {}
                            if objs:
                                for k in objs[0]:
                                    merged[k] = [o.get(k) for o in objs]
                            record_fields("drivers", merged, year)
                        else:
                            record_fields(sf.replace(".json", ""), data, year)
                        sfile_samples[year][sf] += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [ERR] {fpath}: {exc}")

    # --- report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FIELD PRESENCE / TYPES / NULLS  (Y = present that year)")
    print("=" * 78)
    order = ["telemetry", "laptimes", "session_laptimes", "weather",
             "corners", "rcm", "drivers"]
    for ftype in order:
        if ftype not in presence:
            continue
        print(f"\n[{ftype}]  sampled_files={sample_sizes[ftype]}")
        for field in sorted(presence[ftype]):
            yrs = "".join("Y" if y in presence[ftype][field] else "." for y in YEARS)
            t, ex = field_type[ftype].get(field, ("?", "?"))
            nnull = null_counts[ftype].get(field, 0)
            extra = f"  nulls={nnull}" if nnull else ""
            print(f"  {field:24s} [{yrs}]  {t:5s}  e.g. {ex}{extra}")

    # envelope check
    print("\n" + "=" * 78)
    print("TELEMETRY ENVELOPE")
    print("=" * 78)
    for year in YEARS:
        print(f"  {year}: files-with-'tel'-key={envelope_ok.get(year, 0)}")

    # write machine-readable summary
    out_dir = ROOT / "data" / "bronze"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "years": YEARS,
        "gp_per_year": {y: sorted(gp_per_year[y]) for y in YEARS},
        "session_per_year": {y: dict(session_per_year[y]) for y in YEARS},
        "field_presence": {ft: {f: sorted(v) for f, v in fields.items()}
                           for ft, fields in presence.items()},
        "field_type": {ft: dict(f) for ft, f in field_type.items()},
        "null_counts": {ft: dict(f) for ft, f in null_counts.items()},
    }
    outp = out_dir / "phase1_schema_report.json"
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nMachine-readable summary written to: {outp}")

    print("\n" + "=" * 78)
    print("Phase 1 exploration COMPLETE — review above, then fill the annotation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
