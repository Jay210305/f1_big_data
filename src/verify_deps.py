"""Task 8.H.3.1 — Benchmark installed environment against requirements.txt.

Reports each pinned requirement, the installed version, and whether it
satisfies the constraint. Useful for reproducibility audits and CI.

    python -m src.verify_deps
"""
from __future__ import annotations

import importlib.metadata as im
import re
from pathlib import Path

from packaging.version import Version
from packaging.requirements import Requirement

from src.ingest.config import ROOT

REQ = ROOT / "requirements.txt"
IGNORE = {"torch"}  # installed separately with CUDA wheels


def _parse_reqs(path: Path) -> list[Requirement]:
    reqs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if ";" in line:
            line = line.split(";")[0]
        line = re.sub(r"\s+#.*$", "", line).strip()
        if not line:
            continue
        reqs.append(Requirement(line))
    return reqs


def main() -> None:
    reqs = _parse_reqs(REQ)
    print("=" * 78)
    print("Dependency benchmark: requirements.txt vs installed environment")
    print("=" * 78)
    rows = []
    for req in reqs:
        name = req.name
        if name in IGNORE:
            rows.append((name, "-", "-", "skipped (CUDA, separate)"))
            continue
        try:
            installed = im.version(name)
        except im.PackageNotFoundError:
            rows.append((name, "MISSING", str(req.specifier), "FAIL"))
            continue
        spec = req.specifier
        ok = spec.contains(Version(installed)) if installed else False
        rows.append((name, installed, str(spec),
                     "ok" if ok else "MISMATCH"))
    for name, installed, spec, status in rows:
        print(f"  {name:16s} {installed:16s} {spec:20s} {status}")

    missing = [r for r in rows if r[3] == "MISSING"]
    mismatch = [r for r in rows if r[3] == "MISMATCH"]
    print("-" * 78)
    print(f"  {len(rows)} requirements | {len(missing)} missing | "
          f"{len(mismatch)} version mismatch")
    if missing or mismatch:
        raise SystemExit("Dependency drift detected — run `uv sync` or "
                         "`pip install -r requirements.txt`")
    print("  All pinned requirements satisfied.")
    print("=" * 78)


if __name__ == "__main__":
    main()
