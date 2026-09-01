"""Phase 5 — Materialize per-lap micro-sector vectors from Bronze (Option B).

Transition spec action item 4: no Autoencoder on 325M raw rows. Instead, each
lap becomes a fixed-length vector: telemetry aggregated over 10 standardized
distance bins (by the normalized `rel_distance` channel) x 4 channels:

    ms{b}_latg_max  = max |acc_x|          (lateral G peak per bin)
    ms{b}_thr_mean  = avg throttle         (throttle application per bin)
    ms{b}_brk_frac  = avg brake            (braking fraction per bin)
    ms{b}_spd_mean  = avg speed            (speed profile per bin)

Cached to data/gold/micro_sectors.parquet (one-off Bronze scan per season).
"""
from __future__ import annotations

import duckdb
import polars as pl

from src.ingest.config import BRONZE_DIR, ROOT

GOLD = ROOT / "data" / "gold"
YEARS = ["2021", "2022", "2023", "2024", "2025"]

MS_SQL = """
SELECT season, gp, session, driver, lap,
       least(greatest(floor(10 * rel_distance), 0), 9) AS bin,
       max(abs(acc_x)) AS latg_max,
       avg(throttle)  AS thr_mean,
       avg(brake)     AS brk_frac,
       avg(speed)     AS spd_mean
FROM read_parquet('{glob}', hive_partitioning=1)
WHERE rel_distance IS NOT NULL
GROUP BY season, gp, session, driver, lap, bin
"""


def _normalize_names(df: pl.DataFrame) -> pl.DataFrame:
    ren = {}
    for c in df.columns:
        for val, short in [("latg_max", "latg"), ("thr_mean", "thr"),
                           ("brk_frac", "brk"), ("spd_mean", "spd")]:
            if c.startswith(val + "_"):
                b = c.split("_")[-1].split(".")[0]
                ren[c] = f"ms{b}_{short}"
    return df.rename(ren)


def build_micro_sectors(years: list[str] | None = None) -> pl.DataFrame:
    years = years or YEARS
    cache = GOLD / "micro_sectors.parquet"
    if cache.exists():
        df = _normalize_names(pl.read_parquet(cache))
        have = sorted(df["season"].unique().to_list())
        if all(y in have for y in years):
            return df.filter(pl.col("season").is_in(years))
    con = duckdb.connect()
    frames = []
    for y in years:
        glob = str(BRONZE_DIR / f"season={y}" / "**" / "*.parquet").replace("\\", "/")
        long = con.execute(MS_SQL.format(glob=glob)).fetchdf()
        long = pl.from_pandas(long).with_columns(pl.col("season").cast(pl.Utf8))
        wide = long.pivot(
            on="bin", index=["season", "gp", "session", "driver", "lap"],
            values=["latg_max", "thr_mean", "brk_frac", "spd_mean"],
            aggregate_function="first",
        )
        wide = _normalize_names(wide)
        frames.append(wide)
        print(f"   micro_sectors {y}: {wide.shape[0]:,} laps x {wide.shape[1]} cols")
    out = pl.concat(frames, how="diagonal_relaxed")
    out.write_parquet(cache, compression="zstd")
    print(f"   cached -> {cache}")
    return out


if __name__ == "__main__":
    df = build_micro_sectors()
    print(df.shape)
    print(df.head(3))
