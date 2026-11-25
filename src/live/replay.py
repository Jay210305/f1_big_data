"""Validate replay uploads and load the public Bahrain 2025 sample."""
from math import isfinite

import polars as pl

from src.ingest.config import ROOT
from src.models.optimizer import COMPOUNDS
from .state_updater import FUEL_ALPHA

BAHRAIN_GP = "Bahrain Grand Prix"
BAHRAIN_RACE = ROOT / "examples" / "bahrain_2025_ver.csv"


def validate_laps(payload, n_laps):
    if isinstance(payload, dict):
        payload = payload.get("laps")
    if not isinstance(payload, list) or not payload:
        raise ValueError("El archivo debe incluir una lista no vacía de vueltas.")

    previous = 0
    rows = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Registro {index}: se esperaba un objeto de vuelta.")
        try:
            lap = _integer(row["lap"])
            laptime = _number(row["laptime"])
            compound = row.get("compound")
            if compound is not None and (
                not isinstance(compound, str) or compound.upper() not in COMPOUNDS
            ):
                raise ValueError
            if not previous < lap <= n_laps or laptime <= 0:
                raise ValueError
            for field in ("stint", "cluster"):
                if row.get(field) is not None:
                    _integer(row[field])
            for field in ("track_temp_mean", "track_temp"):
                if row.get(field) is not None:
                    _number(row[field])
            if row.get("stint") is not None and _integer(row["stint"]) < 1:
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"Registro {index}: revisa el número y tiempo de vuelta, el compuesto "
                "y los valores opcionales. Las vueltas deben estar en orden y dentro de la carrera."
            ) from error
        rows.append(row)
        previous = lap
    return rows


def _number(value):
    if isinstance(value, bool) or not isfinite(number := float(value)):
        raise ValueError
    return number


def _integer(value):
    number = _number(value)
    integer = int(value)
    if integer != number:
        raise ValueError
    return integer


def bahrain_frame():
    return pl.read_csv(BAHRAIN_RACE, schema_overrides={"track_status": pl.String}).with_columns(
        pl.lit("2025").alias("season"),
        pl.lit(BAHRAIN_GP).alias("gp"),
        pl.lit("Race").alias("session"),
        pl.lit("VER").alias("driver"),
        (pl.col("laptime") + FUEL_ALPHA * pl.col("lap")).alias("fuel_corr"),
        pl.col("track_status").str.contains("[467]").cast(pl.Int32).alias("is_sc_lap"),
        pl.lit(-1).alias("cluster"),
        pl.lit(None, dtype=pl.Float64).alias("dirty_air_frac"),
        pl.lit(None, dtype=pl.Int64).alias("t_end_life"),
    ).with_columns(pl.col("fuel_corr").min().over("stint").alias("stint_best_pace"))


def reference_params():
    """Reduced dry demonstration parameters from PROJECT_OVERVIEW §9 and §17.3."""
    return {
        "beta": {"SOFT": .12461, "MEDIUM": .09198, "HARD": .05158},
        "sigma": {"SOFT": .16185, "MEDIUM": .13224, "HARD": .08019},
        "pace_offset": {"SOFT": -.3015, "MEDIUM": 0., "HARD": .1522},
        "temp_coef": .00671,
        "temp_mean": 35.7,
    }
