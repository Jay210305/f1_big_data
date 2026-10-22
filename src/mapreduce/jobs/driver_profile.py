"""Per-driver/session metrics on the labeled Race/Sprint population."""
from statistics import mean, median
from collections import defaultdict
from .stint_stats import KEYS, COMPOUNDS, reduce_stint
from ..types import JobSpec


def map_lap(record, ctx):
    columns = (*KEYS, "compound", "lap", "laptime", "fuel_corr", "life", "dirty_air_frac",
               "is_sc_lap", "speed_mean", "throttle_mean")
    return [(f"{record['driver']}|{record['session_type']}", {k: record.get(k) for k in columns})]


def reduce_driver(key, values):
    def avg(name):
        numbers = [r[name] for r in values if r.get(name) is not None]
        return mean(numbers) if numbers else None
    pace = [r["laptime"] for r in values if r.get("laptime") is not None]
    stints = defaultdict(list)
    for row in values:
        if row.get("compound") in COMPOUNDS:
            stints[tuple(row[k] for k in KEYS)].append(row)
    slopes = [result[0][1]["beta"] for key, rows in stints.items()
              if (result := reduce_stint(str(key), rows))]
    return [(key, dict(n_laps=len(values), mean_pace=mean(pace) if pace else None,
                      median_pace=median(pace) if pace else None,
                      degradation_slope=mean(slopes) if slopes else None, sc_lap_share=avg("is_sc_lap"),
                      dirty_air_share=avg("dirty_air_frac"), speed_mean=avg("speed_mean"),
                      throttle_mean=avg("throttle_mean")))]


def job(**kwargs):
    return JobSpec("driver_profile", map_lap, reduce_driver, **kwargs)
