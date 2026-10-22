"""Exact warm-up/SC-filtered OLS estimator used by fit_degradation.py."""
import math
from statistics import mean, median, stdev

from ..types import JobSpec

KEYS = ["season", "gp", "session", "driver", "stint"]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def map_lap(record, ctx):
    if record.get("compound") not in COMPOUNDS:
        return []
    columns = ("life", "lap", "fuel_corr", "compound", "is_sc_lap", "cluster",
               "stint_best_pace", "dirty_air_frac", "track_temp_mean", "t_end_life")
    return [("|".join(str(record[k]) for k in KEYS), {k: record.get(k) for k in columns})]


def slope(rows, x="life", y="fuel_corr"):
    pairs = [(float(r[x]), float(r[y])) for r in rows if r.get(x) is not None and r.get(y) is not None]
    all_x = [float(r[x]) for r in rows if r.get(x) is not None]
    if len(pairs) < 2:
        return None
    xm, ym = mean(p[0] for p in pairs), mean(p[1] for p in pairs)
    # Polars cov(x,y) uses complete pairs; var(x) retains x where y is null.
    # Their ddof=1 factors only cancel when those sample counts are equal.
    marginal_mean = mean(all_x)
    denominator = sum((a - marginal_mean) ** 2 for a in all_x)
    covariance_sum = sum((a - xm) * (b - ym) for a, b in pairs)
    return (covariance_sum / denominator * (len(all_x) - 1) / (len(pairs) - 1)
            if denominator else None)


def reduce_stint(key, values):
    rows = [r for r in sorted(values, key=lambda r: (r["life"] is not None, r["life"] or 0, r["lap"]))[3:]
            if r.get("is_sc_lap", 0) == 0]
    beta = slope(rows)
    if len(rows) < 4 or beta is None or not math.isfinite(beta):
        return []
    def avg(name):
        numbers = [r[name] for r in rows if r.get(name) is not None]
        return mean(numbers) if numbers else None
    return [(key, dict(beta=beta, n_laps=len(rows), compound=rows[0]["compound"],
        cluster=rows[0].get("cluster", -1), best_pace=rows[0].get("stint_best_pace"),
        mean_dirty_air_frac=avg("dirty_air_frac"), track_temp_mean=avg("track_temp_mean"),
        life0=min(r["life"] for r in rows if r["life"] is not None), t_end_life=rows[0].get("t_end_life")))]


def nearest_quantile(values, q):
    # Polars default quantile interpolation='nearest' rounds half upward.
    values = sorted(values)
    return values[math.floor((len(values) - 1) * q + .5)]


def rollup(stints):
    stints = list(stints)
    all_beta = [s["beta"] for s in stints]
    if not all_beta:
        raise ValueError("no eligible stints")
    result = dict(beta={}, sigma={}, n_stints=len(stints), beta_global=median(all_beta))
    for compound in COMPOUNDS:
        values = [s["beta"] for s in stints if s["compound"] == compound]
        result["beta"][compound] = median(values) if len(values) >= 5 else median(all_beta)
        result["sigma"][compound] = (max(.02, min(.5,
            (nearest_quantile(values, .75) - nearest_quantile(values, .25)) / 1.349))
            if len(values) >= 5 else ((stdev(all_beta) or .05) if len(all_beta) > 1 else .05))
    return result


def job(**kwargs):
    return JobSpec("stint_stats", map_lap, reduce_stint, **kwargs)
