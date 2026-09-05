"""Incremental race state for live strategy recommendations.

The updater stores only the current stint and running aggregates. It never
loads or scans the historical lakehouse, so it can be fed by any lap stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

FUEL_ALPHA = 0.03


@dataclass
class IncrementalState:
    season: str
    gp: str
    driver: str
    n_laps: int
    cluster: int = -1
    cur_lap: int = 0
    compound: str = "MEDIUM"
    stint: int = 1
    laps_into_stint: int = 0
    base_pace: float = 0.0
    fuel_corr: float = 0.0
    pace_above_best: float = 0.0
    track_temp_mean: float = 36.0
    in_cliff_now: bool = False
    compounds_used: list[str] = field(default_factory=list)
    best_pace: float | None = None
    _temp_sum: float = 0.0
    _temp_count: int = 0
    _x_sum: float = 0.0
    _y_sum: float = 0.0
    _xx_sum: float = 0.0
    _xy_sum: float = 0.0
    _pace_history: list[float] = field(default_factory=list)

    def as_optimizer_state(self) -> dict[str, Any]:
        return {
            "season": self.season, "gp": self.gp, "session": "Race",
            "driver": self.driver, "cur_lap": self.cur_lap,
            "n_laps": self.n_laps, "compound": self.compound,
            "stint": self.stint, "laps_into_stint": self.laps_into_stint,
            "base_pace": self.base_pace, "in_cliff_now": self.in_cliff_now,
            "temp": self.track_temp_mean, "cluster": self.cluster,
            "compounds_used": list(self.compounds_used),
        }

    def snapshot(self) -> dict[str, Any]:
        result = self.as_optimizer_state()
        result.update({"fuel_corr": self.fuel_corr,
                       "pace_above_best": self.pace_above_best,
                       "track_temp_mean": self.track_temp_mean,
                       "degradation_slope": self.degradation_slope})
        return result

    @property
    def degradation_slope(self) -> float:
        n = self.laps_into_stint
        denom = n * self._xx_sum - self._x_sum * self._x_sum
        if n < 2 or abs(denom) < 1e-9:
            return 0.0
        return (n * self._xy_sum - self._x_sum * self._y_sum) / denom

    def update_lap(self, lap: dict[str, Any]) -> dict[str, Any]:
        """Consume one normalized lap and return the new snapshot.

        Required keys are ``lap``, ``laptime`` and ``compound``. Weather and
        cluster values are optional, which matches the OpenF1 adapter payload.
        """
        lap_number = int(lap["lap"])
        compound = str(lap.get("compound") or self.compound).upper()
        if compound != self.compound:
            self.stint += 1
            self.laps_into_stint = 0
            self.best_pace = None
            self._x_sum = self._y_sum = self._xx_sum = self._xy_sum = 0.0
            self._pace_history.clear()
            self.compound = compound
            if compound not in self.compounds_used:
                self.compounds_used.append(compound)
        self.cur_lap = lap_number
        self.laps_into_stint += 1
        raw_time = float(lap["laptime"])
        self.fuel_corr = raw_time + FUEL_ALPHA * lap_number
        self.base_pace = self.fuel_corr
        self.best_pace = (self.fuel_corr if self.best_pace is None
                          else min(self.best_pace, self.fuel_corr))
        self.pace_above_best = self.fuel_corr - self.best_pace
        self._pace_history.append(self.pace_above_best)
        self.in_cliff_now = self.pace_above_best > 1.5
        x = float(self.laps_into_stint)
        y = self.fuel_corr
        self._x_sum += x
        self._y_sum += y
        self._xx_sum += x * x
        self._xy_sum += x * y
        temp = lap.get("track_temp_mean", lap.get("track_temp"))
        if temp is not None and isfinite(float(temp)):
            self._temp_sum += float(temp)
            self._temp_count += 1
            self.track_temp_mean = self._temp_sum / self._temp_count
        if lap.get("cluster") is not None:
            self.cluster = int(lap["cluster"])
        return self.snapshot()


def new_state(season: str, gp: str, driver: str, n_laps: int,
              compound: str = "MEDIUM", cluster: int = -1) -> IncrementalState:
    """Create an empty updater state suitable for ``optimizer.optimize``."""
    return IncrementalState(season=str(season), gp=gp, driver=driver,
                            n_laps=int(n_laps), compound=compound.upper(),
                            cluster=cluster, compounds_used=[compound.upper()])
