"""The repository path must preserve the unfiltered race length and state semantics."""
import polars as pl


class SessionRepository:
    def __init__(self, rows):
        self.rows = rows

    def lap_history(self, season, gp, session, driver):
        return [r for r in self.rows if r["driver"] == driver]

    def session_lap_count(self, season, gp, session):
        return 57


def test_state_repository_matches_parquet_for_five_decision_laps():
    from src.models.optimizer import get_state
    rows = [dict(season="2025", gp="Test", session="Race", driver="VER", lap=i,
        life=i, stint=1 if i < 5 else 2, compound="SOFT" if i < 5 else "MEDIUM",
        fuel_corr=100. + i * .1, pace_above_best=.1 * i, track_temp_mean=32., cluster=-1)
        for i in range(1, 8)]
    frame = pl.DataFrame(rows)
    lapf = pl.DataFrame([dict(season="2025", gp="Test", session="Race", lap=57)])
    for lap in (1, 3, 4, 5, 7):
        expected = get_state(frame, "2025", "Test", "Race", "VER", lap, lapf=lapf)
        actual = get_state(None, "2025", "Test", "Race", "VER", lap, repo=SessionRepository(rows))
        assert actual == expected


def test_runner_stages_preserve_model_dependencies():
    from src.run_all import STAGES
    assert STAGES.index("style") < STAGES.index("labeled") < STAGES.index("mapreduce") < STAGES.index("mongo")
    assert "backtest" not in STAGES
