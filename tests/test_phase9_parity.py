"""Regression guard for the independent, columnar reference estimator."""
import polars as pl
import pytest
import json


def test_reference_excludes_warmup_before_sc_mask():
    from src.mapreduce.evidence import polars_stints
    rows = [{"season": "2025", "gp": "Test", "session": "Race", "driver": "VER",
             "stint": 1, "life": i, "lap": i + 1, "fuel_corr": 100 + i * .2,
             "compound": "MEDIUM", "is_sc_lap": int(i == 3),
             "cluster": -1, "track_temp_mean": 30., "dirty_air_frac": .1} for i in range(8)]
    result = polars_stints(pl.DataFrame(rows))
    assert result.height == 1
    assert result["n_laps"][0] == 4
    assert result["beta"][0] == pytest.approx(.2)


def test_null_pace_uses_pairwise_covariance_but_marginal_life_variance():
    from src.mapreduce.evidence import polars_stints
    from src.mapreduce.jobs.stint_stats import reduce_stint
    rows = [dict(season="2025", gp="Test", session="Race", driver="VER", stint=1,
        life=i, lap=i+1, fuel_corr=None if i == 4 else 100+i*.2, compound="SOFT",
        is_sc_lap=0, cluster=0, dirty_air_frac=.1, track_temp_mean=30.) for i in range(8)]
    reference = polars_stints(pl.DataFrame(rows))["beta"][0]
    assert reference == pytest.approx(.2333333333333333)
    assert reduce_stint("test", rows)[0][1]["beta"] == pytest.approx(reference)


def test_silver_parity_rejects_missing_feature(tmp_path, monkeypatch):
    from src.mapreduce import evidence
    from src.mapreduce.io import write_jsonl
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    baseline = tmp_path / "data/silver/stint_features/season=2025"
    baseline.mkdir(parents=True)
    row = dict(season="2025", gp="Test", session="Race", driver="VER", lap=1, n_samples=3)
    pl.DataFrame([{**row, "speed_mean": 123.}]).write_parquet(baseline / "test.parquet")
    output = tmp_path / "output"
    output.mkdir()
    write_jsonl(output / "part-00000.jsonl", [("2025|Test|Race|VER|1", row)])
    with pytest.raises(AssertionError, match="missing Silver columns"):
        evidence.silver_parity(output)


@pytest.mark.parametrize("field", ["beta", "sigma"])
def test_saved_model_mismatch_fails_checkpoint(tmp_path, monkeypatch, field):
    from src.mapreduce import evidence
    from src.mapreduce.io import write_jsonl
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    rows = [dict(season="2025", gp="Test", session="Race", driver="VER", stint=stint,
        life=i, lap=i+1, compound="SOFT", fuel_corr=100+i*.2, is_sc_lap=0,
        cluster=0, dirty_air_frac=.1, track_temp_mean=30.) for stint in range(5) for i in range(8)]
    frame = pl.DataFrame(rows)
    reference = evidence.polars_stints(frame)
    model = dict(beta=dict(SOFT=.2, MEDIUM=.2, HARD=.2), sigma=dict(SOFT=.02, MEDIUM=.05, HARD=.05))
    model[field]["SOFT"] += 1
    models = tmp_path / "data/models"
    models.mkdir(parents=True)
    (models / "degradation_model.json").write_text(json.dumps(model))
    gold = tmp_path / "data/gold"
    gold.mkdir()
    reference.with_columns(pl.col("beta").alias("degradation_slope")).write_parquet(gold / "stint_degradation.parquet")
    output = tmp_path / "output"
    output.mkdir()
    write_jsonl(output / "part-00000.jsonl", [("|".join(str(r[k]) for k in evidence.KEYS), r)
                                             for r in reference.to_dicts()])
    with pytest.raises(AssertionError, match="saved dry model"):
        evidence.stint_parity(frame, output)
