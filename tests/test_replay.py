from io import BytesIO
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.streamlit_app import _load_rows
from src.live.demo import replay_decision
from src.live.replay import bahrain_frame
from src.mapreduce.evidence import polars_stints


@pytest.mark.parametrize("payload", [
    b"{", b"42", b'{"laps": "bad"}', b'[{"lap": 1, "laptime": NaN}]',
    b'[{"lap": 58, "laptime": 90}]',
    b'[{"lap": 2, "laptime": 90}, {"lap": 1, "laptime": 90}]',
])
def test_invalid_replay_has_readable_error(payload):
    with pytest.raises(ValueError, match="archivo|Registro"):
        _load_rows(BytesIO(payload), 57)


def test_real_bahrain_sample_produces_stints_and_recommendation():
    frame = bahrain_frame()
    assert frame.height == 57
    assert frame.filter(frame["is_sc_lap"] == 1)["lap"].to_list() == [32, 33, 34, 35]
    assert polars_stints(frame).height == 3
    decision = replay_decision(frame)
    assert decision["lap"] == 12
    assert decision["actual_strategy"] == "HARD->26 | MEDIUM->57"
    assert decision["rec_strategy"].endswith("->57")
    assert 0 < decision["p5"] <= decision["p95"]


def test_streamlit_shows_real_race_and_finishes_cleanly():
    script = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
    app = AppTest.from_file(script, default_timeout=15).run()
    assert not app.exception
    assert "12 / 57" in [metric.value for metric in app.metric]
    assert "Recomendación" in [heading.value for heading in app.subheader]
    app.slider[0].set_value(57).run()
    assert not app.exception
    assert "Carrera finalizada." in [message.value for message in app.info]
