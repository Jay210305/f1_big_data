"""Streamlit live timing dashboard.

Run with ``streamlit run app/streamlit_app.py``. Uploading a JSON lap stream
keeps the dashboard usable offline; the OpenF1 adapter is an optional source.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.models import optimizer
from src.live.state_updater import new_state

ROOT = Path(__file__).resolve().parents[1]


def _load_rows(upload):
    if upload is None:
        return []
    payload = json.loads(upload.getvalue().decode("utf-8"))
    return payload if isinstance(payload, list) else payload.get("laps", [])


def main():
    st.set_page_config(page_title="F1 Live Strategy", layout="wide")
    st.title("F1 Live Strategy")
    with st.sidebar:
        season = st.text_input("Season", "2025")
        gp = st.text_input("Grand Prix", "Live session")
        driver = st.text_input("Driver", "VER")
        n_laps = st.number_input("Race laps", min_value=1, value=57)
        compound = st.selectbox("Starting compound",
                                ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
                                index=1)
        upload = st.file_uploader("Lap stream JSON", type="json")
    rows = _load_rows(upload)
    state = new_state(season, gp, driver, n_laps, compound)
    snapshots = []
    for row in rows:
        snapshots.append(state.update_lap(row))
    if not snapshots:
        st.info("Carga un JSON de vueltas para iniciar el estado en vivo.")
        return
    current = snapshots[-1]
    left, mid, right = st.columns(3)
    left.metric("Lap", f"{current['cur_lap']} / {current['n_laps']}")
    mid.metric("Fuel-corrected pace", f"{current['fuel_corr']:.3f}s")
    right.metric("Degradation slope", f"{current['degradation_slope']:.4f}s/lap")
    st.subheader(f"{driver} | {current['compound']}")
    st.line_chart({"Pace above best": [s["pace_above_best"] for s in snapshots],
                   "Degradation slope": [s["degradation_slope"] for s in snapshots]})
    st.metric("Track temperature", f"{current['track_temp_mean']:.1f} C")
    st.metric("Compounds used", ", ".join(current["compounds_used"]))
    model_path = ROOT / "data" / "models" / "degradation_model.json"
    if model_path.exists():
        params = json.loads(model_path.read_text(encoding="utf-8"))
        scenarios = optimizer.load_scenarios(params)
        recommendations = optimizer.optimize(state.as_optimizer_state(),
                                             params, 150, scenarios)
        best = recommendations["clean_air"][0]
        st.subheader("Recommendation")
        st.write(best["strategy"])
        st.caption(f"{best['mean']:.1f}s expected | "
                   f"p5-p95 {best['p5']:.1f}-{best['p95']:.1f}s")
    else:
        st.info("Genera data/models/degradation_model.json para activar recomendaciones.")
    st.json(current)


if __name__ == "__main__":
    main()
