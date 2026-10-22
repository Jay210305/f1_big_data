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


def _render_live():
    with st.sidebar:
        season = st.text_input("Temporada", "2025")
        gp = st.text_input("Gran Premio", "Sesión en vivo")
        driver = st.text_input("Piloto", "VER")
        n_laps = st.number_input("Vueltas de carrera", min_value=1, value=57)
        compound = st.selectbox("Compuesto inicial",
                                ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
                                index=1)
        upload = st.file_uploader("Secuencia de vueltas JSON", type="json")
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
    left.metric("Vuelta", f"{current['cur_lap']} / {current['n_laps']}")
    mid.metric("Ritmo corregido por combustible", f"{current['fuel_corr']:.3f}s")
    right.metric("Pendiente de degradación", f"{current['degradation_slope']:.4f}s/vuelta")
    st.subheader(f"{driver} | {current['compound']}")
    st.line_chart({"Ritmo sobre el mejor": [s["pace_above_best"] for s in snapshots],
                   "Pendiente de degradación": [s["degradation_slope"] for s in snapshots]})
    st.metric("Temperatura de pista", f"{current['track_temp_mean']:.1f} C")
    st.metric("Compuestos utilizados", ", ".join(current["compounds_used"]))
    model_path = ROOT / "data" / "models" / "degradation_model.json"
    if model_path.exists():
        params = json.loads(model_path.read_text(encoding="utf-8"))
        scenarios = optimizer.load_scenarios(params)
        recommendations = optimizer.optimize(state.as_optimizer_state(),
                                             params, 150, scenarios)
        best = recommendations["clean_air"][0]
        st.subheader("Recomendación")
        st.write(best["strategy"])
        st.caption(f"{best['mean']:.1f}s esperados | "
                   f"p5-p95 {best['p5']:.1f}-{best['p95']:.1f}s")
    else:
        st.info("Genera data/models/degradation_model.json para activar recomendaciones.")
    st.json(current)


def _render_history():
    import pandas as pd
    from pymongo.errors import PyMongoError
    from src.nosql.repository import Repository

    season_col, gp_col, driver_col, compound_col = st.columns(4)
    season = season_col.text_input("Temporada del historial", "2025")
    gp = gp_col.text_input("GP del historial", "")
    driver = driver_col.text_input("Piloto del historial", "")
    compound = compound_col.selectbox("Compuesto", ["Todos", "SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
    page = st.number_input("Página", min_value=1, value=1, step=1)
    filters = dict(season=season or None, gp=gp or None, driver=driver or None,
                   compound=None if compound == "Todos" else compound)
    db = None
    try:
        repo = Repository()
        db = repo.db
        count = repo.decision_count(**filters)
        st.caption(f"{count} decisiones guardadas · 25 por página")
        st.dataframe(repo.decisions(**filters, page=int(page) - 1, page_size=25), use_container_width=True)
        bins = repo.delta_histogram(**filters)
        if bins:
            chart = pd.DataFrame(bins)
            chart["Intervalo de delta (s)"] = chart.apply(lambda r: f"{r['bin_start']:g} a {r['bin_end']:g}", axis=1)
            st.bar_chart(chart.set_index("Intervalo de delta (s)")["count"])
        with st.expander("Estado de la colección y sus índices"):
            st.json(repo.collection_stats())
        st.caption("Delta positivo: la recomendación mejora el tiempo simulado de la estrategia real.")
    except PyMongoError:
        st.info("MongoDB no disponible — inicia el servidor o usa python -m src.nosql.cli health")
    finally:
        if db is not None:
            db.client.close()


def main():
    st.set_page_config(page_title="Estrategia F1", layout="wide")
    st.title("Estrategia F1")
    live, history = st.tabs(["Vivo", "Historial MongoDB"])
    with live:
        _render_live()
    with history:
        _render_history()


if __name__ == "__main__":
    main()
