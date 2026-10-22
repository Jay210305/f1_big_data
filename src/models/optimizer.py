"""Phase 6 — Prescriptive Dynamic Pit-Window & Stint Optimizer.

Components:
  1. Degradation model (data/models/degradation_model.json, fitted by
     fit_degradation.py): beta per compound (+style/temp), sigma, empirical
     cliff-lap distributions, compound pace offsets.
  2. Monte Carlo strategy evaluator: analytic stint times (no lap loops),
     N simulations per candidate, pit loss + warm-up per stop.
  3. Candidate search: 0/1/2-stop grids over pit laps x compounds, with the
     F1 two-compound rule enforced.
  4. Scenario context: clean_air / drs_train / backmarker multipliers on
     degradation rate and pace.
  5. In-race Pit Monitor (backtest on real stints): Phase 4 RUL (XGBoost) +
     cliff probability (XGB+LGB ensemble) per lap, with the transition-spec
     rules: debouncing (>=2 consecutive laps above threshold) and the joint
     cliff/RUL anomaly rule (cliff alert + RUL > 15 => thermal anomaly, no pit).

CLI:
    python -m src.models.optimizer                       # demo: 2025 Bahrain VER
    python -m src.models.optimizer --gp "Spanish Grand Prix" --driver HAM --lap 20
    python -m src.models.optimizer --n-sims 1000 --pit-loss 21
    python -m src.models.optimizer --sc-hazard --pit-loss-sc 7
"""
from __future__ import annotations

import argparse
import json
import time
from typing import cast

import numpy as np
import polars as pl

from src.ingest.config import ROOT
from src.models.eval_rul import build_labeled_frame
from src.models.rul_data import STINT_KEYS
from src.models.style_data import WET_COMPOUNDS

MODELS_DIR = ROOT / "data" / "models"
DRY = ["SOFT", "MEDIUM", "HARD"]
COMPOUNDS = DRY + WET_COMPOUNDS

SCENARIOS = {
    "clean_air": {"beta_mult": 1.00, "pace_pen": 0.00},
    "drs_train": {"beta_mult": 1.15, "pace_pen": 0.25},
    "backmarker": {"beta_mult": 1.25, "pace_pen": 0.50},
}
# traffic pace penalty (s/lap) per scenario — separate from degradation slope;
# the beta multipliers are now fitted empirically (Task 8.0.4 Option A) and
# overridden by load_scenarios() from degradation_model.json.
PACE_PEN = {"clean_air": 0.00, "drs_train": 0.25, "backmarker": 0.50}


def load_scenarios(params, include_sc=False, pit_loss_sc=None, circuit=None):
    """Merge empirically fitted scenario_beta_mult (if present) with the
    hand-set traffic pace penalty, yielding the runtime scenario table."""
    fitted = params.get("scenario_beta_mult", {})
    scenarios = {}
    for sname, pen in PACE_PEN.items():
        mult = float(fitted.get(sname, SCENARIOS[sname]["beta_mult"]))
        scenarios[sname] = {"beta_mult": mult, "pace_pen": pen}
    if include_sc:
        hazard_cfg = params.get("sc_hazard", {})
        if circuit is not None:
            hazard_cfg = {
                **hazard_cfg,
                "lap_hazard": hazard_cfg.get("lap_hazard", {}).get(circuit, {}),
            }
        scenarios["sc_hazard"] = {
            "beta_mult": 1.0,
            "pace_pen": 0.0,
            "pit_loss_sc": (float(pit_loss_sc) if pit_loss_sc is not None
                             else PIT_LOSS / 3.0),
            "sc_hazard": hazard_cfg,
        }
    return scenarios
PIT_LOSS = 21.0          # s lost per stop (circuit-dependent; CLI flag)
PIT_LOSS_SC = PIT_LOSS / 3.0  # discounted loss when stopping under SC/VSC
NEW_STINT_WARMUP = 2.0   # s lump per stop (pit exit + tyre warm-up)
CLIFF_BETA_MULT = 3.0    # degradation accelerates 3x after the cliff
JOINT_RUL_LIMIT = 15     # cliff alert + RUL > this => anomaly, no pit
RUL_IMMINENT = 2         # RUL <= this => pit now regardless


# ---------------------------------------------------------------------------
# Phase 4 model loading + per-lap inference
# ---------------------------------------------------------------------------
def load_lstm(meta):
    lstm_path = MODELS_DIR / "lstm_rul.pt"
    if not lstm_path.exists():
        return None
    try:
        import torch
        import torch.nn as nn
        feats = meta["features"]

        class RULNet(nn.Module):
            def __init__(self, f, h=96, l=2):
                super().__init__()
                self.emb_comp = nn.Embedding(9, 4)
                self.emb_st = nn.Embedding(4, 2)
                self.lstm = nn.LSTM(f - 2 + 4 + 2, h, num_layers=l,
                                    batch_first=True, dropout=0.4)
                self.drop = nn.Dropout(0.3)
                self.head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(),
                                          nn.Dropout(0.3), nn.Linear(64, 1))

            def forward(self, x):
                comp = x[..., feats.index("compound")].long()
                st = x[..., feats.index("session_type")].long()
                rest = [x[..., j] for j in range(x.shape[-1])
                        if feats[j] not in ("compound", "session_type")]
                z = torch.cat([self.emb_comp(comp), self.emb_st(st)]
                              + [r.unsqueeze(-1) for r in rest], dim=-1)
                out, _ = self.lstm(z)
                out = self.drop(out)
                return self.head(out).squeeze(-1)

        net = RULNet(len(feats))
        state = torch.load(str(lstm_path), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        return net
    except Exception as exc:
        print(f"[load_phase4] Note: LSTM load bypassed ({exc})")
        return None


def encode_sequence(df: pl.DataFrame, meta) -> np.ndarray:
    """Encodes a single stint into a normalized (1, L, F) tensor for the LSTM."""
    feats = meta["features"]
    codes = meta.get("codes", {})
    means = meta.get("norm_means", {})
    stds = meta.get("norm_stds", {})
    categorical = {"compound", "session_type", "gp", "driver"}

    df = df.sort("life")
    L = df.height
    mat = np.zeros((1, L, len(feats)), dtype=np.float32)

    df_enc = df.select(feats).clone()
    for c in categorical:
        if c in df_enc.columns and c in codes:
            df_enc = df_enc.with_columns(
                pl.col(c).fill_null("UNKNOWN").replace(codes[c]).cast(pl.Int32).alias(c))

    for j, f in enumerate(feats):
        if f in categorical:
            mat[0, :, j] = df_enc[f].to_numpy()
        else:
            col = df_enc[f].to_numpy().astype(np.float32)
            col = np.where(np.isfinite(col), col, 0.0)
            m = means.get(f, 0.0)
            s = stds.get(f, 1.0) or 1.0
            mat[0, :, j] = (col - m) / s
    return mat


def load_phase4():
    import xgboost as xgb
    import lightgbm as lgb
    with open(MODELS_DIR / "feature_meta.json") as fh:
        meta = json.load(fh)
    with open(MODELS_DIR / "phase4_metrics.json") as fh:
        m4 = json.load(fh)
    rul_m = xgb.Booster()
    rul_m.load_model(str(MODELS_DIR / "xgb_rul.json"))
    cliff_x = xgb.Booster()
    cliff_x.load_model(str(MODELS_DIR / "xgb_cliff.json"))
    cliff_l = lgb.Booster(model_file=str(MODELS_DIR / "lgb_cliff.txt"))
    thr = float(m4["xgb_cliff"]["threshold"])
    # Task 8.H.6: ensemble weights (validation-ROC-AUC proportional); fall
    # back to 50/50 if an older metrics file predates the weighted ensemble.
    weights = m4.get("xgb_cliff", {}).get("ensemble_weights",
                                          {"xgb": 0.5, "lgb": 0.5})
    lstm_m = load_lstm(meta)
    return meta, rul_m, cliff_x, cliff_l, thr, weights, lstm_m


def encode_frame(df: pl.DataFrame, meta) -> np.ndarray:
    X = df.select(meta["features"]).clone()
    for c, codes in meta["codes"].items():
        X = X.with_columns(pl.col(c).replace(codes).cast(pl.Int32).alias(c))
    return X.to_pandas().to_numpy(dtype=np.float32, na_value=np.nan)


def predict_laps(df: pl.DataFrame, meta, rul_m, cliff_x, cliff_l,
                 weights=(0.5, 0.5), lstm_m=None):
    import xgboost as xgb
    Xn = encode_frame(df, meta)
    rul_xgb = np.clip(rul_m.predict(xgb.DMatrix(Xn)), 0, None)
    if isinstance(weights, dict):
        w_x, w_l = float(weights.get("xgb", 0.5)), float(weights.get("lgb", 0.5))
    else:
        w_x, w_l = float(weights[0]), float(weights[1])
    prob = w_x * cliff_x.predict(xgb.DMatrix(Xn)) + w_l * cliff_l.predict(Xn)

    # Task 8.0.1 Option B: Piecewise continuous blending with LSTM in critical zone
    if lstm_m is not None and df.height > 0:
        import torch
        seq_mat = encode_sequence(df, meta)
        with torch.no_grad():
            rul_lstm = lstm_m(torch.from_numpy(seq_mat)).squeeze(0).numpy()
        rul_lstm = np.clip(rul_lstm, 0, None)
        # Smooth continuous transition:
        # rul_xgb <= 6: full LSTM (w=1.0)
        # rul_xgb >= 12: full XGBoost (w=0.0)
        # 6 < rul_xgb < 12: linear blend
        w = np.clip((12.0 - rul_xgb) / 6.0, 0.0, 1.0)
        rul = w * rul_lstm + (1.0 - w) * rul_xgb
    else:
        rul = rul_xgb

    return rul, prob


# ---------------------------------------------------------------------------
# Monte Carlo strategy simulation (analytic stint times)
# ---------------------------------------------------------------------------
def _sample_cliff(rng, cliff_cfg, n):
    """cliff_rem: laps until cliff (relative to stint/now start); inf if none."""
    out = np.full(n, np.inf)
    cen = cliff_cfg["censor_prob"]
    qs = np.array(cliff_cfg["quantiles"])
    wet = rng.random(n) < cen
    out[~wet] = qs[rng.integers(0, len(qs), size=int((~wet).sum()))]
    return out


def _sample_beta(rng, mu, sigma, n):
    """Beta sampler with a physical cap: the raw slope distribution has
    traffic-locked outliers, so uncapped samples + the lower clip would
    explode E[beta] far above the fitted median rate.

    The upper bound is derived from the *non-negative* part of the fitted
    mean. A negative fitted mean (wet compounds, where the rain term
    dominates the slope) must not push the upper clip below the physical
    lower bound: np.clip with lo > hi returns hi, which would silently
    produce negative degradation rates.
    """
    lo = 0.005
    hi = min(max(mu, 0.0) * 4.0 + 0.15, 1.0)
    return np.clip(rng.normal(mu, sigma, n), lo, hi)


def _sample_sc_windows(rng, hazard_cfg, n_sims, n_laps):
    """Sample one SC/VSC window per simulation using conditional lap hazards."""
    starts = np.full(n_sims, np.inf)
    hazard = hazard_cfg.get("lap_hazard", {})
    for lap in range(1, n_laps + 1):
        active = np.isinf(starts)
        p = float(hazard.get(str(lap), 0.0))
        if p > 0.0 and active.any():
            hit = active & (rng.random(n_sims) < p)
            starts[hit] = lap
    durations = np.ones(n_sims, dtype=float)
    choices = np.asarray(hazard_cfg.get("duration_laps", [1]), dtype=int)
    finite = np.isfinite(starts)
    if finite.any():
        durations[finite] = choices[rng.integers(0, len(choices), finite.sum())]
    return starts, durations


def _stop_is_under_sc(stop_laps, starts, durations):
    """Return a boolean matrix: simulation x stop, inclusive lap windows."""
    stops = np.asarray(stop_laps, dtype=float)
    return ((stops[None, :] >= starts[:, None])
            & (stops[None, :] < starts[:, None] + durations[:, None]))


def stint_time(base, pen, beta, L, cliff_rem, jump, slope):
    """Vectorized total seconds for one stint of L laps.

    lap k (k=1..L from stint/now start): base + pen + beta*k
    cliff at k > cliff_rem: one-time +jump at crossing, then +slope per lap.
    deg term = beta * L(L+1)/2 ; cliff = jump*m + slope*m(m+1)/2 with
    m = laps in cliff = max(0, L - max(cliff_rem, 0)).
    """
    deg = beta * L * (L + 1) / 2.0
    m = np.maximum(L - np.maximum(cliff_rem, 0.0), 0.0)
    cliff_cost = jump * m + slope * m * (m + 1) / 2.0
    return L * (base + pen) + deg + cliff_cost


def _elapsed_laps(state) -> int:
    """Laps of the current stint completed BEFORE the lap the car is on.

    ``laps_into_stint`` counts the laps of the current stint that have been
    run, including the lap the car is currently on. The empirical cliff
    distribution is indexed from the start of the stint, so the number of
    already-elapsed laps is one less.
    """
    return max(int(state.get("laps_into_stint", 0)) - 1, 0)


def simulate(strategy, state, params, scen, n_sims, rng, sc_windows=None):
    """Returns (n_sims,) total seconds for the remaining race."""
    totals = np.zeros(n_sims)
    jump = params.get("cliff_jump", 1.5)
    slope = params.get("cliff_slope", 0.15)
    scen_pen = scen["pace_pen"]
    for i, (comp, end_lap, is_current) in enumerate(strategy):
        L = end_lap - (state["cur_lap"] if is_current else
                       strategy[i - 1][1])
        L = max(int(L), 0)
        if L == 0:
            continue
        if is_current:
            base = state["base_pace"]
            beta_mu = (params.get("beta", {}).get(comp,
                                                   params.get("beta_global", 0.05))
                       + params["cluster_offset"].get(str(state["cluster"]), 0.0)
                       + params["temp_coef"] * (state["temp"] - params["temp_mean"]))
            sigma = params.get("sigma", {}).get(comp, 0.05)
            cliff_cfg = params.get("cliff", {}).get(
                comp, {"censor_prob": 1.0, "quantiles": [20.0]})
            beta = _sample_beta(rng, beta_mu, sigma, n_sims)
            cliff_rem = _sample_cliff(rng, cliff_cfg, n_sims) \
                - _elapsed_laps(state)
            if state["in_cliff_now"]:
                cliff_rem = np.minimum(cliff_rem, -1.0)
        else:
            pace_offset = params.get("pace_offset", {})
            base = (state["base_pace"]
                + pace_offset.get(comp, 0.0)
                - pace_offset.get(state["compound"], 0.0))
            beta_mu = (params.get("beta", {}).get(comp,
                                                   params.get("beta_global", 0.05))
                       + params["cluster_offset"].get(str(state["cluster"]), 0.0)
                       + params["temp_coef"] * (state["temp"] - params["temp_mean"]))
            sigma = params.get("sigma", {}).get(comp, 0.05)
            cliff_cfg = params.get("cliff", {}).get(
                comp, {"censor_prob": 1.0, "quantiles": [20.0]})
            beta = _sample_beta(rng, beta_mu, sigma, n_sims)
            cliff_rem = _sample_cliff(rng, cliff_cfg, n_sims)
        beta = beta * scen["beta_mult"]
        totals += stint_time(base, scen_pen, beta, L, cliff_rem, jump, slope)
    # The pit happens at the END of every stint that is followed by another
    # stint, so a strategy with k stints has exactly k-1 stops. The previous
    # implementation took the end lap of the *following* stint, which
    # (a) applied the SC/VSC pit-loss discount to the wrong lap and
    # (b) charged the race-finish lap as a pit stop.
    # NOTE: no lap filter here. The first pit can legitimately fall on the
    # current lap (the decision is taken at the pit), and dropping it would
    # charge the executed strategy one pit loss fewer than the candidate.
    stop_laps = [strategy[i][1] for i in range(len(strategy) - 1)]
    if stop_laps and sc_windows is not None:
        under_sc = _stop_is_under_sc(stop_laps, *sc_windows)
        pit_loss_sc = float(scen.get("pit_loss_sc", PIT_LOSS_SC))
        totals += np.where(under_sc, pit_loss_sc, PIT_LOSS).sum(axis=1)
        totals += len(stop_laps) * NEW_STINT_WARMUP
    else:
        totals += len(stop_laps) * (PIT_LOSS + NEW_STINT_WARMUP)
    return totals


def _allowed_compounds(state) -> list[str]:
    """Compound pool the strategy may use, given the weather context.

    A dry race cannot be run on wet tyres, so the dry pool is restricted to
    the three slicks. A wet race may use wet tyres and, as the track dries
    out, slicks.
    """
    if state.get("wet"):
        return list(WET_COMPOUNDS) + list(DRY)
    return list(DRY)


def enumerate_strategies(state):
    """Strategies as [(compound, end_lap, is_current), ...].

    Two physical rules constrain the candidate set:

    1. *Weather feasibility* — in a dry context only dry compounds may
       appear; wet tyres would be unusable on a dry track.
    2. *The F1 two-compound rule* — a dry race must be run on at least two
       different compounds, so a candidate that would finish the race on a
       single compound is rejected. The compounds already used earlier in the
       race count towards that total.
    """
    cur, n = state["cur_lap"], state["n_laps"]
    used = set(state["compounds_used"])
    pool = _allowed_compounds(state)
    dry_context = not state.get("wet")

    def _legal(compounds) -> bool:
        return not dry_context or len(used | set(compounds)) >= 2

    out = []
    for p1 in range(cur + 2, n - 2, 2):
        for c1 in pool:
            if not _legal([c1]):
                continue
            out.append([(state["compound"], p1, True), (c1, n, False)])
    for p1 in range(cur + 3, n - 10, 4):
        for p2 in range(p1 + 6, n - 2, 4):
            for c1 in pool:
                for c2 in pool:
                    if not _legal([c1, c2]):
                        continue
                    out.append([(state["compound"], p1, True),
                                (c1, p2, False), (c2, n, False)])
    # A zero-stop run is only legal when the compounds used so far already
    # satisfy the two-compound rule.
    if _legal([]):
        out.append([(state["compound"], n, True)])
    if not out:
        # Degenerate case: the remaining distance is too short for any legal
        # two-compound plan, so running the current compound to the flag is
        # the only physically possible option.
        out.append([(state["compound"], n, True)])
    if len(out) > 400:
        idx = np.linspace(0, len(out) - 1, 400).astype(int)
        out = [out[i] for i in idx]
    return out


def optimize(state, params, n_sims, scenarios):
    rng = np.random.default_rng(42)
    results = {}
    cands = enumerate_strategies(state)
    for sname, scen in scenarios.items():
        sc_windows = scen.get("_sc_windows")
        if sc_windows is None and scen.get("sc_hazard"):
            sc_windows = _sample_sc_windows(
                rng, scen["sc_hazard"], n_sims, state["n_laps"])
        rows = []
        for strat in cands:
            tot = simulate(strat, state, params, scen, n_sims, rng,
                           sc_windows=sc_windows)
            desc = " | ".join(
                f"{c}->{e}" for c, e, cur in strat)
            rows.append({
                "strategy": desc,
                "stops": sum(1 for c, e, cur in strat if not cur),
                "mean": float(tot.mean()), "std": float(tot.std()),
                "p5": float(np.quantile(tot, 0.05)),
                "p95": float(np.quantile(tot, 0.95)),
                "_strat": strat,
            })
        rows.sort(key=lambda r: r["mean"])
        best = rows[0]["mean"]
        for r in rows:
            r["delta_vs_best"] = r["mean"] - best
        results[sname] = rows
    return results


def branch_point_summary(results):
    """Summarize the green-flag and SC/VSC branches from ranked results."""
    green = results.get("clean_air", [])
    sc = results.get("sc_hazard", [])
    if not green or not sc:
        return None
    green_best = green[0]
    sc_best = sc[0]
    return {
        "green_flag": {"strategy": green_best["strategy"],
                       "mean": green_best["mean"],
                       "p5": green_best["p5"], "p95": green_best["p95"]},
        "sc_window": {"strategy": sc_best["strategy"],
                      "mean": sc_best["mean"],
                      "p5": sc_best["p5"], "p95": sc_best["p95"]},
        "sc_minus_green_mean": sc_best["mean"] - green_best["mean"],
    }


def _practice_session_order(session):
    """Prefer FP2, then the latest available practice session."""
    name = str(session).lower()
    if name == "practice 2":
        return 0
    if name.startswith("practice"):
        return 1
    return 2


def get_prerace_state(season, gp, driver, starting_compound):
    """Build a lap-zero state from the driver's longest practice stints."""
    if starting_compound not in COMPOUNDS:
        raise ValueError(f"unsupported starting compound: {starting_compound}")
    from src.models.rul_data import load_lap_features

    df = load_lap_features().with_columns(pl.col("season").cast(pl.Utf8))
    sub = df.filter(
        (pl.col("season") == str(season)) & (pl.col("gp") == gp)
        & (pl.col("driver") == driver)
        & pl.col("session").str.to_lowercase().str.starts_with("practice")
        & pl.col("compound").is_in(COMPOUNDS))
    if sub.height == 0:
        raise SystemExit(f"no practice data for {season} {gp} {driver}")
    session = sorted(sub["session"].unique().to_list(),
                     key=_practice_session_order)[0]
    sub = sub.filter(pl.col("session") == session)
    all_sub = df.filter(
        (pl.col("season") == str(season)) & (pl.col("gp") == gp)
        & (pl.col("session") == session)
        & pl.col("compound").is_in(COMPOUNDS))
    stints = (sub.group_by(["stint", "compound"])
              .agg(pl.len().alias("length"))
              .sort("length", descending=True))
    pace_rows = {}
    pace_sources = []
    for compound in COMPOUNDS:
        candidates = stints.filter(pl.col("compound") == compound)
        source = sub
        if candidates.height == 0:
            source = all_sub
            candidates = (source.group_by(["driver", "stint", "compound"])
                          .agg(pl.len().alias("length"))
                          .filter(pl.col("compound") == compound)
                          .sort("length", descending=True))
        if candidates.height == 0:
            continue
        stint = candidates["stint"][0]
        filters = [(pl.col("stint") == stint),
                   (pl.col("compound") == compound)]
        if "driver" in candidates.columns and source is all_sub:
            filters.append(pl.col("driver") == candidates["driver"][0])
        values = source.filter(filters[0] & filters[1] & filters[2]
                               if len(filters) == 3
                               else filters[0] & filters[1])["laptime"]
        values = values.drop_nulls()
        if values.len():
            median = values.median()
            if median is None:
                continue
            pace_rows[compound] = float(cast(float, median))
            pace_sources.append("driver" if source is sub else "field")
    if starting_compound not in pace_rows:
        raise SystemExit(f"no {starting_compound} practice stint for "
                         f"{season} {gp} {driver}")
    race = df.filter((pl.col("season") == str(season))
                     & (pl.col("gp") == gp) & (pl.col("session") == "Race"))
    if race.height == 0:
        raise SystemExit(f"no race distance for {season} {gp}")
    max_lap = race["lap"].max()
    if max_lap is None:
        raise SystemExit(f"race has no lap numbers for {season} {gp}")
    n_laps = int(cast(int, max_lap))
    temp_col = "track_temp_mean" if "track_temp_mean" in sub.columns else None
    temp = 36.0
    if temp_col:
        median_temp = sub[temp_col].median()
        if median_temp is not None:
            temp = float(cast(float, median_temp))
    cluster = -1
    embeddings = ROOT / "data" / "gold" / "driver_embeddings.parquet"
    if embeddings.exists():
        emb = pl.read_parquet(embeddings).with_columns(
            pl.col("season").cast(pl.Utf8))
        match = emb.filter((pl.col("season") == str(season))
                           & (pl.col("driver") == driver))
        if match.height and "cluster" in match.columns:
            cluster = int(match["cluster"][0])
    return {
        "season": str(season), "gp": gp, "session": "Race",
        "driver": driver, "cur_lap": 0, "n_laps": n_laps,
        "compound": starting_compound, "stint": 1, "laps_into_stint": 0,
        "base_pace": pace_rows[starting_compound], "in_cliff_now": False,
        "temp": temp, "cluster": cluster,
        "compounds_used": [starting_compound],
        "wet": starting_compound in WET_COMPOUNDS,
        "pace_source": f"{session} longest stints (median laptime; "
                   f"{', '.join(pace_sources)})",
        "practice_pace": pace_rows,
    }


def parse_strategy(text, state):
    """Parse ``COMPOUND->lap`` segments into simulate() tuples."""
    if not text or not text.strip():
        raise ValueError("strategy cannot be empty")
    result = []
    previous = state["cur_lap"]
    for index, raw in enumerate(text.split(",")):
        parts = raw.strip().upper().split("->")
        if len(parts) != 2 or parts[0] not in COMPOUNDS:
            raise ValueError(f"invalid strategy segment: {raw.strip()}")
        end = state["n_laps"] if parts[1].strip() == "END" else None
        if end is None:
            try:
                end = int(parts[1].strip())
            except ValueError as exc:
                raise ValueError(f"invalid end lap: {parts[1].strip()}") from exc
        if end <= previous or end > state["n_laps"]:
            raise ValueError(f"stint ends must increase between {state['cur_lap']} "
                             f"and {state['n_laps']}: {end}")
        result.append((parts[0], end, index == 0))
        previous = end
    if result[-1][1] != state["n_laps"]:
        raise ValueError("strategy must end at race distance (use ->end)")
    if result[0][0] != state["compound"]:
        raise ValueError("first segment must use the current compound")
    return result


def compare_strategies(strategy_a, strategy_b, state, params, scen,
                       n_sims, seed=42, sc_windows=None):
    """Evaluate two strategies with common random numbers."""
    totals = []
    for strategy in (strategy_a, strategy_b):
        totals.append(simulate(strategy, state, params, scen, n_sims,
                               np.random.default_rng(seed),
                               sc_windows=sc_windows))
    a, b = totals
    delta = a - b
    return {
        "a": a, "b": b, "delta": delta,
        "mean_delta": float(delta.mean()),
        "p5_delta": float(np.quantile(delta, 0.05)),
        "p95_delta": float(np.quantile(delta, 0.95)),
        "a_win_probability": float((delta < 0).mean()),
    }


def degradation_profile(strategy, state, params, scen):
    """Return the fitted degradation slope and lap count for each stint."""
    profile = []
    previous = state["cur_lap"]
    for compound, end_lap, is_current in strategy:
        laps = end_lap - previous
        beta = (params["beta"][compound]
                + params["cluster_offset"].get(str(state["cluster"]), 0.0)
                + params["temp_coef"] * (state["temp"] - params["temp_mean"]))
        profile.append({"compound": compound, "laps": laps,
                        "beta_per_lap": float(beta * scen["beta_mult"])})
        previous = end_lap
    return profile


# ---------------------------------------------------------------------------
# In-race Pit Monitor (transition-spec rules) — backtest on real stints
# ---------------------------------------------------------------------------
def run_monitor(df_driver, meta, rul_m, cliff_x, cliff_l, thr, label="",
                weights=(0.5, 0.5), lstm_m=None):
    """Applies debounced cliff + joint RUL rules lap by lap per stint."""
    blend_desc = "blended XGB+LSTM" if lstm_m is not None else "pure XGB"
    print(f"\n[pit-monitor] {label} (RUL engine: {blend_desc})")
    df_driver = df_driver.sort(STINT_KEYS + ["life"])
    for (key, grp) in df_driver.group_by(STINT_KEYS, maintain_order=True):
        season, gp, session, driver, stint = key
        grp = grp.sort("life")
        rul, prob = predict_laps(grp, meta, rul_m, cliff_x, cliff_l,
                                 weights=weights, lstm_m=lstm_m)
        laps = grp["lap"].to_numpy()
        life = grp["life"].to_numpy()
        n = len(laps)
        above = prob > thr
        trigger = None
        anomaly = None
        imminent = None
        deb = 0
        for i in range(n):
            if rul[i] <= RUL_IMMINENT:
                imminent = int(laps[i])
                break
            if above[i]:
                deb += 1
                if deb >= 2:
                    if rul[i] > JOINT_RUL_LIMIT:
                        anomaly = int(laps[i])
                        deb = 0
                    else:
                        trigger = int(laps[i])
                        break
            else:
                deb = 0
        actual_end = int(laps[-1])
        comp = grp["compound"][0]
        print(f"  stint {stint} ({comp}, {n} laps, laps {laps[0]}-{actual_end}): "
              f"pit_trigger={trigger if trigger else '-'} "
              f"anomaly_flag={anomaly if anomaly else '-'} "
              f"rul_imminent={imminent if imminent else '-'} "
              f"| actual stint end lap {actual_end}")


# ---------------------------------------------------------------------------
# Demo: real race state -> optimization + monitor backtest
# ---------------------------------------------------------------------------
def known_compounds(compounds):
    """The compounds that are actually known, sorted.

    A lap history read from MongoDB can carry a canonical BSON null for a
    compound the source could not resolve. An unknown tyre is not part of the
    compound set: it is dropped instead of reaching the sort that compares
    the compounds, which would fail on None.
    """
    return sorted(c for c in compounds if c is not None)


def get_state(df, season, gp, session, driver, lap, lapf=None, repo=None):
    if repo is not None:
        from pymongo.errors import ConnectionFailure
        try:
            history = repo.lap_history(str(season), gp, session, driver)
            if not history:
                raise ValueError(f"no MongoDB history for {season} {gp} {session} {driver}; run ingest")
            length = repo.session_lap_count(str(season), gp, session)
            session_frame = pl.DataFrame([dict(season=str(season), gp=gp,
                                               session=session, lap=length)])
            return get_state(pl.DataFrame(history, infer_schema_length=None), str(season),
                             gp, session, driver, lap, lapf=session_frame)
        except ConnectionFailure as exc:
            print(f"[source] MongoDB unavailable ({type(exc).__name__}); using parquet")
            if df is None:
                from .export_labeled import export_labeled
                df = pl.read_parquet(export_labeled())
                lapf = pl.read_parquet(ROOT / "data/gold/lap_features.parquet")
    sub = df.filter((pl.col("season") == season) & (pl.col("gp") == gp)
                    & (pl.col("session") == session)
                    & (pl.col("driver") == driver))
    if sub.height == 0:
        raise SystemExit(f"no data for {season} {gp} {session} {driver}")
    season = str(season)
    if lapf is not None:
        lapf = lapf.with_columns(pl.col("season").cast(pl.Utf8))
        n_laps = int(lapf.filter((pl.col("season") == season)
                                 & (pl.col("gp") == gp)
                                 & (pl.col("session") == session))["lap"].max())
    else:
        df = df.with_columns(pl.col("season").cast(pl.Utf8))
        n_laps = int(df.filter((pl.col("season") == season)
                               & (pl.col("gp") == gp)
                               & (pl.col("session") == session))["lap"].max())
    used = known_compounds(sub.filter(pl.col("lap") <= lap)["compound"].unique().to_list())
    rows = (sub.filter(pl.col("lap") == lap).sort("life").to_dicts())
    if not rows:
        raise SystemExit(f"lap {lap} not found for {driver} in {gp} {session}")
    row = rows[-1]
    stint = row["stint"]
    sst = sub.filter(pl.col("stint") == stint).sort("life")
    # laps_into_stint counts the laps of the current stint that have been
    # run, INCLUDING the lap the car is currently on. The zero-based index
    # of the current lap inside its stint is that count minus one.
    pos = int((sst["lap"] <= lap).sum())
    # weather context: rain already observed in this session, or a wet tyre
    # currently in use. Used to restrict the candidate compound pool.
    so_far = sub.filter(pl.col("lap") <= lap)
    rain = (float(so_far["rain_flag"].max() or 0.0)
            if "rain_flag" in so_far.columns else 0.0)
    wet = bool(rain > 0) or str(row["compound"] or "") in WET_COMPOUNDS
    return {
        "season": season, "gp": gp, "session": session, "driver": driver,
        "cur_lap": int(lap), "n_laps": n_laps, "compound": row["compound"],
        "stint": stint, "laps_into_stint": max(pos, 0),
        "base_pace": float(row["fuel_corr"]),
        "in_cliff_now": bool((row["pace_above_best"] or 0) > 1.5),
        "temp": float(row["track_temp_mean"] or 36.0),
        "cluster": int(row["cluster"]),
        "compounds_used": used,
        "wet": wet,
    }


def main():
    global PIT_LOSS, PIT_LOSS_SC
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp", default="Bahrain Grand Prix")
    ap.add_argument("--driver", default="VER")
    ap.add_argument("--season", default="2025")
    ap.add_argument("--lap", type=int, default=12)
    ap.add_argument("--source", choices=["parquet", "mongo"], default="parquet")
    ap.add_argument("--mongo-uri", default=None)
    ap.add_argument("--mongo-db", default=None)
    ap.add_argument("--pre-race", action="store_true",
                    help="seed lap-zero state from the driver's practice data")
    ap.add_argument("--starting-compound", choices=COMPOUNDS, default="MEDIUM")
    ap.add_argument("--compare", nargs=2, metavar=("STRATEGY_A", "STRATEGY_B"),
                    help="compare two plans, e.g. SOFT->18,MEDIUM->end")
    ap.add_argument("--n-sims", type=int, default=400)
    ap.add_argument("--pit-loss", type=float, default=PIT_LOSS)
    ap.add_argument("--sc-hazard", action="store_true",
                    help="enable empirical SC/VSC hazard Monte Carlo scenario")
    ap.add_argument("--pit-loss-sc", type=float, default=None,
                    help="pit loss under SC/VSC (default: pit-loss / 3)")
    args = ap.parse_args()
    PIT_LOSS = args.pit_loss
    PIT_LOSS_SC = (args.pit_loss_sc if args.pit_loss_sc is not None
                   else args.pit_loss / 3.0)
    t0 = time.time()

    with open(MODELS_DIR / "degradation_model.json") as fh:
        params = json.load(fh)
    if args.pre_race:
        state = get_prerace_state(args.season, args.gp, args.driver,
                                  args.starting_compound)
        df = None
        meta = rul_m = cliff_x = cliff_l = thr = weights = lstm_m = None
    else:
        meta, rul_m, cliff_x, cliff_l, thr, weights, lstm_m = load_phase4()
        repo = None
        if args.source == "mongo":
            from pymongo.errors import ConnectionFailure
            from src.nosql.connection import get_database
            from src.nosql.repository import Repository
            try:
                repo = Repository(get_database(args.mongo_uri, args.mongo_db))
                df = pl.DataFrame(repo.lap_history(args.season, args.gp, "Race", args.driver),
                                  infer_schema_length=None)
                state = get_state(None, args.season, args.gp, "Race", args.driver, args.lap, repo=repo)
                print("[source] mongo (labeled lap history + unfiltered session length)")
            except ConnectionFailure as exc:
                print(f"[source] MongoDB unavailable ({type(exc).__name__}); using parquet")
                repo = None
        if repo is None:
            from .export_labeled import export_labeled
            df = pl.read_parquet(export_labeled())
            state = get_state(df, args.season, args.gp, "Race", args.driver, args.lap,
                              lapf=pl.read_parquet(ROOT / "data/gold/lap_features.parquet"))
            print("[source] parquet")
    print(f"\n[state] {args.season} {args.gp} Race | {args.driver} @ lap "
          f"{state['cur_lap']}/{state['n_laps']}")
    print(f"        stint {state['stint']} on {state['compound']} "
          f"({state['laps_into_stint']} laps in, in_cliff={state['in_cliff_now']})")
    print(f"        track {state['temp']:.1f}C, style cluster {state['cluster']}, "
          f"compounds used so far: {state['compounds_used']}")

    scenarios = load_scenarios(params, include_sc=args.sc_hazard,
                               pit_loss_sc=args.pit_loss_sc,
                               circuit=state["gp"])
    if args.compare:
        strategy_a = parse_strategy(args.compare[0], state)
        strategy_b = parse_strategy(args.compare[1], state)
        paired = compare_strategies(strategy_a, strategy_b, state, params,
                                    scenarios["clean_air"], args.n_sims)
        print("\n[compare] A vs B (positive delta means A is slower)")
        print(f"  A: {args.compare[0]}")
        print(f"  B: {args.compare[1]}")
        print(f"  delta A-B: {paired['mean_delta']:+.2f}s "
              f"(90% interval {paired['p5_delta']:+.2f}.."
              f"{paired['p95_delta']:+.2f}s)")
        print(f"  P(A faster): {paired['a_win_probability']:.1%}")
        for label, strategy in (("A", strategy_a), ("B", strategy_b)):
            profile = degradation_profile(strategy, state,
                                          params, scenarios["clean_air"])
            values = ", ".join(f"{p['compound']} {p['laps']} laps "
                               f"(beta {p['beta_per_lap']:.3f})"
                               for p in profile)
            print(f"  profile {label}: {values}")
        return
    results = optimize(state, params, args.n_sims, scenarios)
    for sname, rows in results.items():
        print(f"\n[strategy] scenario: {sname} "
          f"(beta x{scenarios[sname]['beta_mult']}, "
          f"pace +{scenarios[sname]['pace_pen']}s/lap)")
        print(f"  {'strategy':32s} {'stops':>5s} {'mean':>8s} {'std':>6s} "
              f"{'p5':>8s} {'p95':>8s} {'dvs best':>9s}")
        for r in rows[:6]:
            print(f"  {r['strategy']:32s} {r['stops']:5d} {r['mean']:8.1f} "
                  f"{r['std']:6.1f} {r['p5']:8.1f} {r['p95']:8.1f} "
                  f"{r['delta_vs_best']:+9.1f}")

    if args.pre_race and args.sc_hazard:
        branch = branch_point_summary(results)
        if branch is not None:
            green = branch["green_flag"]
            sc = branch["sc_window"]
            print("\n[branch] pre-race decision point")
            print(f"  green flag : {green['strategy']} | {green['mean']:.1f}s "
                  f"(p5-p95 {green['p5']:.1f}-{green['p95']:.1f})")
            print(f"  SC/VSC     : {sc['strategy']} | {sc['mean']:.1f}s "
                  f"(p5-p95 {sc['p5']:.1f}-{sc['p95']:.1f})")
            print(f"  SC - green : {branch['sc_minus_green_mean']:+.1f}s")

    if args.pre_race:
        print(f"        pace source: {state['pace_source']}")
        return

    # monitor backtest on the same driver's actual race stints
    assert df is not None
    drv_df = df.filter((pl.col("season") == args.season)
                       & (pl.col("gp") == args.gp)
                       & (pl.col("session") == "Race")
                       & (pl.col("driver") == args.driver))
    run_monitor(drv_df, meta, rul_m, cliff_x, cliff_l, thr,
                f"{args.season} {args.gp} Race {args.driver} "
                f"(threshold {thr:.3f})", weights=weights, lstm_m=lstm_m)

    out = {
        "state": {k: v for k, v in state.items()},
        "n_sims": args.n_sims, "pit_loss": args.pit_loss,
        "scenarios": {s: [{k: v for k, v in r.items() if k != "_strat"}
                          for r in rows[:10]] for s, rows in results.items()},
    }
    with open(MODELS_DIR / "phase6_demo.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n[opt] demo saved -> {MODELS_DIR / 'phase6_demo.json'} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
