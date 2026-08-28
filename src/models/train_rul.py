"""Phase 4 — Train & evaluate the RUL / Degradation Cliff Predictor.

Models:
  - Naive baselines (constant mean; life-aware mean stint length)
  - XGBoost (GPU) RUL regressor
  - LightGBM (GPU/CPU) RUL regressor
  - XGBoost (GPU) cliff classifier (cliff within next 3 laps)
  - PyTorch LSTM RUL regressor (per-stint sequences, GPU)

Temporal split: train 2021-2023 / val 2024 / test 2025 (no same-GP leakage).

CLI:
    python -m src.models.train_rul
    python -m src.models.train_rul --no-lstm      # skip the LSTM (fast)
    python -m src.models.train_rul --epochs 15
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.ingest.config import ROOT
from src.models.rul_data import prepare

MODELS_DIR = ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "n": int(mask.sum())}


def clf_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # ROC-AUC (simple)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_pos": int(y_true.sum()), "n": int(len(y_true))}


# ---------------------------------------------------------------------------
# Tree models
# ---------------------------------------------------------------------------
def to_numpy(X) -> np.ndarray:
    return X.to_pandas().to_numpy(dtype=np.float32, na_value=np.nan)


def train_xgb_rul(Xtr, ytr, Xva, yva, Xte, yte):
    import xgboost as xgb
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    dte = xgb.DMatrix(Xte, label=yte)
    params = dict(objective="reg:squarederror", tree_method="hist",
                  device="cuda", max_depth=6, eta=0.1,
                  subsample=0.9, colsample_bytree=0.9, min_child_weight=4)
    model = xgb.train(params, dtr, num_boost_round=400, evals=[(dva, "val")],
                      early_stopping_rounds=25, verbose_eval=False)
    pred = model.predict(dte)
    model.save_model(str(MODELS_DIR / "xgb_rul.json"))
    return reg_metrics(yte, pred), pred


def train_lgb_rul(Xtr, ytr, Xva, yva, Xte, yte):
    import lightgbm as lgb
    device = "gpu"
    try:
        m = lgb.LGBMRegressor(n_estimators=400, device=device, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)])
    except Exception:
        device = "cpu"
        m = lgb.LGBMRegressor(n_estimators=400, device="cpu", verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)])
    pred = m.predict(Xte)
    m.booster_.save_model(str(MODELS_DIR / "lgb_rul.txt"))
    pred = np.clip(pred, 0, None)
    return reg_metrics(yte, pred), pred, device


def train_xgb_cliff(Xtr, ytr, Xva, yva, Xte, yte):
    import xgboost as xgb
    spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    dte = xgb.DMatrix(Xte, label=yte)
    params = dict(objective="binary:logistic", tree_method="hist", device="cuda",
                  max_depth=5, eta=0.08, subsample=0.9, colsample_bytree=0.9,
                  scale_pos_weight=spw, eval_metric="auc")
    model = xgb.train(params, dtr, num_boost_round=400, evals=[(dva, "val")],
                      early_stopping_rounds=30, verbose_eval=False)
    prob = model.predict(dte)
    pred = (prob > 0.5).astype(int)
    model.save_model(str(MODELS_DIR / "xgb_cliff.json"))
    return clf_metrics(yte, pred, prob), pred, prob


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
MAXLEN = 64


def pad_batch(seqs, ruls):
    n, f = len(seqs), seqs[0].shape[1]
    X = np.zeros((n, MAXLEN, f), dtype=np.float32)
    M = np.zeros((n, MAXLEN), dtype=np.float32)
    Y = np.full((n, MAXLEN), np.nan, dtype=np.float32)
    for i, (s, y) in enumerate(zip(seqs, ruls)):
        L = min(len(s), MAXLEN)
        X[i, :L] = s[:L]
        M[i, :L] = 1
        Y[i, :L] = y[:L]
    return X, M, Y


def train_lstm(seq, feats, epochs, batch_size, device):
    import torch
    import torch.nn as nn

    Xtr, Mtr, Ytr = pad_batch(seq["train"]["X"], seq["train"]["y_rul"])
    Xva, Mva, Yva = pad_batch(seq["val"]["X"], seq["val"]["y_rul"])
    Xte, Mte, Yte = pad_batch(seq["test"]["X"], seq["test"]["y_rul"])

    def t(a):
        return torch.from_numpy(a).to(device)

    class RULNet(nn.Module):
        def __init__(self, f, h=128, l=2, ncat=(9, 4)):
            super().__init__()
            self.emb_comp = nn.Embedding(9, 4)
            self.emb_st = nn.Embedding(4, 2)
            self.lstm = nn.LSTM(f - 2 + 4 + 2, h, num_layers=l,
                                batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(),
                                      nn.Linear(64, 1))

        def forward(self, x):
            comp = x[..., feats.index("compound")].long()
            st = x[..., feats.index("session_type")].long()
            rest = [x[..., j] for j in range(x.shape[-1])
                    if feats[j] not in ("compound", "session_type")]
            z = torch.cat([self.emb_comp(comp), self.emb_st(st)]
                          + [r.unsqueeze(-1) for r in rest], dim=-1)
            out, _ = self.lstm(z)
            return self.head(out).squeeze(-1)

    net = RULNet(len(feats)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, epochs=epochs,
        steps_per_epoch=int(np.ceil(len(Xtr) / batch_size)))
    crit = nn.SmoothL1Loss(beta=1.0)

    def epoch(X, M, Y, train):
        net.train() if train else net.eval()
        idx = np.arange(len(X))
        if train:
            np.random.shuffle(idx)
        losses, n = [], 0
        with torch.set_grad_enabled(train):
            for s in range(0, len(idx), batch_size):
                bi = idx[s:s + batch_size]
                xb, mb, yb = t(X[bi]), t(M[bi]), t(Y[bi])
                p = net(xb)
                mask = mb * torch.isfinite(yb)
                loss = crit(p * mask, torch.nan_to_num(yb) * mask)
                if train:
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step(); sched.step()
                losses.append(float(loss.detach()) * int(mask.sum().item()))
                n += int(mask.sum().item())
        return sum(losses) / max(n, 1)

    best, best_state = 1e9, None
    for e in range(epochs):
        tr = epoch(Xtr, Mtr, Ytr, True)
        va = epoch(Xva, Mva, Yva, False)
        if va < best:
            best, best_state = va, {k: v.cpu().clone()
                                     for k, v in net.state_dict().items()}
        if (e + 1) % 5 == 0 or e == 0:
            print(f"   lstm epoch {e+1:2d}  tr={tr:.4f} va={va:.4f}")
    net.load_state_dict(best_state)
    torch.save(best_state, str(MODELS_DIR / "lstm_rul.pt"))

    # test predictions
    net.eval()
    with torch.no_grad():
        pt = net(t(Xte)).cpu().numpy()
    yt, yp = [], []
    for i in range(len(Xte)):
        L = int(Mte[i].sum())
        yt.extend(Yte[i, :L].tolist())
        yp.extend(pt[i, :L].tolist())
    return reg_metrics(np.array(yt), np.array(yp)), np.array(yp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    t0 = time.time()
    print("Preparing data...")
    tab, seq, feats, codes, norm = prepare()
    Xtr, ytr_r, ytr_c = tab["train"]
    Xva, yva_r, yva_c = tab["val"]
    Xte, yte_r, yte_c = tab["test"]
    Xtr_n, Xva_n, Xte_n = to_numpy(Xtr), to_numpy(Xva), to_numpy(Xte)
    print(f"train={Xtr_n.shape} val={Xva_n.shape} test={Xte_n.shape}")

    results = {}

    # --- Naive baselines ---
    print("\n[1] Naive baselines")
    mean_rul = float(np.nanmean(ytr_r))
    pred_const = np.full_like(yte_r, mean_rul)
    results["naive_const"] = reg_metrics(yte_r, pred_const)

    stint_len = float(np.nanmean(ytr_r + Xtr_n[:, feats.index("life")]))
    pred_life = np.clip(stint_len - Xte_n[:, feats.index("life")], 0, None)
    results["naive_life"] = reg_metrics(yte_r, pred_life)
    print(f"  naive_const : MAE={results['naive_const']['MAE']:.3f}")
    print(f"  naive_life  : MAE={results['naive_life']['MAE']:.3f}")

    # --- XGBoost RUL ---
    print("\n[2] XGBoost RUL (GPU)")
    t1 = time.time()
    m_xgb, pred_xgb = train_xgb_rul(Xtr_n, ytr_r, Xva_n, yva_r, Xte_n, yte_r)
    results["xgb_rul"] = m_xgb
    print(f"  {m_xgb}  ({time.time()-t1:.0f}s)")

    # --- LightGBM RUL ---
    print("\n[3] LightGBM RUL")
    t1 = time.time()
    m_lgb, pred_lgb, dev = train_lgb_rul(Xtr_n, ytr_r, Xva_n, yva_r, Xte_n, yte_r)
    results["lgb_rul"] = {**m_lgb, "device": dev}
    print(f"  {m_lgb} device={dev} ({time.time()-t1:.0f}s)")

    # --- XGBoost cliff ---
    print("\n[4] XGBoost cliff classifier (GPU)")
    t1 = time.time()
    m_cliff, pred_cliff, prob_cliff = train_xgb_cliff(
        Xtr_n, ytr_c, Xva_n, yva_c, Xte_n, yte_c)
    results["xgb_cliff"] = m_cliff
    print(f"  {m_cliff}  ({time.time()-t1:.0f}s)")

    # --- LSTM RUL ---
    if not args.no_lstm:
        print("\n[5] PyTorch LSTM RUL")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  device={device}")
        t1 = time.time()
        m_lstm, pred_lstm = train_lstm(seq, feats, args.epochs, 256, device)
        results["lstm_rul"] = m_lstm
        print(f"  {m_lstm}  ({time.time()-t1:.0f}s)")

    # --- Save metadata + metrics ---
    meta = {"features": feats, "codes": codes,
            "split": {"train": ["2021", "2022", "2023"], "val": ["2024"],
                      "test": ["2025"]},
            "naive_mean_rul": mean_rul, "naive_stint_len": stint_len}
    with open(MODELS_DIR / "feature_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    with open(MODELS_DIR / "phase4_metrics.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    # --- Report ---
    print("\n" + "=" * 72)
    print("PHASE 4 — RUL / Cliff Predictor — RESULTS (test set 2025)")
    print("=" * 72)
    print(f"\n{'Model':22s} {'MAE':>8s} {'RMSE':>8s} {'R2':>7s}")
    for k in ["naive_const", "naive_life", "lgb_rul", "xgb_rul", "lstm_rul"]:
        if k in results:
            r = results[k]
            print(f"{k:22s} {r['MAE']:8.3f} {r['RMSE']:8.3f} {r['R2']:7.3f}")
    print(f"\nCliff classifier (cliff within 3 laps):")
    c = results["xgb_cliff"]
    print(f"  precision={c['precision']:.3f} recall={c['recall']:.3f} "
          f"f1={c['f1']:.3f} auc={c['auc']:.3f}  (n_pos={c['n_pos']}/{c['n']})")
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Artifacts: {MODELS_DIR}")
    print(f"Metrics : {MODELS_DIR/'phase4_metrics.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
