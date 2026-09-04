"""Phase 4 v2 — Train & evaluate RUL / Cliff models per the refactored spec.

Changes vs v1:
  - Regression target: piecewise performance-based target_rul (compound caps).
  - Custom asymmetric XGBoost/LightGBM objective (late predictions penalized
    2.5x — overestimating tyre life is the costly error).
  - NASA PHM asymmetric score (a1=10, a2=5) reported alongside MAE/RMSE/R2.
  - Critical-zone MAE (true RUL <= 10 laps) — the acceptance metric.
  - Cliff classifier: predicts onset within 3 laps from PURE telemetry
    features (all cliff-derived features excluded upstream in rul_data).
  - Naive baselines: constant mean + per-compound mean.

Temporal split: train 2021-2023 / val 2024 / test 2025.

CLI:
    python -m src.models.train_rul
    python -m src.models.train_rul --no-lstm
    python -m src.models.train_rul --epochs 50
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

C1, C2 = 1.0, 2.5      # asymmetric loss weights (late = C2, early = C1)
A1, A2 = 10.0, 5.0     # NASA PHM score parameters
CRIT_ZONE = 10         # RUL <= 10 laps = critical degradation zone


# ---------------------------------------------------------------------------
# Metrics (spec Task 5)
# ---------------------------------------------------------------------------
def nasa_phm_score(y_true, y_pred, a1=A1, a2=A2) -> float:
    diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    scores = np.where(diff < 0, np.exp(-diff / a1) - 1.0, np.exp(diff / a2) - 1.0)
    return float(np.sum(scores))


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

    crit = yt <= CRIT_ZONE
    n_crit = int(crit.sum())
    if n_crit > 0:
        yt_c, yp_c = yt[crit], yp[crit]
        mae_crit = float(np.mean(np.abs(yt_c - yp_c)))
        rmse_crit = float(np.sqrt(np.mean((yt_c - yp_c) ** 2)))
        ss_res_c = float(np.sum((yt_c - yp_c) ** 2))
        ss_tot_c = float(np.sum((yt_c - yt_c.mean()) ** 2)) or 1.0
        r2_crit = 1.0 - ss_res_c / ss_tot_c
    else:
        mae_crit = rmse_crit = r2_crit = float("nan")

    noncrit = ~crit
    n_noncrit = int(noncrit.sum())
    if n_noncrit > 0:
        yt_nc, yp_nc = yt[noncrit], yp[noncrit]
        mae_noncrit = float(np.mean(np.abs(yt_nc - yp_nc)))
        rmse_noncrit = float(np.sqrt(np.mean((yt_nc - yp_nc) ** 2)))
        ss_res_nc = float(np.sum((yt_nc - yp_nc) ** 2))
        ss_tot_nc = float(np.sum((yt_nc - yt_nc.mean()) ** 2)) or 1.0
        r2_noncrit = 1.0 - ss_res_nc / ss_tot_nc
    else:
        mae_noncrit = rmse_noncrit = r2_noncrit = float("nan")

    return {"MAE": mae, "RMSE": rmse, "R2": r2,
            "MAE_crit": mae_crit, "RMSE_crit": rmse_crit, "R2_crit": r2_crit,
            "MAE_noncrit": mae_noncrit, "RMSE_noncrit": rmse_noncrit, "R2_noncrit": r2_noncrit,
            "PHM": nasa_phm_score(yt, yp),
            "n": int(mask.sum()), "n_crit": n_crit, "n_noncrit": n_noncrit}


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
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_pos": int(y_true.sum()), "n": int(len(y_true))}


def _best_f1_threshold(y_true, prob):
    """Smallest threshold whose val F1 is within 1% of max (recall-biased,
    robust to val->test threshold shift)."""
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    thrs = np.unique(prob)
    if len(thrs) > 200:
        thrs = np.quantile(prob, np.linspace(0.01, 0.99, 200))
    pr_points = []
    for t in thrs:
        pred = (prob > t).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        pr_points.append((float(t), p, r, f1))
    best_f1 = max(f for _, _, _, f in pr_points)
    cands = [t for t, _, _, f in pr_points if f >= 0.99 * best_f1]
    best_t = min(cands) if cands else 0.5
    return best_t, best_f1, pr_points


# ---------------------------------------------------------------------------
# Asymmetric custom objectives (spec Task 5.2)
# ---------------------------------------------------------------------------
def _asym_grad_hess(y_true, y_pred):
    residual = y_pred - y_true
    grad = np.where(residual <= 0, 2.0 * C1 * residual, 2.0 * C2 * residual)
    hess = np.where(residual <= 0, 2.0 * C1, 2.0 * C2)
    return grad, hess


def xgb_asym_obj(y_pred, dtrain):
    y_true = dtrain.get_label()
    return _asym_grad_hess(y_true, y_pred)


def lgb_asym_obj(preds, dataset):
    y_true = dataset.get_label()
    return _asym_grad_hess(y_true, preds)


def to_numpy(X) -> np.ndarray:
    return X.to_pandas().to_numpy(dtype=np.float32, na_value=np.nan)


# ---------------------------------------------------------------------------
# Tree models
# ---------------------------------------------------------------------------
def train_xgb_rul(Xtr, ytr, Xva, yva, Xte, yte, symmetric=False):
    import xgboost as xgb
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    dte = xgb.DMatrix(Xte, label=yte)
    params = dict(tree_method="hist", device="cuda", max_depth=8, eta=0.05,
                  subsample=0.9, colsample_bytree=0.9, min_child_weight=4,
                  eval_metric="rmse")
    if symmetric:
        params["objective"] = "reg:squarederror"
        obj = None
        name = "xgb_rul_sym"
    else:
        obj = xgb_asym_obj
        name = "xgb_rul"
    model = xgb.train(params, dtr, num_boost_round=800, evals=[(dva, "val")],
                      obj=obj, early_stopping_rounds=40, verbose_eval=False)
    pred = np.clip(model.predict(dte), 0, None)
    model.save_model(str(MODELS_DIR / f"{name}.json"))
    return reg_metrics(yte, pred), pred


def train_lgb_rul(Xtr, ytr, Xva, yva, Xte, yte):
    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    params = dict(max_depth=-1, num_leaves=63, learning_rate=0.05,
                  feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=1, min_data_in_leaf=20, verbose=-1,
                  objective=lgb_asym_obj, metric="rmse")
    model = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
    pred = np.clip(model.predict(Xte), 0, None)
    model.save_model(str(MODELS_DIR / "lgb_rul.txt"))
    return reg_metrics(yte, pred), pred


def train_xgb_cliff(Xtr, ytr, Xva, yva, Xte, yte):
    """XGB + LGB probability ensemble; weight proportional to each member's
    validation ROC-AUC (Task 8.H.6). Falls back to 50/50 if AUC is unreliable."""
    import xgboost as xgb
    import lightgbm as lgb
    pos = int(ytr.sum())
    neg = int(len(ytr) - pos)
    w_pos = float(np.sqrt(neg / max(pos, 1)))
    sw = np.where(ytr > 0, w_pos, 1.0).astype(np.float32)

    dtr = xgb.DMatrix(Xtr, label=ytr, weight=sw)
    dva = xgb.DMatrix(Xva, label=yva)
    dte = xgb.DMatrix(Xte, label=yte)
    xp = dict(objective="binary:logistic", tree_method="hist", device="cuda",
              max_depth=7, eta=0.06, subsample=0.85, colsample_bytree=0.85,
              min_child_weight=3, eval_metric="aucpr", max_delta_step=1,
              reg_lambda=2.0, reg_alpha=0.5)
    xm = xgb.train(xp, dtr, num_boost_round=800, evals=[(dva, "val")],
                   early_stopping_rounds=40, verbose_eval=False)
    xm.save_model(str(MODELS_DIR / "xgb_cliff.json"))

    ldtr = lgb.Dataset(Xtr, label=ytr, weight=sw)
    ldva = lgb.Dataset(Xva, label=yva, reference=ldtr)
    lp = dict(max_depth=-1, num_leaves=63, learning_rate=0.06,
              feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
              min_data_in_leaf=20, verbose=-1, objective="binary")
    lm = lgb.train(lp, ldtr, num_boost_round=800, valid_sets=[ldva],
                   callbacks=[lgb.early_stopping(40, verbose=False)])
    lm.save_model(str(MODELS_DIR / "lgb_cliff.txt"))

    # member val probabilities
    xgb_va = xm.predict(dva)
    lgb_va = lm.predict(Xva)
    xgb_te = xm.predict(dte)
    lgb_te = lm.predict(Xte)

    # --- Task 8.H.6.2: weighting proportional to validation ROC-AUC ---
    try:
        from sklearn.metrics import roc_auc_score
        auc_x = float(roc_auc_score(yva, xgb_va))
        auc_l = float(roc_auc_score(yva, lgb_va))
    except Exception:
        auc_x = auc_l = 0.5
    # AUC below 0.5 is worse than random; clamp and normalize to weights.
    auc_x = max(auc_x, 0.0)
    auc_l = max(auc_l, 0.0)
    if (auc_x + auc_l) <= 1e-6:
        w_x = w_l = 0.5
    else:
        w_x = auc_x / (auc_x + auc_l)
        w_l = 1.0 - w_x

    val_prob = w_x * xgb_va + w_l * lgb_va
    test_prob = w_x * xgb_te + w_l * lgb_te
    best_thr, best_val_f1, pr_points = _best_f1_threshold(yva, val_prob)
    pred = (test_prob > best_thr).astype(int)
    metrics = clf_metrics(yte, pred, test_prob)
    metrics["threshold"] = best_thr
    metrics["val_f1_at_threshold"] = best_val_f1
    metrics["w_pos"] = w_pos
    metrics["ensemble"] = "xgb+lgb auc-weighted"
    metrics["ensemble_weights"] = {"xgb": round(w_x, 3), "lgb": round(w_l, 3)}
    metrics["ensemble_auc"] = {"xgb": round(auc_x, 3), "lgb": round(auc_l, 3)}
    pr_points.sort(key=lambda x: x[0])
    summary = [pr_points[i] for i in np.linspace(0, len(pr_points) - 1, 7).astype(int)]
    metrics["pr_summary"] = [{"thr": round(t, 3), "prec": round(p, 3),
                              "rec": round(r, 3), "f1": round(f, 3)}
                             for t, p, r, f in summary]
    return metrics, pred, test_prob


# ---------------------------------------------------------------------------
# LSTM (target = piecewise target_rul)
# ---------------------------------------------------------------------------
MAXLEN = 80


def pad_batch(seqs, ruls, maxlen=MAXLEN):
    n, f = len(seqs), seqs[0].shape[1]
    X = np.zeros((n, maxlen, f), dtype=np.float32)
    M = np.zeros((n, maxlen), dtype=np.float32)
    Y = np.full((n, maxlen), np.nan, dtype=np.float32)
    for i, (s, y) in enumerate(zip(seqs, ruls)):
        L = min(len(s), maxlen)
        X[i, :L] = s[:L]
        M[i, :L] = 1
        Y[i, :L] = y[:L]
    return X, M, Y


def train_lstm(seq, feats, epochs, batch_size, device):
    import torch
    import torch.nn as nn

    # Ensure maxlen accommodates all stints across splits (no truncation)
    maxlen = max(
        max(len(s) for s in seq["train"]["X"]),
        max(len(s) for s in seq["val"]["X"]),
        max(len(s) for s in seq["test"]["X"]),
        MAXLEN
    )

    Xtr, Mtr, Ytr = pad_batch(seq["train"]["X"], seq["train"]["y_rul"], maxlen=maxlen)
    Xva, Mva, Yva = pad_batch(seq["val"]["X"], seq["val"]["y_rul"], maxlen=maxlen)
    Xte, Mte, Yte = pad_batch(seq["test"]["X"], seq["test"]["y_rul"], maxlen=maxlen)

    def t(a):
        return torch.from_numpy(a).to(device)

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

    net = RULNet(len(feats)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, epochs=epochs,
        steps_per_epoch=int(np.ceil(len(Xtr) / batch_size)),
        pct_start=0.2)

    class AsymSmoothL1(nn.Module):
        def forward(self, pred, target, mask=None):
            res = pred - target
            w = torch.where(res > 0, C2, C1)
            raw = w * nn.functional.smooth_l1_loss(
                pred, target, reduction="none", beta=1.0)
            if mask is not None:
                return (raw * mask).sum() / mask.sum().clamp(min=1.0)
            return raw.mean()

    crit = AsymSmoothL1()

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
                loss = crit(p, torch.nan_to_num(yb), mask=mask)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                    sched.step()
                n_valid = int(mask.sum().item())
                losses.append(float(loss.detach().item()) * n_valid)
                n += n_valid
        return sum(losses) / max(n, 1)

    best, best_state, best_epoch, best_lr = 1e9, None, 0, 0.0
    patience, bad = 10, 0
    for e in range(epochs):
        tr = epoch(Xtr, Mtr, Ytr, True)
        va = epoch(Xva, Mva, Yva, False)
        current_lr = opt.param_groups[0]["lr"]
        if va < best - 1e-4:
            best, best_epoch, bad = va, e + 1, 0
            best_lr = current_lr
            best_state = {k: v.cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            bad += 1
        if (e + 1) % 5 == 0 or e == 0:
            print(f"   lstm epoch {e+1:2d}  lr={current_lr:.6f}  tr={tr:.4f} va={va:.4f}"
                  f"  best=va{best_epoch}={best:.4f}")
        if bad >= patience:
            print(f"   lstm early stop at epoch {e+1}")
            break
    net.load_state_dict(best_state)
    torch.save(best_state, str(MODELS_DIR / "lstm_rul.pt"))
    print(f"   lstm best epoch = {best_epoch} (val={best:.4f}, lr={best_lr:.6f})")

    net.eval()
    with torch.no_grad():
        pt = net(t(Xte)).cpu().numpy()
    yt, yp = [], []
    for i in range(len(Xte)):
        L = int(Mte[i].sum())
        yt.extend(Yte[i, :L].tolist())
        yp.extend(pt[i, :L].tolist())
    yt = np.array(yt, dtype=np.float32)
    yp = np.clip(np.array(yp, dtype=np.float32), 0, None)

    # Task 8.0.2.2: assert test-time unpadded lap count matches exactly
    expected_laps = sum(len(y) for y in seq["test"]["y_rul"])
    assert len(yt) == expected_laps, f"Sequence flattening dropped laps! {len(yt)} != {expected_laps}"

    metrics = reg_metrics(yt, yp)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = round(float(best), 4)
    metrics["lr_at_best_epoch"] = round(float(best_lr), 7)
    return metrics, yp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    t0 = time.time()
    print("Preparing data (refactored spec: filtered stints, fuel-corrected, "
          "piecewise RUL, no cliff-derived features)...")
    tab, seq, feats, codes, norm = prepare()
    Xtr, ytr_r, ytr_c = tab["train"]
    Xva, yva_r, yva_c = tab["val"]
    Xte, yte_r, yte_c = tab["test"]
    Xtr_n, Xva_n, Xte_n = to_numpy(Xtr), to_numpy(Xva), to_numpy(Xte)
    print(f"train={Xtr_n.shape} val={Xva_n.shape} test={Xte_n.shape}")

    results = {}
    comp_idx = feats.index("compound")

    # --- Naive baselines ---
    print("\n[1] Naive baselines")
    mean_rul = float(np.nanmean(ytr_r))
    results["naive_const"] = reg_metrics(
        yte_r, np.full_like(yte_r, mean_rul))
    comp_mean = {}
    for c in np.unique(Xtr_n[:, comp_idx]):
        m = ytr_r[Xtr_n[:, comp_idx] == c].mean()
        comp_mean[float(c)] = float(m)
    pred_comp = np.array([comp_mean.get(float(c), mean_rul)
                          for c in Xte_n[:, comp_idx]])
    results["naive_compound"] = reg_metrics(yte_r, pred_comp)
    print(f"  naive_const   : MAE={results['naive_const']['MAE']:.3f} "
          f"crit={results['naive_const']['MAE_crit']:.3f}")
    print(f"  naive_compound: MAE={results['naive_compound']['MAE']:.3f} "
          f"crit={results['naive_compound']['MAE_crit']:.3f}")

    # --- XGBoost RUL (asymmetric objective) ---
    print("\n[2] XGBoost RUL (GPU, asymmetric objective c2/c1=2.5)")
    t1 = time.time()
    m_xgb, pred_xgb = train_xgb_rul(Xtr_n, ytr_r, Xva_n, yva_r, Xte_n, yte_r)
    results["xgb_rul"] = m_xgb
    print(f"  MAE={m_xgb['MAE']:.3f} RMSE={m_xgb['RMSE']:.3f} R2={m_xgb['R2']:.3f} "
          f"MAE_crit={m_xgb['MAE_crit']:.3f} PHM={m_xgb['PHM']:.0f} "
          f"({time.time()-t1:.0f}s)")

    # --- XGBoost RUL (symmetric reference: isolates the asymmetry cost) ---
    print("\n[2b] XGBoost RUL (GPU, symmetric reference)")
    t1 = time.time()
    m_sym, _ = train_xgb_rul(Xtr_n, ytr_r, Xva_n, yva_r, Xte_n, yte_r,
                             symmetric=True)
    results["xgb_rul_sym"] = m_sym
    print(f"  MAE={m_sym['MAE']:.3f} RMSE={m_sym['RMSE']:.3f} R2={m_sym['R2']:.3f} "
          f"MAE_crit={m_sym['MAE_crit']:.3f} PHM={m_sym['PHM']:.0f} "
          f"({time.time()-t1:.0f}s)")

    # --- LightGBM RUL (asymmetric objective) ---
    print("\n[3] LightGBM RUL (CPU, asymmetric objective)")
    t1 = time.time()
    m_lgb, pred_lgb = train_lgb_rul(Xtr_n, ytr_r, Xva_n, yva_r, Xte_n, yte_r)
    results["lgb_rul"] = m_lgb
    print(f"  MAE={m_lgb['MAE']:.3f} RMSE={m_lgb['RMSE']:.3f} R2={m_lgb['R2']:.3f} "
          f"MAE_crit={m_lgb['MAE_crit']:.3f} PHM={m_lgb['PHM']:.0f} "
          f"({time.time()-t1:.0f}s)")

    # --- XGBoost cliff (pure telemetry features) ---
    print("\n[4] XGBoost cliff classifier (GPU, pure telemetry features)")
    t1 = time.time()
    m_cliff, pred_cliff, prob_cliff = train_xgb_cliff(
        Xtr_n, ytr_c, Xva_n, yva_c, Xte_n, yte_c)
    results["xgb_cliff"] = m_cliff
    print(f"  w_pos={m_cliff['w_pos']:.2f} threshold={m_cliff['threshold']:.3f} "
          f"(val F1={m_cliff['val_f1_at_threshold']:.3f})")
    print(f"  prec={m_cliff['precision']:.3f} rec={m_cliff['recall']:.3f} "
          f"f1={m_cliff['f1']:.3f} auc={m_cliff['auc']:.3f}  "
          f"({time.time()-t1:.0f}s)")
    print("  PR operating points (thr / prec / rec / f1):")
    for pt in m_cliff["pr_summary"]:
        print(f"    {pt['thr']:.3f}  {pt['prec']:.3f}  {pt['rec']:.3f}  {pt['f1']:.3f}")

    # --- LSTM RUL ---
    if not args.no_lstm:
        print("\n[5] PyTorch LSTM RUL (GPU, asymmetric SmoothL1)")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  device={device}")
        t1 = time.time()
        m_lstm, pred_lstm = train_lstm(seq, feats, args.epochs, 256, device)
        results["lstm_rul"] = m_lstm
        print(f"  MAE={m_lstm['MAE']:.3f} RMSE={m_lstm['RMSE']:.3f} "
              f"R2={m_lstm['R2']:.3f} MAE_crit={m_lstm['MAE_crit']:.3f} "
              f"PHM={m_lstm['PHM']:.0f}  ({time.time()-t1:.0f}s)")

        # --- Piecewise Blended RUL (Production: Option B) ---
        print("\n[6] Piecewise Blended RUL (Production: XGBoost + LSTM)")
        w_blend = np.clip((12.0 - pred_xgb) / 6.0, 0.0, 1.0)
        pred_blend = w_blend * pred_lstm + (1.0 - w_blend) * pred_xgb
        m_blend = reg_metrics(yte_r, pred_blend)
        results["piecewise_blend"] = m_blend
        print(f"  MAE={m_blend['MAE']:.3f} RMSE={m_blend['RMSE']:.3f} "
              f"R2={m_blend['R2']:.3f} MAE_crit={m_blend['MAE_crit']:.3f} "
              f"PHM={m_blend['PHM']:.0f}")

    # --- Save metadata + metrics ---
    meta = {"features": feats, "codes": codes,
            "norm_means": {k: float(v) for k, v in norm[0].items()},
            "norm_stds": {k: float(v) for k, v in norm[1].items()},
            "split": {"train": ["2021", "2022", "2023"], "val": ["2024"],
                      "test": ["2025"]},
            "target": "target_rul (piecewise, performance-based)",
            "cliff_target": "cliff_within_3 (onset within 3 laps, pure features)",
            "asymmetric_loss": {"c1": C1, "c2": C2},
            "nasa_phm": {"a1": A1, "a2": A2},
            "naive_compound_means": comp_mean,
            "blend_policy": {"type": "piecewise_linear", "rul_lo": 6.0, "rul_hi": 12.0}}
    with open(MODELS_DIR / "feature_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    with open(MODELS_DIR / "phase4_metrics.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    # --- Report ---
    print("\n" + "=" * 78)
    print("PHASE 4 v2 — RUL / Cliff (refactored) — RESULTS (test 2025)")
    print("=" * 78)
    print(f"\n{'Model':18s} {'MAE':>7s} {'RMSE':>7s} {'R2':>7s} "
          f"{'MAE<=10':>8s} {'R2_crit':>8s} {'R2_nc':>7s} {'PHM':>10s}")
    for k in ["naive_const", "naive_compound", "lgb_rul", "xgb_rul",
              "xgb_rul_sym", "lstm_rul", "piecewise_blend"]:
        if k in results:
            r = results[k]
            r2_c = f"{r.get('R2_crit', float('nan')):8.3f}"
            r2_nc = f"{r.get('R2_noncrit', float('nan')):7.3f}"
            print(f"{k:18s} {r['MAE']:7.3f} {r['RMSE']:7.3f} {r['R2']:7.3f} "
                  f"{r['MAE_crit']:8.3f} {r2_c} {r2_nc} {r['PHM']:10.0f}")
    c = results["xgb_cliff"]
    print(f"\nCliff (onset within 3 laps, NO leakage):")
    print(f"  threshold={c['threshold']:.3f} prec={c['precision']:.3f} "
          f"rec={c['recall']:.3f} f1={c['f1']:.3f} auc={c['auc']:.3f} "
          f"(n_pos={c['n_pos']}/{c['n']})")
    print("\nAcceptance criteria (spec section 5):")
    print(f"  classifier F1 target 0.82-0.91  ->  actual {c['f1']:.3f}")
    best_r2 = max(results.get(k, {}).get("R2", 0)
                  for k in ["xgb_rul", "xgb_rul_sym", "lgb_rul"])
    print(f"  regressor R2 >= 0.80            ->  actual {best_r2:.3f}")
    best_crit = min(results.get(k, {}).get("MAE_crit", 99)
                    for k in ["xgb_rul", "xgb_rul_sym", "lgb_rul"])
    print(f"  critical-zone MAE < 2.0         ->  actual {best_crit:.3f}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Artifacts: {MODELS_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
