import argparse
import glob
import math
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def metrics(y_true: np.ndarray, y_pred: np.ndarray, cap: float) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    nrmse = rmse / cap
    nmae = mae / cap
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100.0)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "NRMSE": nrmse, "NMAE": nmae, "sMAPE": smape}


def pred_to_long(pred: pd.DataFrame, year: int, horizon: int = 96) -> pd.DataFrame:
    pred = pred.copy()
    pred["target_start_time"] = pd.to_datetime(pred["target_start_time"], errors="coerce")
    pred = pred.dropna(subset=["target_start_time"]).reset_index(drop=True)

    rows = []
    for _, r in pred.iterrows():
        t0 = r["target_start_time"]
        for k in range(1, horizon + 1):
            rows.append(
                {
                    "year": year,
                    "target_start_time": t0,
                    "target_time": t0 + pd.Timedelta(minutes=15 * (k - 1)),
                    "lead": k,
                    "y_true_mw": float(r[f"y_true_t+{k}_mw"]),
                    "y_pred_mw": float(r[f"y_pred_t+{k}_mw"]),
                }
            )
    return pd.DataFrame(rows)


def build_dataset(pred_glob: str, typhoon_path: str) -> pd.DataFrame:
    parts = []
    for fp in sorted(glob.glob(pred_glob)):
        m = re.search(r"(\d{4})", os.path.basename(fp))
        if not m:
            continue
        year = int(m.group(1))
        parts.append(pred_to_long(pd.read_csv(fp), year=year, horizon=96))

    if not parts:
        raise RuntimeError("No prediction files found")

    long_pred = pd.concat(parts, ignore_index=True)

    ty = pd.read_csv(typhoon_path)
    ty["target_time"] = pd.to_datetime(ty["target_time"], errors="coerce")
    ty = ty.dropna(subset=["target_time", "台风编号"]).copy()

    if "场站距离_km" in ty.columns:
        ty["场站距离_km"] = pd.to_numeric(ty["场站距离_km"], errors="coerce")
        ty = ty.sort_values(["target_time", "场站距离_km"]).drop_duplicates(subset=["target_time"], keep="first")
    else:
        ty = ty.sort_values(["target_time"]).drop_duplicates(subset=["target_time"], keep="first")

    m = long_pred.merge(ty, on="target_time", how="inner")
    if len(m) == 0:
        raise RuntimeError("Predictions and typhoon data do not overlap")

    m = m.sort_values(["台风编号", "target_time", "lead"]).reset_index(drop=True)
    m["delta_mw"] = m["y_true_mw"] - m["y_pred_mw"]
    m["lead_norm"] = (m["lead"] - 1) / 95.0
    return m


def fit_binner(train: pd.DataFrame, feature: str) -> Optional[Tuple[np.ndarray, pd.Series, pd.Series]]:
    vals = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if vals.notna().sum() < 20:
        return None

    q = np.quantile(vals.dropna(), [0.0, 0.25, 0.5, 0.75, 1.0])
    q = np.unique(q)
    if len(q) < 3:
        return None

    b = pd.cut(vals, bins=q, labels=False, include_lowest=True, duplicates="drop")
    tmp = train.copy()
    tmp["_bin"] = b
    tmp = tmp.dropna(subset=["_bin"])
    if len(tmp) < 20:
        return None

    gb = tmp.groupby([tmp["lead"], tmp["_bin"]])["delta_mw"].mean()
    lead_bias = tmp.groupby("lead")["delta_mw"].mean()
    return q, gb, lead_bias


def apply_binner(df: pd.DataFrame, feature: str, pack: Tuple[np.ndarray, pd.Series, pd.Series]) -> np.ndarray:
    q, gb, lead_bias = pack
    vals = pd.to_numeric(df[feature], errors="coerce")
    b = pd.cut(vals, bins=q, labels=False, include_lowest=True, duplicates="drop")
    out = []
    for lead, bin_id, base in zip(df["lead"].to_numpy(), b.to_numpy(), df["y_pred_mw"].to_numpy()):
        key = (lead, bin_id)
        if key in gb.index:
            d = gb[key]
        else:
            d = lead_bias.get(lead, 0.0)
        out.append(base + d)
    return np.asarray(out, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_soft_alpha(train_df: pd.DataFrame, p_dist_tr: np.ndarray, p_hvg_tr: np.ndarray,
                   val_df: pd.DataFrame, p_dist_val: np.ndarray, p_hvg_val: np.ndarray) -> Dict:
    # Features for gating: holland intensity + distance + lead + predictor disagreement + base prediction.
    def make_X(df: pd.DataFrame, p_d: np.ndarray, p_h: np.ndarray) -> np.ndarray:
        hvg = pd.to_numeric(df["hollandvg"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        dist = pd.to_numeric(df["场站距离_km"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        lead = pd.to_numeric(df["lead_norm"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        gap = np.abs(p_h - p_d).astype(np.float32)
        base = pd.to_numeric(df["y_pred_mw"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        return np.column_stack([hvg, dist, lead, gap, base])

    Xtr = make_X(train_df, p_dist_tr, p_hvg_tr)
    ytr = train_df["y_true_mw"].to_numpy(dtype=np.float32)
    Xva = make_X(val_df, p_dist_val, p_hvg_val)
    yva = val_df["y_true_mw"].to_numpy(dtype=np.float32)

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    nfeat = Xtr.shape[1]
    w = np.zeros(nfeat, dtype=np.float32)
    b = 0.0
    lr = 0.03
    reg = 1e-3

    # Train alpha gate on train events, pick best on validation event.
    best = {"val_mse": float("inf"), "w": w.copy(), "b": b}

    for _ in range(600):
        z = Xtr @ w + b
        a = _sigmoid(z)
        p = (1.0 - a) * p_dist_tr + a * p_hvg_tr
        err = p - ytr

        da_dz = a * (1.0 - a)
        dp_da = p_hvg_tr - p_dist_tr
        dz_term = 2.0 * err * dp_da * da_dz

        grad_w = (Xtr.T @ dz_term) / len(Xtr) + reg * w
        grad_b = float(np.mean(dz_term))

        w -= lr * grad_w
        b -= lr * grad_b

        zva = Xva @ w + b
        ava = _sigmoid(zva)
        pva = (1.0 - ava) * p_dist_val + ava * p_hvg_val
        vmse = float(np.mean((pva - yva) ** 2))
        if vmse < best["val_mse"]:
            best = {"val_mse": vmse, "w": w.copy(), "b": b}

    return {"w": best["w"], "b": best["b"], "mu": mu, "sd": sd, "val_mse": best["val_mse"]}


def apply_soft_alpha(df: pd.DataFrame, p_dist: np.ndarray, p_hvg: np.ndarray, gate: Dict) -> np.ndarray:
    hvg = pd.to_numeric(df["hollandvg"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    dist = pd.to_numeric(df["场站距离_km"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    lead = pd.to_numeric(df["lead_norm"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    gap = np.abs(p_hvg - p_dist).astype(np.float32)
    base = pd.to_numeric(df["y_pred_mw"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    X = np.column_stack([hvg, dist, lead, gap, base])
    X = (X - gate["mu"]) / gate["sd"]
    a = _sigmoid(X @ gate["w"] + gate["b"])
    p = (1.0 - a) * p_dist + a * p_hvg
    return p.astype(np.float32), a.astype(np.float32)


def run_loto(df: pd.DataFrame, cap: float):
    events = df.groupby("台风编号")["target_time"].min().sort_values().index.tolist()
    if len(events) < 4:
        raise RuntimeError("Need at least 4 typhoon events")

    rows = []

    for test_event in events:
        remain = [e for e in events if e != test_event]
        remain_sorted = (
            df[df["台风编号"].isin(remain)]
            .groupby("台风编号")["target_time"]
            .min()
            .sort_values()
            .index.tolist()
        )
        val_event = remain_sorted[-1]
        train_events = remain_sorted[:-1]

        tr = df[df["台风编号"].isin(train_events)].copy()
        va = df[df["台风编号"].isin([val_event])].copy()
        te = df[df["台风编号"].isin([test_event])].copy()

        dist_pack = fit_binner(tr, "场站距离_km")
        hvg_pack = fit_binner(tr, "hollandvg")
        if dist_pack is None or hvg_pack is None:
            continue

        p_dist_tr = np.clip(apply_binner(tr, "场站距离_km", dist_pack), 0.0, cap)
        p_hvg_tr = np.clip(apply_binner(tr, "hollandvg", hvg_pack), 0.0, cap)
        p_dist_va = np.clip(apply_binner(va, "场站距离_km", dist_pack), 0.0, cap)
        p_hvg_va = np.clip(apply_binner(va, "hollandvg", hvg_pack), 0.0, cap)
        p_dist_te = np.clip(apply_binner(te, "场站距离_km", dist_pack), 0.0, cap)
        p_hvg_te = np.clip(apply_binner(te, "hollandvg", hvg_pack), 0.0, cap)

        gate = fit_soft_alpha(tr, p_dist_tr, p_hvg_tr, va, p_dist_va, p_hvg_va)
        p_soft_te, alpha_te = apply_soft_alpha(te, p_dist_te, p_hvg_te, gate)
        p_soft_te = np.clip(p_soft_te, 0.0, cap)

        y = te["y_true_mw"].to_numpy(dtype=np.float32)
        p_base = np.clip(te["y_pred_mw"].to_numpy(dtype=np.float32), 0.0, cap)

        mb = metrics(y, p_base, cap)
        md = metrics(y, p_dist_te, cap)
        mh = metrics(y, p_hvg_te, cap)
        ms = metrics(y, p_soft_te, cap)

        event_name = str(te["台风中文名称"].dropna().iloc[0]) if "台风中文名称" in te.columns and te["台风中文名称"].notna().any() else ""
        rows.append(
            {
                "test_event": test_event,
                "test_event_name": event_name,
                "test_year": int(te["year"].mode().iloc[0]),
                "base_RMSE": mb["RMSE"],
                "dist_RMSE": md["RMSE"],
                "hvg_RMSE": mh["RMSE"],
                "soft_RMSE": ms["RMSE"],
                "base_MAE": mb["MAE"],
                "dist_MAE": md["MAE"],
                "hvg_MAE": mh["MAE"],
                "soft_MAE": ms["MAE"],
                "soft_vs_base_rmse_imp_pct": (mb["RMSE"] - ms["RMSE"]) / (mb["RMSE"] + 1e-9) * 100.0,
                "soft_vs_dist_rmse_imp_pct": (md["RMSE"] - ms["RMSE"]) / (md["RMSE"] + 1e-9) * 100.0,
                "soft_vs_base_mae_imp_pct": (mb["MAE"] - ms["MAE"]) / (mb["MAE"] + 1e-9) * 100.0,
                "soft_vs_dist_mae_imp_pct": (md["MAE"] - ms["MAE"]) / (md["MAE"] + 1e-9) * 100.0,
                "alpha_mean": float(np.mean(alpha_te)),
                "alpha_p90": float(np.quantile(alpha_te, 0.9)),
                "alpha_share_gt_0.5": float(np.mean(alpha_te > 0.5)),
            }
        )

    fold_df = pd.DataFrame(rows)
    if len(fold_df) == 0:
        raise RuntimeError("No valid folds")

    summary = {
        "folds": len(fold_df),
        "base_RMSE_mean": fold_df["base_RMSE"].mean(),
        "dist_RMSE_mean": fold_df["dist_RMSE"].mean(),
        "hvg_RMSE_mean": fold_df["hvg_RMSE"].mean(),
        "soft_RMSE_mean": fold_df["soft_RMSE"].mean(),
        "base_MAE_mean": fold_df["base_MAE"].mean(),
        "dist_MAE_mean": fold_df["dist_MAE"].mean(),
        "hvg_MAE_mean": fold_df["hvg_MAE"].mean(),
        "soft_MAE_mean": fold_df["soft_MAE"].mean(),
        "soft_vs_base_rmse_imp_mean_pct": fold_df["soft_vs_base_rmse_imp_pct"].mean(),
        "soft_vs_dist_rmse_imp_mean_pct": fold_df["soft_vs_dist_rmse_imp_pct"].mean(),
        "soft_vs_base_mae_imp_mean_pct": fold_df["soft_vs_base_mae_imp_pct"].mean(),
        "soft_vs_dist_mae_imp_mean_pct": fold_df["soft_vs_dist_mae_imp_pct"].mean(),
        "soft_vs_base_rmse_win_rate": (fold_df["soft_RMSE"] < fold_df["base_RMSE"]).mean(),
        "soft_vs_dist_rmse_win_rate": (fold_df["soft_RMSE"] < fold_df["dist_RMSE"]).mean(),
        "avg_alpha_mean": fold_df["alpha_mean"].mean(),
        "avg_alpha_share_gt_0.5": fold_df["alpha_share_gt_0.5"].mean(),
    }

    return fold_df, pd.DataFrame([summary])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-glob", default="data/predictions/bilstm_pred_*.csv")
    p.add_argument("--typhoon-path", default="data/typhoon/typhoon19-24complete_15min_within1000km_aug8.csv")
    p.add_argument("--capacity", type=float, default=200.0)
    p.add_argument("--out-dir", default="loto_results_softblend")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(args.pred_glob, args.typhoon_path)
    fold_df, summary_df = run_loto(df, args.capacity)

    fold_path = out_dir / "typhoon_loto_softblend_fold_metrics.csv"
    summary_path = out_dir / "typhoon_loto_softblend_summary.csv"

    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("saved", fold_path)
    print("saved", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
