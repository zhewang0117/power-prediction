import argparse
import glob
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

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
        name = os.path.basename(fp)
        m = re.search(r"(\d{4})", name)
        if not m:
            continue
        year = int(m.group(1))
        pred = pd.read_csv(fp)
        parts.append(pred_to_long(pred, year=year, horizon=96))

    if not parts:
        raise RuntimeError("No prediction files found")

    long_pred = pd.concat(parts, ignore_index=True)

    ty = pd.read_csv(typhoon_path)
    ty["target_time"] = pd.to_datetime(ty["target_time"], errors="coerce")
    ty = ty.dropna(subset=["target_time", "台风编号"]).copy()

    use_cols = [
        "target_time",
        "台风编号",
        "台风中文名称",
        "经度",
        "纬度",
        "台风等级",
        "风速",
        "气压",
        "移动速度",
        "场站距离_km",
    ]
    ty = ty[use_cols].drop_duplicates(subset=["target_time"]).sort_values("target_time")

    m = long_pred.merge(ty, on="target_time", how="inner")
    if len(m) == 0:
        raise RuntimeError("Predictions and typhoon data do not overlap")

    m = m.sort_values(["台风编号", "target_time", "lead"]).reset_index(drop=True)
    m["delta_mw"] = m["y_true_mw"] - m["y_pred_mw"]
    return m


def fit_feature_binner(train: pd.DataFrame, feature: str):
    vals = train[feature].astype(float)
    q = np.quantile(vals, [0.0, 0.25, 0.5, 0.75, 1.0])
    q = np.unique(q)
    if len(q) < 3:
        return None

    train_bin = pd.cut(vals, bins=q, labels=False, include_lowest=True, duplicates="drop")
    gb = train.groupby([train["lead"], train_bin])["delta_mw"].mean()
    lead_bias = train.groupby("lead")["delta_mw"].mean()
    return q, gb, lead_bias


def apply_feature_binner(df: pd.DataFrame, feature: str, q, gb, lead_bias) -> np.ndarray:
    b = pd.cut(df[feature].astype(float), bins=q, labels=False, include_lowest=True, duplicates="drop")
    pred = []
    for lead, bin_id, base in zip(df["lead"].to_numpy(), b.to_numpy(), df["y_pred_mw"].to_numpy()):
        key = (lead, bin_id)
        if key in gb.index:
            d = gb[key]
        else:
            d = lead_bias.get(lead, 0.0)
        pred.append(base + d)
    return np.asarray(pred, dtype=np.float32)


def select_feature(train_df: pd.DataFrame, val_df: pd.DataFrame, candidate_feats: List[str], capacity: float):
    y_val = val_df["y_true_mw"].to_numpy(dtype=np.float32)

    best_feat = None
    best_pack = None
    best_rmse = float("inf")

    for feat in candidate_feats:
        pack = fit_feature_binner(train_df, feat)
        if pack is None:
            continue
        q, gb, lb = pack
        p_val = np.clip(apply_feature_binner(val_df, feat, q, gb, lb), 0.0, capacity)
        rmse = math.sqrt(np.mean((y_val - p_val) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_feat = feat
            best_pack = pack

    if best_feat is None:
        raise RuntimeError("No valid feature binner found on training split")

    return best_feat, best_pack, best_rmse


def run_loto(df: pd.DataFrame, capacity: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events = df.groupby("台风编号")["target_time"].min().sort_values().index.tolist()
    if len(events) < 4:
        raise RuntimeError("Need at least 4 typhoon events for LOTO + validation")

    candidate_feats = ["移动速度", "场站距离_km", "风速", "气压", "台风等级"]

    fold_rows = []

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

        train_df = df[df["台风编号"].isin(train_events)].copy()
        val_df = df[df["台风编号"].isin([val_event])].copy()
        test_df = df[df["台风编号"].isin([test_event])].copy()

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            continue

        feat, pack, val_rmse = select_feature(train_df, val_df, candidate_feats, capacity)
        q, gb, lb = pack

        y_true = test_df["y_true_mw"].to_numpy(dtype=np.float32)
        p_base = np.clip(test_df["y_pred_mw"].to_numpy(dtype=np.float32), 0.0, capacity)
        p_corr = np.clip(apply_feature_binner(test_df, feat, q, gb, lb), 0.0, capacity)

        m_base = metrics(y_true, p_base, capacity)
        m_corr = metrics(y_true, p_corr, capacity)

        event_name = str(test_df["台风中文名称"].dropna().iloc[0]) if test_df["台风中文名称"].notna().any() else ""
        test_year = int(test_df["year"].mode().iloc[0]) if len(test_df) else -1

        fold_rows.append(
            {
                "test_event": test_event,
                "test_event_name": event_name,
                "test_year": test_year,
                "selected_feature": feat,
                "val_rmse": val_rmse,
                "base_RMSE": m_base["RMSE"],
                "corr_RMSE": m_corr["RMSE"],
                "base_MAE": m_base["MAE"],
                "corr_MAE": m_corr["MAE"],
                "base_NMAE": m_base["NMAE"],
                "corr_NMAE": m_corr["NMAE"],
                "rmse_improve_pct": (m_base["RMSE"] - m_corr["RMSE"]) / (m_base["RMSE"] + 1e-9) * 100.0,
                "mae_improve_pct": (m_base["MAE"] - m_corr["MAE"]) / (m_base["MAE"] + 1e-9) * 100.0,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    if len(fold_df) == 0:
        raise RuntimeError("No valid LOTO folds produced")

    summary = {
        "folds": len(fold_df),
        "base_RMSE_mean": fold_df["base_RMSE"].mean(),
        "base_RMSE_std": fold_df["base_RMSE"].std(ddof=1),
        "corr_RMSE_mean": fold_df["corr_RMSE"].mean(),
        "corr_RMSE_std": fold_df["corr_RMSE"].std(ddof=1),
        "base_MAE_mean": fold_df["base_MAE"].mean(),
        "base_MAE_std": fold_df["base_MAE"].std(ddof=1),
        "corr_MAE_mean": fold_df["corr_MAE"].mean(),
        "corr_MAE_std": fold_df["corr_MAE"].std(ddof=1),
        "rmse_improve_pct_mean": fold_df["rmse_improve_pct"].mean(),
        "mae_improve_pct_mean": fold_df["mae_improve_pct"].mean(),
        "rmse_win_rate": (fold_df["corr_RMSE"] < fold_df["base_RMSE"]).mean(),
        "mae_win_rate": (fold_df["corr_MAE"] < fold_df["base_MAE"]).mean(),
    }
    summary_df = pd.DataFrame([summary])
    return fold_df, summary_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-glob", default="data/predictions/bilstm_pred_*.csv")
    p.add_argument("--typhoon-path", default="data/typhoon/typhoon19-24complete_15min_within1000km.csv")
    p.add_argument("--capacity", type=float, default=200.0)
    p.add_argument("--out-dir", default="loto_results")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(args.pred_glob, args.typhoon_path)

    fold_df, summary_df = run_loto(df, capacity=args.capacity)

    fold_path = out_dir / "typhoon_loto_fold_metrics.csv"
    summary_path = out_dir / "typhoon_loto_summary.csv"
    overlap_path = out_dir / "typhoon_overlap_overview.csv"

    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    overlap = (
        df.groupby(["台风编号", "台风中文名称", "year"]).size().reset_index(name="rows").sort_values(["year", "台风编号"])
    )
    overlap.to_csv(overlap_path, index=False, encoding="utf-8-sig")

    print("saved", fold_path)
    print("saved", summary_path)
    print("saved", overlap_path)
    print("===== LOTO Summary =====")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
