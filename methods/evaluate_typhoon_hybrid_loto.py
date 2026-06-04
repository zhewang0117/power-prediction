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
    # Multiple events can share the same timestamp; keep the closest one to the station for stable matching.
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


def select_hybrid_on_val(train_df: pd.DataFrame, val_df: pd.DataFrame, cap: float):
    dist_pack = fit_binner(train_df, "场站距离_km")
    hvg_pack = fit_binner(train_df, "hollandvg")
    if dist_pack is None:
        raise RuntimeError("Failed to fit distance binner")

    # Distance-only fallback.
    p_dist_val = np.clip(apply_binner(val_df, "场站距离_km", dist_pack), 0.0, cap)

    if hvg_pack is None:
        y = val_df["y_true_mw"].to_numpy(dtype=np.float32)
        rmse = math.sqrt(np.mean((y - p_dist_val) ** 2))
        return {
            "mode": "distance_only",
            "dist_pack": dist_pack,
            "hvg_pack": None,
            "th_hvg": None,
            "th_dist": None,
            "val_rmse": rmse,
        }

    p_hvg_val = np.clip(apply_binner(val_df, "hollandvg", hvg_pack), 0.0, cap)
    y_val = val_df["y_true_mw"].to_numpy(dtype=np.float32)

    best = {
        "mode": "distance_only",
        "dist_pack": dist_pack,
        "hvg_pack": hvg_pack,
        "th_hvg": None,
        "th_dist": None,
        "val_rmse": math.sqrt(np.mean((y_val - p_dist_val) ** 2)),
    }

    # Grid search: use holland only when strong local storm signature and close enough.
    hvg_vals = pd.to_numeric(train_df["hollandvg"], errors="coerce").dropna()
    if len(hvg_vals) == 0:
        return best

    hvg_thresholds = np.quantile(hvg_vals, [0.5, 0.6, 0.7, 0.8, 0.9])
    dist_thresholds = [200.0, 300.0, 400.0, 500.0, 700.0]

    dist_val = pd.to_numeric(val_df["场站距离_km"], errors="coerce").to_numpy()
    hvg_val = pd.to_numeric(val_df["hollandvg"], errors="coerce").to_numpy()

    for th_hvg in hvg_thresholds:
        for th_dist in dist_thresholds:
            use_hvg = (hvg_val >= th_hvg) & (dist_val <= th_dist)
            p_mix = np.where(use_hvg, p_hvg_val, p_dist_val)
            p_mix = np.clip(p_mix, 0.0, cap)
            rmse = math.sqrt(np.mean((y_val - p_mix) ** 2))
            if rmse < best["val_rmse"]:
                best = {
                    "mode": "hybrid",
                    "dist_pack": dist_pack,
                    "hvg_pack": hvg_pack,
                    "th_hvg": float(th_hvg),
                    "th_dist": float(th_dist),
                    "val_rmse": rmse,
                }

    return best


def apply_hybrid(df: pd.DataFrame, cfg: Dict, cap: float) -> np.ndarray:
    p_dist = np.clip(apply_binner(df, "场站距离_km", cfg["dist_pack"]), 0.0, cap)
    if cfg["mode"] != "hybrid" or cfg["hvg_pack"] is None:
        return p_dist

    p_hvg = np.clip(apply_binner(df, "hollandvg", cfg["hvg_pack"]), 0.0, cap)
    dist_vals = pd.to_numeric(df["场站距离_km"], errors="coerce").to_numpy()
    hvg_vals = pd.to_numeric(df["hollandvg"], errors="coerce").to_numpy()
    use_hvg = (hvg_vals >= cfg["th_hvg"]) & (dist_vals <= cfg["th_dist"])
    p_mix = np.where(use_hvg, p_hvg, p_dist)
    return np.clip(p_mix, 0.0, cap)


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

        cfg = select_hybrid_on_val(tr, va, cap)

        y = te["y_true_mw"].to_numpy(dtype=np.float32)
        p_base = np.clip(te["y_pred_mw"].to_numpy(dtype=np.float32), 0.0, cap)
        p_dist = np.clip(apply_binner(te, "场站距离_km", cfg["dist_pack"]), 0.0, cap)
        p_mix = apply_hybrid(te, cfg, cap)

        mb = metrics(y, p_base, cap)
        md = metrics(y, p_dist, cap)
        mm = metrics(y, p_mix, cap)

        event_name = str(te["台风中文名称"].dropna().iloc[0]) if "台风中文名称" in te.columns and te["台风中文名称"].notna().any() else ""
        rows.append(
            {
                "test_event": test_event,
                "test_event_name": event_name,
                "test_year": int(te["year"].mode().iloc[0]),
                "mode": cfg["mode"],
                "th_hvg": cfg["th_hvg"],
                "th_dist": cfg["th_dist"],
                "base_RMSE": mb["RMSE"],
                "dist_RMSE": md["RMSE"],
                "hybrid_RMSE": mm["RMSE"],
                "base_MAE": mb["MAE"],
                "dist_MAE": md["MAE"],
                "hybrid_MAE": mm["MAE"],
                "hybrid_vs_base_rmse_imp_pct": (mb["RMSE"] - mm["RMSE"]) / (mb["RMSE"] + 1e-9) * 100.0,
                "hybrid_vs_dist_rmse_imp_pct": (md["RMSE"] - mm["RMSE"]) / (md["RMSE"] + 1e-9) * 100.0,
                "hybrid_vs_base_mae_imp_pct": (mb["MAE"] - mm["MAE"]) / (mb["MAE"] + 1e-9) * 100.0,
                "hybrid_vs_dist_mae_imp_pct": (md["MAE"] - mm["MAE"]) / (md["MAE"] + 1e-9) * 100.0,
            }
        )

    fold_df = pd.DataFrame(rows)
    summary = {
        "folds": len(fold_df),
        "base_RMSE_mean": fold_df["base_RMSE"].mean(),
        "dist_RMSE_mean": fold_df["dist_RMSE"].mean(),
        "hybrid_RMSE_mean": fold_df["hybrid_RMSE"].mean(),
        "base_MAE_mean": fold_df["base_MAE"].mean(),
        "dist_MAE_mean": fold_df["dist_MAE"].mean(),
        "hybrid_MAE_mean": fold_df["hybrid_MAE"].mean(),
        "hybrid_vs_base_rmse_imp_mean_pct": fold_df["hybrid_vs_base_rmse_imp_pct"].mean(),
        "hybrid_vs_dist_rmse_imp_mean_pct": fold_df["hybrid_vs_dist_rmse_imp_pct"].mean(),
        "hybrid_vs_base_mae_imp_mean_pct": fold_df["hybrid_vs_base_mae_imp_pct"].mean(),
        "hybrid_vs_dist_mae_imp_mean_pct": fold_df["hybrid_vs_dist_mae_imp_pct"].mean(),
        "hybrid_vs_base_rmse_win_rate": (fold_df["hybrid_RMSE"] < fold_df["base_RMSE"]).mean(),
        "hybrid_vs_dist_rmse_win_rate": (fold_df["hybrid_RMSE"] < fold_df["dist_RMSE"]).mean(),
        "hybrid_mode_ratio": (fold_df["mode"] == "hybrid").mean(),
    }

    return fold_df, pd.DataFrame([summary])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-glob", default="data/predictions/bilstm_pred_*.csv")
    p.add_argument("--typhoon-path", default="data/typhoon/typhoon19-24complete_15min_within1000km_aug8.csv")
    p.add_argument("--capacity", type=float, default=200.0)
    p.add_argument("--out-dir", default="loto_results_hybrid")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(args.pred_glob, args.typhoon_path)
    fold_df, summary_df = run_loto(df, args.capacity)

    fold_path = out_dir / "typhoon_loto_hybrid_fold_metrics.csv"
    summary_path = out_dir / "typhoon_loto_hybrid_summary.csv"

    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("saved", fold_path)
    print("saved", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
