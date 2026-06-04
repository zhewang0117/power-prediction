#!/usr/bin/env python3
"""
GLM / 岭回归校正器 LOTO 评估
基于论文的两阶段思路, 先用简单线性方法验证框架.

用法:
  python evaluate_typhoon_glm_loto.py --feature 场站距离_km --out-dir loto_results_glm
"""

import argparse, glob, math, os, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
# 使用 numpy 实现岭回归 (无需 sklearn)



# ======== 评价指标 ========

def metrics(y_true: np.ndarray, y_pred: np.ndarray, cap: float) -> Dict[str, float]:
    y_true = y_true.reshape(-1); y_pred = y_pred.reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae,
            "NRMSE": rmse / cap, "NMAE": mae / cap,
            "sMAPE": float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100.0)}


# ======== 数据加载 (与现有 LOTO 完全一致) ========

def pred_to_long(pred: pd.DataFrame, year: int, horizon: int = 96) -> pd.DataFrame:
    pred = pred.copy()
    pred["target_start_time"] = pd.to_datetime(pred["target_start_time"], errors="coerce")
    pred = pred.dropna(subset=["target_start_time"]).reset_index(drop=True)
    rows = []
    for _, r in pred.iterrows():
        t0 = r["target_start_time"]
        for k in range(1, horizon + 1):
            rows.append({
                "year": year, "target_start_time": t0,
                "target_time": t0 + pd.Timedelta(minutes=15 * (k - 1)),
                "lead": k,
                "y_true_mw": float(r[f"y_true_t+{k}_mw"]),
                "y_pred_mw": float(r[f"y_pred_t+{k}_mw"]),
            })
    return pd.DataFrame(rows)


def build_dataset(pred_glob: str, typhoon_path: str) -> pd.DataFrame:
    parts = []
    for fp in sorted(glob.glob(pred_glob)):
        m = re.search(r"(\d{4})", os.path.basename(fp))
        if not m: continue
        year = int(m.group(1))
        parts.append(pred_to_long(pd.read_csv(fp), year=year, horizon=96))
    if not parts: raise RuntimeError("No prediction files found")
    long_pred = pd.concat(parts, ignore_index=True)

    ty = pd.read_csv(typhoon_path, encoding="utf-8-sig")
    ty["target_time"] = pd.to_datetime(ty["target_time"], errors="coerce")
    ty = ty.dropna(subset=["target_time", "台风编号"]).copy()
    if "场站距离_km" in ty.columns:
        ty["场站距离_km"] = pd.to_numeric(ty["场站距离_km"], errors="coerce")
        ty = ty.sort_values(["target_time", "场站距离_km"]).drop_duplicates(subset=["target_time"], keep="first")
    else:
        ty = ty.sort_values(["target_time"]).drop_duplicates(subset=["target_time"], keep="first")

    m = long_pred.merge(ty, on="target_time", how="inner")
    if len(m) == 0: raise RuntimeError("No overlap between predictions and typhoon data")

    m = m.sort_values(["台风编号", "target_time", "lead"]).reset_index(drop=True)
    m["delta_mw"] = m["y_true_mw"] - m["y_pred_mw"]
    m["lead_norm"] = (m["lead"] - 1) / 95.0
    return m


# ======== GLM 校正器训练 ========

class RidgeRegressor:
    """纯 numpy 岭回归"""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        X_bias = np.column_stack([X, np.ones(n, dtype=X.dtype)])
        # (X^T X + αI)^-1 X^T y
        I = np.eye(X_bias.shape[1], dtype=X.dtype)
        I[-1, -1] = 0  # 不对截距项正则化
        self.coef_ = np.linalg.solve(X_bias.T @ X_bias + self.alpha * I, X_bias.T @ y)
        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


def train_glm_corrector(X_tr: np.ndarray, y_tr: np.ndarray,
                        X_va: np.ndarray, y_va: np.ndarray,
                        alpha: float = 1.0) -> Tuple[RidgeRegressor, float]:
    """训练岭回归校正器, 在验证集上调优."""
    model = RidgeRegressor(alpha=alpha)
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_va)
    val_rmse = math.sqrt(np.mean((val_pred - y_va) ** 2))
    return model, val_rmse


# ======== LOTO 核心 ========

def run_loto_glm(df: pd.DataFrame, feat_cols: List[str], capacity: float,
                 alpha: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """对指定特征组运行 GLM 校正器 LOTO 评估."""
    events = df.groupby("台风编号")["target_time"].min().sort_values().index.tolist()
    if len(events) < 4:
        raise RuntimeError(f"Need at least 4 typhoon events, got {len(events)}")

    fold_rows = []
    for test_event in events:
        remain = [e for e in events if e != test_event]
        remain_sorted = (
            df[df["台风编号"].isin(remain)]
            .groupby("台风编号")["target_time"].min().sort_values().index.tolist()
        )
        val_event = remain_sorted[-1]
        train_events = remain_sorted[:-1]

        tr = df[df["台风编号"].isin(train_events)].copy()
        va = df[df["台风编号"].isin([val_event])].copy()
        te = df[df["台风编号"].isin([test_event])].copy()
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            continue

        X_tr = tr[feat_cols].to_numpy(dtype=np.float32)
        y_tr = tr["delta_mw"].to_numpy(dtype=np.float32)
        X_va = va[feat_cols].to_numpy(dtype=np.float32)
        X_te = te[feat_cols].to_numpy(dtype=np.float32)

        # 标准化
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-6
        X_tr_n = (X_tr - mu) / sd
        X_va_n = (X_va - mu) / sd
        X_te_n = (X_te - mu) / sd

        # 训练 GLM 校正器
        model, val_rmse = train_glm_corrector(X_tr_n, y_tr, X_va_n,
                                               va["delta_mw"].to_numpy(dtype=np.float32), alpha)

        delta_te = model.predict(X_te_n)
        base_te = np.clip(te["y_pred_mw"].to_numpy(dtype=np.float32), 0.0, capacity)
        y_te = te["y_true_mw"].to_numpy(dtype=np.float32)
        corr_te = np.clip(base_te + delta_te, 0.0, capacity)

        mb = metrics(y_te, base_te, capacity)
        mc = metrics(y_te, corr_te, capacity)

        event_name = str(te["台风中文名称"].dropna().iloc[0]) if (
            "台风中文名称" in te.columns and te["台风中文名称"].notna().any()
        ) else ""
        test_year = int(te["year"].mode().iloc[0]) if len(te) else -1

        fold_rows.append({
            "test_event": test_event, "test_event_name": event_name, "test_year": test_year,
            "val_rmse": val_rmse,
            "base_RMSE": mb["RMSE"], "corr_RMSE": mc["RMSE"],
            "base_MAE": mb["MAE"], "corr_MAE": mc["MAE"],
            "rmse_improve_pct": (mb["RMSE"] - mc["RMSE"]) / (mb["RMSE"] + 1e-9) * 100.0,
            "mae_improve_pct": (mb["MAE"] - mc["MAE"]) / (mb["MAE"] + 1e-9) * 100.0,
        })

    fold_df = pd.DataFrame(fold_rows)
    if len(fold_df) == 0: raise RuntimeError("No valid folds")

    s = {
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
    return fold_df, pd.DataFrame([s])


# ======== 主流程 ========

def main():
    p = argparse.ArgumentParser(description="GLM Corrector LOTO")
    p.add_argument("--pred-glob", default="data/predictions/bilstm_pred_*.csv")
    p.add_argument("--typhoon-path", default="data/typhoon/typhoon19-24complete_15min_within1000km_aug8.csv")
    p.add_argument("--capacity", type=float, default=200.0)
    p.add_argument("--out-dir", default="loto_results_glm")
    p.add_argument("--feature", default="场站距离_km", help="Feature column name(s), comma-separated")
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    args = p.parse_args()

    feat_cols = [s.strip() for s in args.feature.split(",")]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("Loading dataset...")
    df = build_dataset(args.pred_glob, args.typhoon_path)
    print(f"Dataset: {len(df)} rows, {df['台风编号'].nunique()} typhoon events")

    # 检查特征
    available = [c for c in feat_cols if c in df.columns]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"WARNING: missing columns: {missing}")
    if not available:
        raise ValueError("No available features")
    print(f"Features: {available}")

    # 运行 LOTO
    print(f"Running GLM LOTO (alpha={args.alpha})...")
    fold_df, summary_df = run_loto_glm(df, available, args.capacity, args.alpha)

    # 保存
    fold_path = out_dir / "loto_fold_glm.csv"
    summary_path = out_dir / "loto_summary_glm.csv"
    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    s = summary_df.iloc[0]
    print(f"\n===== GLM Corrector Results ({'+'.join(available)}) =====")
    print(f"  Folds: {int(s['folds'])}")
    print(f"  Base  RMSE: {s['base_RMSE_mean']:.4f} ± {s['base_RMSE_std']:.4f}")
    print(f"  Corr  RMSE: {s['corr_RMSE_mean']:.4f} ± {s['corr_RMSE_std']:.4f}")
    print(f"  Base  MAE:  {s['base_MAE_mean']:.4f} ± {s['base_MAE_std']:.4f}")
    print(f"  Corr  MAE:  {s['corr_MAE_mean']:.4f} ± {s['corr_MAE_std']:.4f}")
    print(f"  RMSE Improv: {s['rmse_improve_pct_mean']:.2f}%")
    print(f"  MAE  Improv: {s['mae_improve_pct_mean']:.2f}%")
    print(f"  RMSE WinRt:  {s['rmse_win_rate']:.1%}")
    print(f"  MAE  WinRt:  {s['mae_win_rate']:.1%}")

    # 打印每个fold的详情
    print(f"\n  Per-fold breakdown:")
    for _, row in fold_df.iterrows():
        print(f"    {row['test_event_name']:8s} ({int(row['test_year'])}): "
              f"base RMSE={row['base_RMSE']:.2f} -> corr RMSE={row['corr_RMSE']:.2f} "
              f"({row['rmse_improve_pct']:+.1f}%)")

    print(f"\nSaved: {fold_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
