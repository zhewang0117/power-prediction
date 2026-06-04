#!/usr/bin/env python3
"""
GWO-ELM 校正器 LOTO 评估框架
基于: 梁志峰等, 2025, "台风天气下风电功率预测偏差超短期预警与误差修正方法"

核心方法:
  1. ELM (极限学习机) 使用 GBF (高斯基函数) 激活
  2. GWO (灰狼优化) 优化 ELM 输入权重和偏置
  3. 融合误差特性的两阶段校正 (可配置)
  4. LOTO (留一台风) 交叉验证
  5. 多特征组对比实验

用法:
  python evaluate_typhoon_gwo_elm_loto.py --features all
  python evaluate_typhoon_gwo_elm_loto.py --features basic_5,physical_8
"""

import argparse
import glob
import math
import os
import re
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

# ============================================================================
# 特征组定义
# ============================================================================

FEATURE_GROUPS = {
    # 仅距离
    "dist_only": ["场站距离_km"],

    # 论文风格: 距离 + 短期预测功率 + 提前步数
    "paper_style": ["场站距离_km", "y_pred_mw", "lead_norm"],

    # 距离 + Holland 梯度风速
    "dist_holland": ["场站距离_km", "hollandvg"],

    # 距离 + Holland 风速 + 预测功率 + 提前步数
    "dist_holland_pred_lead": ["场站距离_km", "hollandvg", "y_pred_mw", "lead_norm"],

    # --- 以下保留供扩展 ---
    "basic_5": ["移动速度", "场站距离_km", "风速", "气压", "台风等级"],
    "physical_8": ["rmax", "b", "hollandvg", "rmax-ym", "b-ym",
                   "hollandvg-ym", "ym风速", "ym风速(<=250km)"],
    "holland_3": ["rmax", "b", "hollandvg"],
    "yanmeng_5": ["rmax-ym", "b-ym", "hollandvg-ym", "ym风速", "ym风速(<=250km)"],
    "wind_feats": ["风速", "hollandvg", "ym风速"],
    "dist_pred_wind": ["场站距离_km", "y_pred_mw", "lead_norm", "风速", "hollandvg", "ym风速"],
    "combined_all": ["移动速度", "场站距离_km", "风速", "气压", "台风等级",
                     "rmax", "b", "hollandvg", "rmax-ym", "b-ym", "hollandvg-ym",
                     "ym风速", "ym风速(<=250km)",
                     "方位角", "相对方位角", "移动速度(m/s)", "移动方向(°)",
                     "y_pred_mw", "lead_norm"],
    "physics_core": ["场站距离_km", "hollandvg", "ym风速"],
}


# ============================================================================
# 评价指标
# ============================================================================

def metrics(y_true: np.ndarray, y_pred: np.ndarray, cap: float) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    nrmse = rmse / cap
    nmae = mae / cap
    smape = float(
        np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100.0
    )
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "NRMSE": nrmse, "NMAE": nmae, "sMAPE": smape}


# ============================================================================
# ELM 实现
# ============================================================================

class ELM:
    """极限学习机 (Extreme Learning Machine)

    支持多种激活函数, 使用 Moore-Penrose 伪逆求解输出权重.
    可选用 PyTorch GPU 加速矩阵运算.
    """

    def __init__(self, n_hidden: int = 80, gamma: float = 1.0,
                 activation: str = "gbf", use_gpu: bool = True,
                 ridge: float = 0.0):
        self.n_hidden = n_hidden
        self.gamma = gamma
        self.activation = activation
        self.ridge = ridge
        self.input_weights = None
        self.biases = None
        self.beta = None
        # GPU 初始化
        self._device = None
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
            except Exception:
                pass

    def _to_tensor(self, *arrays: np.ndarray):
        """将 numpy 数组转为 GPU tensor (float32)."""
        if self._device is None:
            return arrays
        import torch
        return (torch.from_numpy(a).to(self._device, non_blocking=True) for a in arrays)

    def _activate_torch(self, X_t: "torch.Tensor") -> "torch.Tensor":
        import torch
        if self.activation == "gbf":
            return torch.exp(-self.gamma * torch.clamp(X_t, -50, 50) ** 2)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + torch.exp(-torch.clamp(X_t, -30, 30)))
        elif self.activation == "relu":
            return torch.clamp(X_t, min=0)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activate(self, X: np.ndarray) -> np.ndarray:
        if self.activation == "gbf":
            return np.exp(-self.gamma * np.clip(X, -50, 50) ** 2)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(X, -30, 30)))
        elif self.activation == "relu":
            return np.maximum(0, X)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def fit(self, X: np.ndarray, y: np.ndarray,
            input_weights: Optional[np.ndarray] = None,
            biases: Optional[np.ndarray] = None):
        n = X.shape[0]
        if input_weights is None or biases is None:
            self.input_weights = np.random.randn(X.shape[1], self.n_hidden).astype(np.float32) * 0.1
            self.biases = np.random.randn(self.n_hidden).astype(np.float32) * 0.1
        else:
            self.input_weights = input_weights
            self.biases = biases

        if self._device is not None and n > 10000:
            import torch
            X_t, y_t = self._to_tensor(X, y)
            W_t = torch.from_numpy(self.input_weights).to(self._device, non_blocking=True)
            b_t = torch.from_numpy(self.biases).to(self._device, non_blocking=True)
            with torch.no_grad():
                H_t = self._activate_torch(X_t @ W_t + b_t)
                ones = torch.ones(n, 1, dtype=torch.float32, device=self._device)
                H_t = torch.cat([H_t, ones], dim=1)
                HtH = H_t.T @ H_t
                Hty = H_t.T @ y_t.unsqueeze(1)
                if self.ridge > 0:
                    I = torch.eye(HtH.shape[0], dtype=torch.float32, device=self._device)
                    try:
                        beta_t = torch.linalg.solve(HtH + self.ridge * I, Hty)
                    except RuntimeError:
                        beta_t = torch.linalg.lstsq(H_t, y_t.unsqueeze(1)).solution
                else:
                    try:
                        beta_t = torch.linalg.solve(HtH + 1e-8 * torch.eye(
                            HtH.shape[0], dtype=torch.float32, device=self._device), Hty)
                    except RuntimeError:
                        beta_t = torch.linalg.lstsq(H_t, y_t.unsqueeze(1)).solution
            self.beta = beta_t.cpu().numpy().ravel()
        else:
            H = self._activate(X @ self.input_weights + self.biases)
            H = np.column_stack([H, np.ones(n, dtype=np.float32)])
            HtH = H.T @ H
            Hty = H.T @ y
            if self.ridge > 0:
                I = np.eye(HtH.shape[0], dtype=np.float32)
                try:
                    self.beta = np.linalg.solve(HtH + self.ridge * I, Hty)
                except np.linalg.LinAlgError:
                    self.beta = np.linalg.lstsq(H, y, rcond=None)[0]
            else:
                try:
                    self.beta = np.linalg.solve(HtH + 1e-8 * np.eye(HtH.shape[0], dtype=np.float32), Hty)
                except np.linalg.LinAlgError:
                    self.beta = np.linalg.lstsq(H, y, rcond=None)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._device is not None and len(X) > 10000:
            import torch
            X_t, = self._to_tensor(X)
            W_t = torch.from_numpy(self.input_weights).to(self._device, non_blocking=True)
            b_t = torch.from_numpy(self.biases).to(self._device, non_blocking=True)
            beta_t = torch.from_numpy(self.beta).to(self._device, non_blocking=True)
            with torch.no_grad():
                H_t = self._activate_torch(X_t @ W_t + b_t)
                ones = torch.ones(H_t.shape[0], 1, dtype=torch.float32, device=self._device)
                H_t = torch.cat([H_t, ones], dim=1)
                pred_t = H_t @ beta_t.unsqueeze(1)
            return pred_t.cpu().numpy().ravel()
        else:
            H = self._activate(X @ self.input_weights + self.biases)
            H = np.column_stack([H, np.ones(H.shape[0], dtype=np.float32)])
            return H @ self.beta

    def save(self, path: str):
        np.savez_compressed(path,
            input_weights=self.input_weights, biases=self.biases, beta=self.beta,
            n_hidden=self.n_hidden, gamma=self.gamma,
            activation=self.activation)

    @staticmethod
    def load(path: str) -> "ELM":
        d = np.load(path, allow_pickle=False)
        elm = ELM(n_hidden=int(d["n_hidden"]), gamma=float(d["gamma"]),
                  activation=str(d["activation"]), use_gpu=True)
        elm.input_weights = d["input_weights"]
        elm.biases = d["biases"]
        elm.beta = d["beta"]
        return elm

    def get_params_vec(self) -> np.ndarray:
        return np.concatenate([self.input_weights.ravel(), self.biases.ravel()])

    def set_params_vec(self, vec: np.ndarray, n_in: int):
        n_h = self.n_hidden
        self.input_weights = vec[:n_in * n_h].reshape(n_in, n_h).astype(np.float32)
        self.biases = vec[n_in * n_h:].astype(np.float32)


# ============================================================================
# GWO 实现
# ============================================================================

class GWO:
    """灰狼优化 (Grey Wolf Optimizer)

    用于优化 ELM 的输入权重和偏置, 提升模型稳定性和预测性能.
    """

    def __init__(self, n_wolves: int = 20, max_iter: int = 30, bound: float = 1.0):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.bound = bound
        self.alpha_pos = None
        self.alpha_score = float("inf")
        self.convergence = []

    def optimize(self, dim: int, fitness_fn: Callable) -> Tuple[np.ndarray, float]:
        # 初始化狼群
        positions = np.random.uniform(-self.bound, self.bound,
                                       (self.n_wolves, dim)).astype(np.float32)
        alpha_pos = np.zeros(dim, dtype=np.float32)
        alpha_score = float("inf")
        beta_pos = np.zeros(dim, dtype=np.float32)
        beta_score = float("inf")
        delta_pos = np.zeros(dim, dtype=np.float32)
        delta_score = float("inf")

        for t in range(self.max_iter):
            a = 2.0 * (1.0 - t / self.max_iter)  # 从2线性减小到0

            # 评估所有狼的适应度
            for i in range(self.n_wolves):
                fitness = fitness_fn(positions[i])

                if fitness < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = fitness
                    alpha_pos = positions[i].copy()
                elif fitness < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = fitness
                    beta_pos = positions[i].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_pos = positions[i].copy()

            # 更新位置
            for i in range(self.n_wolves):
                r1 = np.random.random(dim).astype(np.float32)
                r2 = np.random.random(dim).astype(np.float32)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - positions[i])
                X1 = alpha_pos - A1 * D_alpha

                r1 = np.random.random(dim).astype(np.float32)
                r2 = np.random.random(dim).astype(np.float32)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta_pos - positions[i])
                X2 = beta_pos - A2 * D_beta

                r1 = np.random.random(dim).astype(np.float32)
                r2 = np.random.random(dim).astype(np.float32)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta_pos - positions[i])
                X3 = delta_pos - A3 * D_delta

                positions[i] = np.clip((X1 + X2 + X3) / 3.0, -self.bound, self.bound)

            self.convergence.append(alpha_score)

        self.alpha_pos = alpha_pos
        self.alpha_score = alpha_score
        return alpha_pos, alpha_score


# ============================================================================
# GWO-ELM 校正器
# ============================================================================

def _make_elm_fitness(elm: ELM, X_tr: np.ndarray, y_tr: np.ndarray,
                      X_va: np.ndarray, y_va: np.ndarray) -> Callable:
    """构造 GWO 适应度函数: 在验证集上的 MSE"""
    n_in = X_tr.shape[1]
    def fitness(params: np.ndarray) -> float:
        elm.set_params_vec(params, n_in)
        elm.fit(X_tr, y_tr, input_weights=elm.input_weights, biases=elm.biases)
        y_pred = elm.predict(X_va)
        return float(np.mean((y_pred - y_va) ** 2))
    return fitness


def train_elm_simple(X_tr: np.ndarray, y_tr: np.ndarray,
                     X_va: np.ndarray, y_va: np.ndarray,
                     n_hidden: int = 80, gamma: float = 1.0,
                     n_trials: int = 20,
                     max_train_rows: int = 0,
                     ridge: float = 0.0) -> Tuple[ELM, float]:
    """简单 ELM 训练: 多次随机初始化, 在验证集上选最优. 无 GWO, 速度快."""
    best_model = None
    best_rmse = float("inf")

    # 训练集采样加速
    if max_train_rows > 0 and len(X_tr) > max_train_rows:
        idx = np.random.choice(len(X_tr), max_train_rows, replace=False)
        X_tr_sampled = X_tr[idx]
        y_tr_sampled = y_tr[idx]
    else:
        X_tr_sampled = X_tr
        y_tr_sampled = y_tr

    for _ in range(n_trials):
        elm = ELM(n_hidden=n_hidden, gamma=gamma, activation="gbf", ridge=ridge)
        elm.fit(X_tr_sampled, y_tr_sampled)

        val_pred = elm.predict(X_va)
        val_rmse = math.sqrt(float(np.mean((val_pred - y_va) ** 2)))

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = elm

    return best_model, best_rmse


def train_elm_gwo(X_tr: np.ndarray, y_tr: np.ndarray,
                  X_va: np.ndarray, y_va: np.ndarray,
                  n_hidden: int = 80, gamma: float = 1.0,
                  n_wolves: int = 20, max_iter: int = 30,
                  n_trials: int = 3,
                  ridge: float = 0.0) -> Tuple[ELM, float]:
    """训练 GWO-ELM 校正器

    使用 GWO 优化 ELM 输入权重和偏置, 在验证集上选最优.
    多次随机尝试取最佳, 以降低随机性影响.
    """
    best_model = None
    best_rmse = float("inf")

    for trial in range(n_trials):
        base_elm = ELM(n_hidden=n_hidden, gamma=gamma, activation="gbf", ridge=ridge)

        # 随机初始化一套权重/偏置作为基线
        base_elm.fit(X_tr, y_tr)
        dim = base_elm.input_weights.size + base_elm.biases.size
        base_vec = base_elm.get_params_vec()

        # 用 GWO 优化
        fitness_fn = _make_elm_fitness(base_elm, X_tr, y_tr, X_va, y_va)
        gwo = GWO(n_wolves=n_wolves, max_iter=max_iter, bound=1.0)

        # 用随机初始化作为 GWO 的一个起点
        positions = np.random.uniform(-1, 1, (gwo.n_wolves - 1, dim)).astype(np.float32)
        best_pos, best_fit = gwo.optimize(dim, fitness_fn)

        # 在最佳 GWO 位置重新训练
        elm = ELM(n_hidden=n_hidden, gamma=gamma, activation="gbf", ridge=ridge)
        elm.fit(X_tr, y_tr)
        elm.set_params_vec(best_pos, X_tr.shape[1])
        elm.fit(X_tr, y_tr, input_weights=elm.input_weights, biases=elm.biases)

        val_pred = elm.predict(X_va)
        val_rmse = math.sqrt(np.mean((val_pred - y_va) ** 2))

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = elm

    return best_model, best_rmse


def train_elm_gwo_two_stage(X_tr: np.ndarray, y_tr: np.ndarray,
                             X_va: np.ndarray, y_va: np.ndarray,
                             n_hidden: int = 80, gamma: float = 1.0,
                             n_wolves: int = 20, max_iter: int = 30,
                             n_trials: int = 3,
                             ridge: float = 0.0) -> Tuple[ELM, Optional[ELM], float]:
    """两阶段 GWO-ELM 校正器 (融合误差特性)

    阶段 1: GWO-ELM 预测 Δ = y_true - y_pred
    阶段 2: 以阶段 1 预测值为输入, 预测阶段 1 的残差
    结合: 最终校正 = base + Δ̂₁ + λ·Δ̂₂

    返回 (stage1_model, stage2_model, lambda_coef)
    """
    # Stage 1: GWO-ELM
    model1, _ = train_elm_gwo(X_tr, y_tr, X_va, y_va,
                                n_hidden=n_hidden, gamma=gamma,
                                n_wolves=n_wolves, max_iter=max_iter,
                                n_trials=n_trials, ridge=ridge)

    # Stage 1 在训练集上的残差
    pred_tr = model1.predict(X_tr)
    residual_tr = y_tr - pred_tr  # e1

    # Stage 2: 用 pred_tr 作为输入, residual_tr 作为目标
    # 对于 stage2, 输入是 stage1 输出 (1维), 训练一个 ELM 来预测残差
    model2 = ELM(n_hidden=max(10, n_hidden // 4), gamma=gamma, activation="gbf", ridge=ridge)
    X_stage2_tr = pred_tr.reshape(-1, 1)
    model2.fit(X_stage2_tr, residual_tr)

    # 在验证集上确定最优 λ
    pred_va = model1.predict(X_va)
    residual_pred_va = model2.predict(pred_va.reshape(-1, 1))

    # 网格搜索 λ ∈ [0, 2]
    best_lambda = 1.0
    best_val_mse = float("inf")
    for lam in np.linspace(0, 2, 41):
        corrected = pred_va + lam * residual_pred_va
        mse = float(np.mean((corrected - y_va) ** 2))
        if mse < best_val_mse:
            best_val_mse = mse
            best_lambda = lam

    return model1, model2, best_lambda


# ============================================================================
# 数据加载
# ============================================================================

def pred_to_long(pred: pd.DataFrame, year: int, horizon: int = 96) -> pd.DataFrame:
    pred = pred.copy()
    pred["target_start_time"] = pd.to_datetime(pred["target_start_time"], errors="coerce")
    pred = pred.dropna(subset=["target_start_time"]).reset_index(drop=True)

    n = len(pred)
    # 向量化展开: 每行重复 horizon 次
    lead = np.tile(np.arange(1, horizon + 1), n)
    base_times = pred["target_start_time"].values.repeat(horizon)
    offsets = pd.to_timedelta(np.tile(np.arange(horizon), n) * 15, unit="m")

    y_true_cols = [f"y_true_t+{k}_mw" for k in range(1, horizon + 1)]
    y_pred_cols = [f"y_pred_t+{k}_mw" for k in range(1, horizon + 1)]

    return pd.DataFrame({
        "year": np.full(n * horizon, year, dtype=np.int32),
        "target_start_time": base_times,
        "target_time": base_times + offsets,
        "lead": lead,
        "y_true_mw": pred[y_true_cols].values.ravel(),
        "y_pred_mw": pred[y_pred_cols].values.ravel(),
    })


def build_dataset(pred_glob: str, typhoon_path: str) -> pd.DataFrame:
    parts = []
    for fp in sorted(glob.glob(pred_glob)):
        m = re.search(r"(\d{4})", os.path.basename(fp))
        if not m:
            continue
        year = int(m.group(1))
        pred = pd.read_csv(fp)
        parts.append(pred_to_long(pred, year=year, horizon=96))

    if not parts:
        raise RuntimeError("No prediction files found")

    long_pred = pd.concat(parts, ignore_index=True)

    ty = pd.read_csv(typhoon_path, encoding="utf-8-sig")
    ty["target_time"] = pd.to_datetime(ty["target_time"], errors="coerce")
    ty = ty.dropna(subset=["target_time", "台风编号"]).copy()

    # 多台风共享同一时刻时, 保留距离最近的
    if "场站距离_km" in ty.columns:
        ty["场站距离_km"] = pd.to_numeric(ty["场站距离_km"], errors="coerce")
        ty = ty.sort_values(["target_time", "场站距离_km"]).drop_duplicates(
            subset=["target_time"], keep="first"
        )
    else:
        ty = ty.sort_values(["target_time"]).drop_duplicates(
            subset=["target_time"], keep="first"
        )

    m = long_pred.merge(ty, on="target_time", how="inner")
    if len(m) == 0:
        raise RuntimeError("Predictions and typhoon data do not overlap")

    m = m.sort_values(["台风编号", "target_time", "lead"]).reset_index(drop=True)
    m["delta_mw"] = m["y_true_mw"] - m["y_pred_mw"]
    m["lead_norm"] = (m["lead"] - 1) / 95.0

    return m


# ============================================================================
# LOTO 核心
# ============================================================================

def run_loto_feature(df: pd.DataFrame, feat_cols: List[str],
                     capacity: float, two_stage: bool = False,
                     simple_elm: bool = True,
                     n_hidden: int = 80, gamma: float = 1.0,
                     n_wolves: int = 20, max_iter: int = 30,
                     n_trials: int = 3,
                     max_train_rows: int = 0,
                     ridge: float = 0.0,
                     model_dir: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """对指定特征组运行 LOTO 评估.

    返回 (fold_df, summary_df).
    """
    events = df.groupby("台风编号")["target_time"].min().sort_values().index.tolist()
    if len(events) < 4:
        raise RuntimeError("Need at least 4 typhoon events for LOTO")

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    n_events = len(events)
    fold_rows = []
    for fold_i, test_event in enumerate(events, 1):
        t_fold = time.time()
        remain = [e for e in events if e != test_event]
        remain_sorted = (
            df[df["台风编号"].isin(remain)]
            .groupby("台风编号")["target_time"].min().sort_values().index.tolist()
        )
        val_event = remain_sorted[-1]
        train_events = remain_sorted[:-1]

        train_df = df[df["台风编号"].isin(train_events)].copy()
        val_df = df[df["台风编号"].isin([val_event])].copy()
        test_df = df[df["台风编号"].isin([test_event])].copy()

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            continue

        # 准备特征和目标
        X_tr = train_df[feat_cols].to_numpy(dtype=np.float32)
        y_tr = train_df["delta_mw"].to_numpy(dtype=np.float32)
        X_va = val_df[feat_cols].to_numpy(dtype=np.float32)
        y_va = val_df["delta_mw"].to_numpy(dtype=np.float32)
        X_te = test_df[feat_cols].to_numpy(dtype=np.float32)

        # 标准化 (只用训练集统计量)
        feat_mean = X_tr.mean(axis=0, keepdims=True)
        feat_std = X_tr.std(axis=0, keepdims=True) + 1e-6
        X_tr_norm = (X_tr - feat_mean) / feat_std
        X_va_norm = (X_va - feat_mean) / feat_std
        X_te_norm = (X_te - feat_mean) / feat_std

        # 训练校正器
        y_te = test_df["y_true_mw"].to_numpy(dtype=np.float32)
        base_te = np.clip(test_df["y_pred_mw"].to_numpy(dtype=np.float32), 0.0, capacity)

        event_name = str(test_df["台风中文名称"].dropna().iloc[0]) if (
            "台风中文名称" in test_df.columns and test_df["台风中文名称"].notna().any()
        ) else str(test_event)

        if two_stage:
            model1, model2, lam = train_elm_gwo_two_stage(
                X_tr_norm, y_tr, X_va_norm, y_va,
                n_hidden=n_hidden, gamma=gamma,
                n_wolves=n_wolves, max_iter=max_iter, n_trials=n_trials,
                ridge=ridge,
            )
            delta_te = model1.predict(X_te_norm)
            residual_te = model2.predict(delta_te.reshape(-1, 1))
            delta_te = delta_te + lam * residual_te
            if model_dir:
                model1.save(f"{model_dir}/{event_name}_stage1.npz")
                model2.save(f"{model_dir}/{event_name}_stage2.npz")
        elif simple_elm:
            model, _ = train_elm_simple(
                X_tr_norm, y_tr, X_va_norm, y_va,
                n_hidden=n_hidden, gamma=gamma,
                n_trials=n_trials,
                max_train_rows=max_train_rows,
                ridge=ridge,
            )
            delta_te = model.predict(X_te_norm)
            if model_dir:
                model.save(f"{model_dir}/{event_name}.npz")
        else:
            model, _ = train_elm_gwo(
                X_tr_norm, y_tr, X_va_norm, y_va,
                n_hidden=n_hidden, gamma=gamma,
                n_wolves=n_wolves, max_iter=max_iter, n_trials=n_trials,
                ridge=ridge,
            )
            delta_te = model.predict(X_te_norm)
            if model_dir:
                model.save(f"{model_dir}/{event_name}.npz")

        corrected_te = np.clip(base_te + delta_te, 0.0, capacity)
        m_base = metrics(y_te, base_te, capacity)
        m_corr = metrics(y_te, corrected_te, capacity)

        imp = (m_base["RMSE"] - m_corr["RMSE"]) / (m_base["RMSE"] + 1e-9) * 100.0
        t_elapsed = time.time() - t_fold
        print(f"    [{fold_i}/{n_events}] {event_name} | "
              f"RMSE: {m_base['RMSE']:.2f}→{m_corr['RMSE']:.2f} ({imp:+.2f}%) | {t_elapsed:.0f}s")
        sys.stdout.flush()

        test_year = int(test_df["year"].mode().iloc[0]) if len(test_df) else -1

        fold_rows.append({
            "test_event": test_event,
            "test_event_name": event_name,
            "test_year": test_year,
            "base_RMSE": m_base["RMSE"],
            "corr_RMSE": m_corr["RMSE"],
            "base_MAE": m_base["MAE"],
            "corr_MAE": m_corr["MAE"],
            "base_NMAE": m_base["NMAE"],
            "corr_NMAE": m_corr["NMAE"],
            "base_sMAPE": m_base["sMAPE"],
            "corr_sMAPE": m_corr["sMAPE"],
            "rmse_improve_pct": imp,
            "mae_improve_pct": (m_base["MAE"] - m_corr["MAE"]) / (m_base["MAE"] + 1e-9) * 100.0,
        })

    fold_df = pd.DataFrame(fold_rows)
    if len(fold_df) == 0:
        raise RuntimeError("No valid LOTO folds")

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
        "base_sMAPE_mean": fold_df["base_sMAPE"].mean(),
        "corr_sMAPE_mean": fold_df["corr_sMAPE"].mean(),
        "rmse_improve_pct_mean": fold_df["rmse_improve_pct"].mean(),
        "mae_improve_pct_mean": fold_df["mae_improve_pct"].mean(),
        "rmse_win_rate": (fold_df["corr_RMSE"] < fold_df["base_RMSE"]).mean(),
        "mae_win_rate": (fold_df["corr_MAE"] < fold_df["base_MAE"]).mean(),
    }
    summary_df = pd.DataFrame([summary])
    return fold_df, summary_df


# ============================================================================
# 主流程
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="GWO-ELM Corrector LOTO Evaluation")

    p.add_argument("--pred-glob",
                   default="data/predictions/bilstm_pred_*.csv")
    p.add_argument("--typhoon-path",
                   default="data/typhoon/typhoon19-24complete_15min_within1000km_aug8.csv")
    p.add_argument("--capacity", type=float, default=200.0)
    p.add_argument("--out-dir", default="loto_results_gwo_elm")

    # 特征组选择: "all" 运行所有, 或逗号分隔的组名
    p.add_argument("--features", default="all",
                   help="Feature groups to test: 'all', or comma-separated names")

    # GWO-ELM 超参数
    p.add_argument("--n-hidden", type=int, default=80,
                   help="Number of ELM hidden nodes")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="GBF kernel width")
    p.add_argument("--n-wolves", type=int, default=20,
                   help="GWO population size")
    p.add_argument("--max-iter", type=int, default=30,
                   help="GWO iterations")
    p.add_argument("--n-trials", type=int, default=3,
                   help="Number of random trials per fold")

    # 两阶段校正
    p.add_argument("--two-stage", action="store_true",
                   help="Enable two-stage error characteristic fusion")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 确定特征组
    if args.features == "all":
        feature_names = list(FEATURE_GROUPS.keys())
    else:
        feature_names = [s.strip() for s in args.features.split(",")]
        for fn in feature_names:
            if fn not in FEATURE_GROUPS:
                raise ValueError(f"Unknown feature group: {fn}. Available: {list(FEATURE_GROUPS.keys())}")

    # 加载数据
    print("Loading dataset...")
    df = build_dataset(args.pred_glob, args.typhoon_path)
    print(f"Dataset: {len(df)} rows, {df['台风编号'].nunique()} typhoon events")

    # 为每个特征组运行 LOTO
    all_summaries = []
    method_label = "gwo_elm_two_stage" if args.two_stage else "gwo_elm"

    for feat_name in feature_names:
        feat_cols = FEATURE_GROUPS[feat_name]
        # 只保留数据中实际存在的列
        available = [c for c in feat_cols if c in df.columns]
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            print(f"  Warning: {feat_name} missing columns: {missing}")
        if len(available) == 0:
            print(f"  Skipping {feat_name}: no available columns")
            continue

        print(f"\n{'='*60}")
        print(f"Feature group: {feat_name} ({len(available)} features)")
        print(f"Features: {available}")
        print(f"{'='*60}")

        fold_df, summary_df = run_loto_feature(
            df, available, args.capacity,
            two_stage=args.two_stage,
            n_hidden=args.n_hidden,
            gamma=args.gamma,
            n_wolves=args.n_wolves,
            max_iter=args.max_iter,
            n_trials=args.n_trials,
        )

        # 保存单特征组结果
        fold_path = out_dir / f"loto_fold_{feat_name}.csv"
        summary_path = out_dir / f"loto_summary_{feat_name}.csv"
        fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        summary_df.insert(0, "feature_group", feat_name)
        summary_df.insert(1, "n_features", len(available))
        all_summaries.append(summary_df)

        print(f"\nResults for {feat_name}:")
        print(f"  Base RMSE:  {summary_df['base_RMSE_mean'].values[0]:.4f}")
        print(f"  Corr RMSE:  {summary_df['corr_RMSE_mean'].values[0]:.4f}")
        print(f"  Base MAE:   {summary_df['base_MAE_mean'].values[0]:.4f}")
        print(f"  Corr MAE:   {summary_df['corr_MAE_mean'].values[0]:.4f}")
        print(f"  RMSE Imp:   {summary_df['rmse_improve_pct_mean'].values[0]:.2f}%")
        print(f"  MAE Imp:    {summary_df['mae_improve_pct_mean'].values[0]:.2f}%")
        print(f"  Win Rate:   {summary_df['rmse_win_rate'].values[0]:.1%}")
        print(f"  Saved: {fold_path}")
        print(f"         {summary_path}")

    # 汇总对比
    if len(all_summaries) > 0:
        comparison = pd.concat(all_summaries, ignore_index=True)
        comp_path = out_dir / f"{method_label}_feature_comparison.csv"
        comparison.to_csv(comp_path, index=False, encoding="utf-8-sig")

        print(f"\n{'='*60}")
        print(f"FEATURE COMPARISON SUMMARY ({method_label})")
        print(f"{'='*60}")
        show_cols = ["feature_group", "n_features",
                     "base_RMSE_mean", "corr_RMSE_mean",
                     "base_MAE_mean", "corr_MAE_mean",
                     "rmse_improve_pct_mean", "mae_improve_pct_mean",
                     "rmse_win_rate", "mae_win_rate"]
        print(comparison[show_cols].to_string(index=False))
        print(f"\nSaved comparison: {comp_path}")

        # 按 RMSE 提升排序
        best = comparison.loc[comparison["rmse_improve_pct_mean"].idxmax()]
        print(f"\nBest feature group: {best['feature_group']}")
        print(f"  RMSE improvement: {best['rmse_improve_pct_mean']:.2f}%")
        print(f"  MAE improvement: {best['mae_improve_pct_mean']:.2f}%")
        print(f"  Win rate: {best['rmse_win_rate']:.1%}")


if __name__ == "__main__":
    main()
