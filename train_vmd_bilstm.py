#!/usr/bin/env python3
"""
VMD-BiLSTM: 变分模态分解增强的BiLSTM风电功率预测
================================================================================

核心思想:
  VMD (Variational Mode Decomposition) 将历史功率窗口分解为K个频率模态,
  作为BiLSTM的额外输入特征, 帮助模型区分趋势、周期波动和噪声。

方法:
  原始BiLSTM输入: [past_NWP, past_power] → 预测 [future_power]
  VMD-BiLSTM输入: [past_NWP, past_power, vmd_mode1, ..., vmd_modeK] → 预测 [future_power]

  其中 vmd_mode1..K 是对过去7天(lookback)功率做VMD分解得到的模态分量.

实验设计:
  ① BiLSTM (baseline, 无VMD)
  ② VMD-BiLSTM (K=3模态作为额外特征)
  ③ 对比指标: RMSE, MAE, 台风期表现

数据:
  - 功率+NWP: data/power/ghdt_merged_15min_{year}.csv (2019-2024)
  - 训练/验证/测试: 按时间顺序分割 (前70%训练, 15%验证, 15%测试)

注意事项:
  VMD现在使用一次完整序列分解, 而非逐窗口计算, 速度大幅提升.

用法:
  python train_vmd_bilstm.py --mode train               # BiLSTM基线
  python train_vmd_bilstm.py --mode train --use_vmd     # VMD-BiLSTM
  python train_vmd_bilstm.py --mode train --compare     # 同时训练并对比
  python train_vmd_bilstm.py --mode quick               # 快速验证(小数据)
================================================================================
"""

import argparse
import math
import random
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from vmdpy import VMD

# ============================================================================
# 配置
# ============================================================================

class Config:
    """训练超参数"""
    lookback: int = 672        # 历史窗口 (7天 = 672 * 15min)
    horizon: int = 96          # 预测窗口 (24h = 96 * 15min)
    batch_size: int = 128
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 30
    patience: int = 8
    seed: int = 42
    capacity: float = 200.0    # 装机容量 MW
    vmd_k: int = 3             # VMD分解模态数
    vmd_alpha: int = 2000      # VMD带宽约束
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# VMD 特征提取
# ============================================================================

def precompute_vmd(power: np.ndarray, k: int = 3, alpha: int = 2000) -> np.ndarray:
    """对完整功率序列做一次VMD分解, 返回 (K, N) 模态矩阵.

    Args:
        power: 完整功率序列, shape (N,)
        k: 分解模态数
        alpha: VMD带宽约束参数

    Returns:
        modes: shape (k, N) 模态矩阵, 已还原到原始功率尺度并clip到[0, capacity]
    """
    n = len(power)
    if n < 100 or np.std(power) < 0.5:
        return np.zeros((k, n), dtype=np.float32)

    p_mean = power.mean()
    p_std = power.std() + 1e-6
    p_norm = (power - p_mean) / p_std

    try:
        u, _, _ = VMD(p_norm.astype(np.float64), alpha=alpha, tau=0,
                       K=k, DC=0, init=1, tol=1e-7)
    except Exception:
        return np.zeros((k, n), dtype=np.float32)

    # 还原到原始尺度 (VMD可能截断末尾, 需对齐长度)
    n_out = u.shape[1]
    modes = np.zeros((k, n), dtype=np.float32)
    for i in range(k):
        clip_len = min(n_out, n)
        modes[i, :clip_len] = u[i, :clip_len] * p_std + p_mean
        if clip_len < n:
            modes[i, clip_len:] = modes[i, clip_len - 1]  # 末值填充

    return np.clip(modes, 0, None)  # (K, N)


# ============================================================================
# 台风参数特征 (Holland / YanMeng 参数模型)
# ============================================================================

TYPHOON_FEAT_COLS = [
    '场站距离_km',     # 距台风中心距离 (km)
    'hollandvg',       # Holland梯度风速 (m/s)
    'ym风速',          # YanMeng地表风速 (m/s)
    'rmax',            # 最大风速半径 (km)
    'b',               # Holland形状参数
]

def load_typhoon_data(power_df: pd.DataFrame) -> pd.DataFrame:
    """加载台风参数数据, 按时间与功率数据对齐合并.

    自动匹配 power_df 的时间范围.
    对每个时刻, 若有台风在场站1000km内, 取距离最近的那个.
    非台风期: 台风参数填0, is_typhoon=0.
    同时记录每个时刻是否有台风影响, 用于按台风期/非台风期分评估.

    Returns:
        df: 与 power_df 行数一致的DataFrame, 追加了台风特征列和 is_typhoon 列
    """
    ty_path = f'data/typhoon/typhoon19-24complete_15min_within1000km_aug8.csv'
    if not os.path.exists(ty_path):
        print(f"  WARNING: Typhoon data not found at {ty_path}")
        df_out = power_df.copy()
        for c in TYPHOON_FEAT_COLS:
            df_out[c] = 0.0
        df_out['is_typhoon'] = 0.0
        return df_out

    ty = pd.read_csv(ty_path)
    ty['target_time'] = pd.to_datetime(ty['target_time'])
    # 自动匹配功率数据的时间范围 (支持多年数据)
    t_min, t_max = power_df['time'].min(), power_df['time'].max()
    ty = ty[(ty['target_time'] >= t_min) & (ty['target_time'] <= t_max)].copy()
    if len(ty) == 0:
        print(f"  WARNING: No typhoon data in {t_min} ~ {t_max}")
        df_out = power_df.copy()
        for c in TYPHOON_FEAT_COLS:
            df_out[c] = 0.0
        df_out['is_typhoon'] = 0.0
        return df_out

    # 同一时刻多个台风: 保留距离最近的
    ty_best = ty.loc[ty.groupby('target_time')['场站距离_km'].idxmin()].copy()
    ty_best = ty_best[['target_time'] + TYPHOON_FEAT_COLS].copy()

    # 与功率数据左合并
    df_out = power_df.merge(ty_best, left_on='time', right_on='target_time', how='left')

    # 非台风期填充
    df_out['is_typhoon'] = df_out['场站距离_km'].notna().astype(np.float32)
    for c in TYPHOON_FEAT_COLS:
        df_out[c] = df_out[c].fillna(0.0).astype(np.float32)

    n_ty = int(df_out['is_typhoon'].sum())
    print(f"  Typhoon data merged: {n_ty}/{len(df_out)} rows ({n_ty/len(df_out)*100:.1f}%)")
    return df_out


# ============================================================================
# 数据集
# ============================================================================

class WindPowerDataset(Dataset):
    """风电功率序列数据集.

    每个样本:
      x_past:  (lookback, nwp_features + 1 + vmd_k)
               = (lookback, NWP特征数 + 原始功率 + VMD模态)
      x_fut:   (horizon, nwp_features)  未来NWP外源特征
      y:       (horizon,)               未来真实功率
    """
    def __init__(self, x_past: np.ndarray, x_fut_nwp: np.ndarray, y: np.ndarray):
        self.x_past = torch.from_numpy(x_past).float()
        self.x_fut = torch.from_numpy(x_fut_nwp).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x_past)

    def __getitem__(self, idx: int):
        return self.x_past[idx], self.x_fut[idx], self.y[idx]


# ============================================================================
# 模型: BiLSTM + 外源特征
# ============================================================================

class BiLSTMRegressor(nn.Module):
    """BiLSTM编码器 + 未来NWP外源特征解码器.

    Architecture:
      past_input → BiLSTM → context_vector
      future_NWP → MLP → exogenous_features
      [context + exogenous] → MLP_head → predictions
    """
    def __init__(self, input_size: int, exog_size: int,
                 hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.ctx_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.exog_proj = nn.Sequential(
            nn.Linear(exog_size, hidden_size), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, 1),
        )

    def forward(self, x_past: torch.Tensor,
                x_fut_nwp: torch.Tensor) -> torch.Tensor:
        # x_past: (B, L, C_in)  L=lookback
        out, _ = self.lstm(x_past)
        ctx = self.ctx_proj(out[:, -1, :])  # (B, H)

        # x_fut_nwp: (B, H_fut, C_exog)  H_fut=horizon
        fut_feat = self.exog_proj(x_fut_nwp)  # (B, H_fut, H)

        ctx_expand = ctx.unsqueeze(1).expand(-1, fut_feat.size(1), -1)
        z = torch.cat([ctx_expand, fut_feat], dim=-1)  # (B, H_fut, 2H)
        pred = self.head(z).squeeze(-1)  # (B, H_fut)
        return pred


# ============================================================================
# 数据准备
# ============================================================================

def build_sequences(df: pd.DataFrame, cfg: Config, use_vmd: bool = True,
                    use_typhoon: bool = False):
    """从DataFrame构建训练序列.

    Args:
        df: 含 time, VALUE, wind_speed_10m, wind_direction_10m 的DataFrame
        cfg: 配置
        use_vmd: 是否提取VMD模态特征
        use_typhoon: 是否包含台风参数特征

    Returns:
        (x_past, x_fut_nwp, y, t_start, ty_mask)
        ty_mask: shape (n_samples,) bool, 标记每个样本的预测窗口是否受台风影响
    """
    df = df.sort_values('time').reset_index(drop=True)
    # 功率
    power = df['VALUE'].values.astype(np.float32).clip(0, cfg.capacity)
    power_norm = power / cfg.capacity

    # NWP特征
    nwp_cols = ['wind_speed_10m (m/s)', 'wind_direction_10m (°)']
    nwp = df[nwp_cols].values.astype(np.float32)
    # 简单归一化
    nwp_mean = nwp.mean(axis=0, keepdims=True)
    nwp_std = nwp.std(axis=0, keepdims=True) + 1e-6
    nwp_norm = (nwp - nwp_mean) / nwp_std

    # 台风参数特征 (如果有)
    typhoon_norm = None
    typhoon_raw = None
    ty_mask_full = None
    if use_typhoon and all(c in df.columns for c in TYPHOON_FEAT_COLS):
        typhoon_raw = df[TYPHOON_FEAT_COLS].values.astype(np.float32)
        # 每行z-score归一化
        ty_mean = typhoon_raw.mean(axis=0, keepdims=True)
        ty_std = typhoon_raw.std(axis=0, keepdims=True) + 1e-6
        typhoon_norm = (typhoon_raw - ty_mean) / ty_std
    # 只要数据中有 is_typhoon, 就生成台风掩码 (对所有模型统一, 便于分时段评估)
    if 'is_typhoon' in df.columns:
        ty_mask_full = df['is_typhoon'].values.astype(bool)

    x_past_list, x_fut_list, y_list, t_list, ty_mask_list = [], [], [], [], []

    # 预计算完整功率序列的VMD分解 (仅一次)
    vmd_modes_full = None
    if use_vmd:
        print("  Precomputing VMD on full power series...")
        t0 = time.time()
        vmd_modes_full = precompute_vmd(power, k=cfg.vmd_k, alpha=cfg.vmd_alpha)
        t_vmd = time.time() - t0
        print(f"  VMD done: shape={vmd_modes_full.shape} ({t_vmd:.1f}s)")

    n = len(df)
    for i in range(cfg.lookback, n - cfg.horizon):
        # 历史窗口
        past_nwp = nwp_norm[i - cfg.lookback: i]

        # 基础输入: NWP + 功率 (归一化)
        past_power_norm = power_norm[i - cfg.lookback: i]
        past_base = np.column_stack([past_nwp, past_power_norm])

        if use_vmd and vmd_modes_full is not None:
            vmd_window = vmd_modes_full[:, i - cfg.lookback: i].T / cfg.capacity
            x_past = np.column_stack([past_base, vmd_window])
        else:
            x_past = past_base

        if typhoon_norm is not None:
            # 历史窗口的台风参数特征
            ty_window = typhoon_norm[i - cfg.lookback: i]
            x_past = np.column_stack([x_past, ty_window])

        # 统一判定此样本的预测窗口内是否有台风 (所有模型都有掩码)
        if ty_mask_full is not None:
            ty_mask_list.append(ty_mask_full[i: i + cfg.horizon].any())
        else:
            ty_mask_list.append(False)

        # 未来NWP
        x_fut = nwp_norm[i: i + cfg.horizon]

        # 未来功率 (目标)
        y_fut = power_norm[i: i + cfg.horizon]

        x_past_list.append(x_past)
        x_fut_list.append(x_fut)
        y_list.append(y_fut)
        t_list.append(df['time'].iloc[i])

    return (
        np.array(x_past_list, dtype=np.float32),
        np.array(x_fut_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(t_list),
        np.array(ty_mask_list, dtype=bool),
    )


# ============================================================================
# 训练和评估
# ============================================================================

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, xf, yb in loader:
            pred = model(xb.to(device), xf.to(device))
            preds.append(pred.cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def compute_metrics(y_true, y_pred, capacity):
    yt = y_true.reshape(-1) * capacity
    yp = y_pred.reshape(-1) * capacity
    mse = float(np.mean((yt - yp) ** 2))
    mae = float(np.mean(np.abs(yt - yp)))
    return {'RMSE': math.sqrt(mse), 'MAE': mae,
            'NRMSE': math.sqrt(mse) / capacity,
            'NMAE': mae / capacity}


def train_model(model, train_loader, val_loader, cfg, model_path):
    device = torch.device(cfg.device)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float('inf')
    best_epoch = 0
    no_improve = 0

    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, xf, yb in train_loader:
            xb, xf, yb = xb.to(device), xf.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, xf), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_pred, val_true = evaluate(model, val_loader, device)
        val_mse = float(np.mean((val_true - val_pred) ** 2))

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}: train_mse={train_loss:.6f}, val_mse={val_mse:.6f} ({elapsed:.0f}s)")

        if val_mse < best_val:
            best_val = val_mse; best_epoch = epoch; no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  Early stop at epoch {epoch}, best={best_epoch}")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model


# ============================================================================
# 主流程
# ============================================================================

def train_and_eval(cfg, full_df, use_vmd, use_typhoon, model_path, device):
    """单次训练流程: 构建序列 → 建模型 → 训练 → 评估 → 返回结果"""
    label_parts = []
    if use_vmd:
        label_parts.append('VMD')
    if use_typhoon:
        label_parts.append('Typhoon')
    label = '-'.join(label_parts) + '-BiLSTM' if label_parts else 'BiLSTM'
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    x_past, x_fut, y, t_start, ty_mask = build_sequences(
        full_df, cfg, use_vmd=use_vmd, use_typhoon=use_typhoon)
    n_samples = len(y)
    print(f"  Samples: {n_samples}, past_shape={x_past.shape}, fut_shape={x_fut.shape}")
    if ty_mask is not None:
        print(f"  Typhoon samples: {ty_mask.sum()} ({ty_mask.mean()*100:.1f}%)")

    train_n = int(n_samples * 0.7)
    val_n = int(n_samples * 0.85)
    print(f"  Split: train={train_n}, val={val_n-train_n}, test={n_samples-val_n}")

    train_ds = WindPowerDataset(x_past[:train_n], x_fut[:train_n], y[:train_n])
    val_ds = WindPowerDataset(x_past[train_n:val_n], x_fut[train_n:val_n], y[train_n:val_n])
    test_ds = WindPowerDataset(x_past[val_n:], x_fut[val_n:], y[val_n:])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    input_size = x_past.shape[2]
    exog_size = x_fut.shape[2]
    model = BiLSTMRegressor(input_size, exog_size, cfg.hidden_size,
                             cfg.num_layers, cfg.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: input={input_size}, exog={exog_size}, params={n_params:,}")

    model = train_model(model, train_loader, val_loader, cfg, model_path)
    test_pred, test_true = evaluate(model, test_loader, device)

    # 整体指标
    metrics_all = compute_metrics(test_true, test_pred, cfg.capacity)
    print(f"\n  >> {label} Test Results (ALL):")
    for k, v in metrics_all.items():
        print(f"    {k}: {v:.3f}")

    # 台风期 / 非台风期分指标
    test_ty_mask = ty_mask[val_n:] if ty_mask is not None else None
    metrics_ty, metrics_normal = None, None
    if test_ty_mask is not None and test_ty_mask.any():
        metrics_ty = compute_metrics(test_true[test_ty_mask], test_pred[test_ty_mask], cfg.capacity)
        metrics_normal = compute_metrics(test_true[~test_ty_mask], test_pred[~test_ty_mask], cfg.capacity)
        print(f"\n  >> {label} Typhoon Period:")
        for k, v in metrics_ty.items():
            print(f"    {k}: {v:.3f}")
        print(f"\n  >> {label} Non-Typhoon Period:")
        for k, v in metrics_normal.items():
            print(f"    {k}: {v:.3f}")

    return metrics_all, metrics_ty, metrics_normal, test_pred, test_true


def print_comparison_row(name, base_val, new_val):
    """打印一行对比: 含具体数值和百分比变化"""
    change = (new_val - base_val) / base_val * 100
    arrow = "↓" if new_val < base_val else "↑" if new_val > base_val else "="
    return f"  {name:<10} {base_val:<12.3f} {new_val:<12.3f} {arrow} {abs(change):.2f}%"


def print_comparison_block(title, metrics_base, metrics_new, name_base, name_new):
    """打印一组指标的对比"""
    print(f"\n  {title}")
    print(f"  {'Metric':<10} {name_base:<12} {name_new:<12} {'Change':<12}")
    print(f"  {'-'*46}")
    for k in metrics_base:
        line = print_comparison_row(k, metrics_base[k], metrics_new[k])
        print(line)
    print(f"  {'='*46}")


def load_power_data(years, quick_mode=False):
    """加载功率+NWP数据"""
    dfs = []
    for yr in years:
        fp = f'data/power/ghdt_merged_15min_{yr}.csv'
        if not os.path.exists(fp):
            print(f"  WARNING: {fp} not found, skipping")
            continue
        df = pd.read_csv(fp)
        df['time'] = pd.to_datetime(df['time'])
        df = df[['time', 'VALUE', 'wind_speed_10m (m/s)', 'wind_direction_10m (°)']]
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        dfs.append(df)
        print(f"  Loaded {yr}: {len(df)} rows")
    if not dfs:
        return None
    full_df = pd.concat(dfs, ignore_index=True).sort_values('time').reset_index(drop=True)
    print(f"  Total: {len(full_df)} rows")
    return full_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='quick', choices=['quick', 'train'],
                       help='quick=小数据快速验证, train=全量训练')
    parser.add_argument('--use_vmd', action='store_true', default=None,
                       help='是否启用VMD特征')
    parser.add_argument('--typhoon', action='store_true', default=None,
                       help='是否启用台风参数特征')
    parser.add_argument('--compare', action='store_true', default=False,
                       help='同时训练并对比 (VMD vs VMD+Typhoon)')
    parser.add_argument('--typhoon_compare', action='store_true', default=False,
                       help='完整三模型对比: BiLSTM vs VMD-BiLSTM vs Param-BiLSTM')
    parser.add_argument('--year', type=str, default='2020',
                       help='训练数据年份 (e.g. 2020, 2021, 或 "all" 使用2019-2024)')
    args = parser.parse_args()

    if args.use_vmd is None:
        args.use_vmd = False
    if args.typhoon is None:
        args.typhoon = False

    cfg = Config()
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    # 解析年份
    AVAILABLE_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
    if args.mode == 'quick':
        years = [2024]
        cfg.epochs = 5
        cfg.patience = 3
    elif args.year.lower() == 'all':
        years = AVAILABLE_YEARS
    else:
        years = [int(args.year)]

    print(f"Config: mode={args.mode}, years={years}, epochs={cfg.epochs}")
    print(f"Device: {device}")

    # ---- 加载基础功率数据 ----
    full_df = load_power_data(years)
    if full_df is None:
        print("ERROR: No data loaded!")
        return

    # ---- 按需合并台风参数数据 (自动匹配时间范围) ----
    need_typhoon = args.typhoon or args.compare or args.typhoon_compare
    if need_typhoon:
        print("\nMerging typhoon parametric features...")
        full_df = load_typhoon_data(full_df)

    # ==================================================================
    # 模式 1: typhoon_compare — 三模型全面对比
    # ==================================================================
    if args.typhoon_compare:
        print("\n" + "="*60)
        print("  >>> TYPHOON COMPARISON: 3-model benchmark <<<")
        print("="*60)

        # 模型1: BiLSTM baseline (无VMD, 无台风)
        m1_all, m1_ty, m1_norm, p1, t1 = train_and_eval(
            cfg, full_df, use_vmd=False, use_typhoon=False,
            model_path="bilstm_best.pt", device=device)

        # 模型2: VMD-BiLSTM (有VMD, 无台风)
        m2_all, m2_ty, m2_norm, p2, t2 = train_and_eval(
            cfg, full_df, use_vmd=True, use_typhoon=False,
            model_path="vmd_bilstm_best.pt", device=device)

        # 模型3: Param-BiLSTM (VMD + 台风参数)
        m3_all, m3_ty, m3_norm, p3, t3 = train_and_eval(
            cfg, full_df, use_vmd=True, use_typhoon=True,
            model_path="param_bilstm_best.pt", device=device)

        # ---- 打印全面对比 ----
        print("\n" + "="*70)
        print("  THREE-WAY COMPARISON: BiLSTM | VMD-BiLSTM | Param-BiLSTM")
        print("="*70)

        for mode_name, m_all, m_ty, m_norm in [
            ("ALL PERIODS", m1_all, None, None),
            ("TYPHOON ONLY", m1_ty, m2_ty, m3_ty),
            ("NON-TYPHOON", m1_norm, m2_norm, m3_norm),
        ]:
            if m_ty is None and mode_name != "ALL PERIODS":
                continue
            metrics_list = [m_all, m2_all, m3_all] if mode_name == "ALL PERIODS" \
                      else [m_ty, m2_ty, m3_ty] if mode_name == "TYPHOON ONLY" \
                      else [m_norm, m2_norm, m3_norm]

            print(f"\n  [{mode_name}]")
            print(f"  {'Metric':<10} {'BiLSTM':<12} {'VMD-BiLSTM':<12} {'Param-BiLSTM':<12}")
            print(f"  {'-'*50}")
            for k in metrics_list[0]:
                v1, v2, v3 = metrics_list[0][k], metrics_list[1][k], metrics_list[2][k]
                print(f"  {k:<10} {v1:<12.3f} {v2:<12.3f} {v3:<12.3f}")

        # VMD vs Param 对比 (台风期)
        if m1_ty is not None:
            print(f"\n  [VMD vs Param-BiLSTM — TYPHOON PERIOD]")
            print(f"  {'Metric':<10} {'VMD':<12} {'Param':<12} {'Change':<12}")
            print(f"  {'-'*46}")
            for k in m2_ty:
                line = print_comparison_row(k, m2_ty[k], m3_ty[k])
                print(line)
            print(f"  {'='*46}")

        # 验证非台风期无退化
        if m1_norm is not None:
            print(f"\n  [Non-typhoon degradation check]")
            print(f"  {'Metric':<10} {'VMD':<12} {'Param':<12} {'Change':<12}")
            print(f"  {'-'*46}")
            for k in m2_norm:
                line = print_comparison_row(k, m2_norm[k], m3_norm[k])
                print(line)
            print(f"  {'='*46}")

    # ==================================================================
    # 模式 2: compare — VMD vs 非VMD (原有的)
    # ==================================================================
    elif args.compare:
        print("\n>>> COMPARISON MODE: BiLSTM vs VMD-BiLSTM <<<")
        m1_all, _, _, _, _ = train_and_eval(
            cfg, full_df, use_vmd=False, use_typhoon=False,
            model_path="bilstm_best.pt", device=device)
        m2_all, m2_ty, m2_norm, _, _ = train_and_eval(
            cfg, full_df, use_vmd=True, use_typhoon=False,
            model_path="vmd_bilstm_best.pt", device=device)
        print_comparison_block("VMD-BiLSTM vs BiLSTM (overall)", m1_all, m2_all,
                                "BiLSTM", "VMD-BiLSTM")

    # ==================================================================
    # 模式 3: 单模型训练
    # ==================================================================
    else:
        parts = []
        if args.use_vmd:
            parts.append('vmd')
        if args.typhoon:
            parts.append('typhoon')
        suffix = '_' + '_'.join(parts) if parts else ''
        model_path = f"bilstm{suffix}_best.pt"
        train_and_eval(cfg, full_df, use_vmd=args.use_vmd, use_typhoon=args.typhoon,
                        model_path=model_path, device=device)

    print("\nDone!")


if __name__ == '__main__':
    main()
