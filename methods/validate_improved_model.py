#!/usr/bin/env python3
"""
=============================================================================
验证改进参数模型对 NWP 风速的提升效果
=============================================================================

对比: 原始参数风 vs 改进参数风 vs ERA5/NWP风
参考: 实测功率 (VALUE) 作为独立的物理验证

指标:
  1. 参数风与 ERA5 风的相关系数 (越高越好)
  2. 参数风与 ERA5 风的 RMSE (越低越好)
  3. 参数风与实测功率的相关性 (越高越好, 物理验证)
  4. 分距离/分强度分析

用法:
  python methods/validate_improved_model.py
=============================================================================
"""

import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')
# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# 1. 加载数据
# ============================================================================

def load_data():
    """加载台风参数数据和 GHDT 观测数据."""
    print("加载台风参数数据...")
    ty = pd.read_csv(
        os.path.join(BASE, 'data/typhoon/typhoon19-24complete_15min_improved.csv'),
        encoding='utf-8-sig')
    ty['target_time'] = pd.to_datetime(ty['target_time'])

    print("加载 GHDT 观测数据 (ERA5 wind + Power)...")
    ghdt_parts = []
    for yr in range(2019, 2025):
        fp = os.path.join(BASE, f'data/power/ghdt_merged_15min_{yr}.csv')
        if os.path.exists(fp):
            g = pd.read_csv(fp)
            g['time'] = pd.to_datetime(g['time'])
            ghdt_parts.append(g)
    ghdt = pd.concat(ghdt_parts, ignore_index=True)
    ghdt = ghdt.sort_values('time').drop_duplicates('time')

    print(f"  台风数据: {len(ty)} 行, {ty['台风编号'].nunique()} 事件")
    print(f"  GHDT数据: {len(ghdt)} 行")

    return ty, ghdt


def merge_typhoon_ghdt(ty, ghdt):
    """按时间合并台风参数与 GHDT 观测."""
    m = ty.merge(ghdt, left_on='target_time', right_on='time', how='inner')
    m = m.sort_values(['台风编号', 'target_time']).reset_index(drop=True)

    # 提取功率
    m['power'] = pd.to_numeric(m['VALUE'], errors='coerce')
    m['nwp_wind'] = pd.to_numeric(m['wind_speed_10m (m/s)'], errors='coerce')

    print(f"  合并后: {len(m)} 行 ({m['台风编号'].nunique()} 个台风)")
    return m


# ============================================================================
# 2. 核心对比指标
# ============================================================================

def compute_metrics(true_vals, pred_vals, label=''):
    """计算相关性、RMSE、MAE."""
    mask = (true_vals.notna() & pred_vals.notna())
    t = true_vals[mask].values
    p = pred_vals[mask].values
    if len(t) < 10:
        return {}
    r = float(np.corrcoef(t, p)[0, 1])
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae = float(np.mean(np.abs(t - p)))
    bias = float(np.mean(p - t))
    return {'r': r, 'RMSE': rmse, 'MAE': mae, 'Bias': bias, 'N': len(t)}


def evaluate_all(df, nwp_col='nwp_wind', power_col='power'):
    """全量 + 分箱评估."""
    results = {}

    param_cols = {
        'YM_original': 'ym风速',
        'YM_improved': 'ym风速_i',
        'Holland_orig': 'hollandvg',
        'Holland_impr': 'hollandvg_i',
        'Holland_az': 'hollandvg_az',
    }

    # ---- A. vs ERA5 wind ----
    print("\n" + "=" * 75)
    print("A. 参数风速 vs ERA5/NWP 风速 (wind_speed_10m)")
    print("=" * 75)
    print(f"{'模型':20s} {'r':>7s} {'RMSE':>8s} {'MAE':>7s} {'Bias':>7s} {'N':>6s}")
    print("-" * 60)

    for label, col in param_cols.items():
        if col in df.columns:
            m = compute_metrics(df[nwp_col], df[col], label)
            if m:
                results[f'{label}_vs_NWP'] = m
                print(f"{label:20s} {m['r']:+7.4f} {m['RMSE']:8.3f} "
                      f"{m['MAE']:7.3f} {m['Bias']:+7.3f} {m['N']:6d}")

    # ---- B. vs Power (物理验证) ----
    print("\n" + "=" * 75)
    print("B. 参数风速 vs 实测功率 (物理验证, 仅功率>0)")
    print("=" * 75)

    df_pwr = df[df[power_col] > 0].copy()
    print(f"  功率>0 样本数: {len(df_pwr)}")

    print(f"{'模型':20s} {'r':>7s} {'RMSE(MW)':>10s} {'N':>6s}")
    print("-" * 50)

    for label, col in param_cols.items():
        if col in df.columns:
            m = compute_metrics(df_pwr[power_col], df_pwr[col], label)
            if m:
                results[f'{label}_vs_Power'] = m
                print(f"{label:20s} {m['r']:+7.4f} {m['RMSE']:10.1f} {m['N']:6d}")

    # NWP wind vs Power (benchmark)
    m_nwp = compute_metrics(df_pwr[power_col], df_pwr[nwp_col], 'NWP')
    if m_nwp:
        results['NWP_vs_Power'] = m_nwp
        print(f"{'NWP (ERA5)':20s} {m_nwp['r']:+7.4f} {m_nwp['RMSE']:10.1f} {m_nwp['N']:6d}")

    # ---- C. 分距离段 ----
    print("\n" + "=" * 75)
    print("C. 按距离分箱 — 参数风 vs ERA5 风 (相关系数 r)")
    print("=" * 75)

    dist_bins = [(0, 100, '<100km'), (100, 200, '100-200km'),
                 (200, 500, '200-500km'), (500, 1000, '500-1000km')]

    print(f"{'距离段':15s} {'YM原':>7s} {'YM改':>7s} {'Holl原':>7s} "
          f"{'Holl改':>7s} {'Holl_az':>7s} {'N':>6s}")
    print("-" * 60)

    for lo, hi, label in dist_bins:
        sub = df[(df['场站距离_km'] >= lo) & (df['场站距离_km'] < hi)]
        if len(sub) < 20:
            continue
        row = []
        for col_key in ['ym风速', 'ym风速_i', 'hollandvg', 'hollandvg_i', 'hollandvg_az']:
            m = compute_metrics(sub[nwp_col], sub[col_key])
            row.append(f"{m.get('r', 0):+7.4f}" if m else "    N/A")
        print(f"{label:15s} {row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {len(sub):6d}")

    # ---- D. 分台风强度 ----
    print("\n" + "=" * 75)
    print("D. 按台风强度分箱 — 参数风 vs ERA5 风 (相关系数 r)")
    print("=" * 75)

    # 使用风速列
    if '风速' in df.columns:
        df['intensity'] = pd.cut(df['风速'],
            bins=[0, 17.2, 24.5, 32.7, 100],
            labels=['TD(<17)', 'TS(17-24)', 'STS(24-33)', 'TY(>33)'])

        print(f"{'强度':15s} {'YM原':>7s} {'YM改':>7s} {'Holl原':>7s} "
              f"{'Holl改':>7s} {'Holl_az':>7s} {'N':>6s}")
        print("-" * 60)

        for grp, sub in df.groupby('intensity', observed=True):
            if len(sub) < 20:
                continue
            row = []
            for col_key in ['ym风速', 'ym风速_i', 'hollandvg', 'hollandvg_i', 'hollandvg_az']:
                m = compute_metrics(sub[nwp_col], sub[col_key])
                row.append(f"{m.get('r', 0):+7.4f}" if m else "    N/A")
            print(f"{str(grp):15s} {row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {len(sub):6d}")

    return results


# ============================================================================
# 3. 台风级汇总 — 每个台风单独对比
# ============================================================================

def per_typhoon_summary(df, nwp_col='nwp_wind'):
    """逐台风统计改进幅度."""
    print("\n" + "=" * 75)
    print("E. 逐台风改进幅度 (YM表面风 vs ERA5)")
    print("=" * 75)

    rows = []
    for tid, sub in df.groupby('台风编号'):
        name = sub['台风中文名称'].iloc[0]
        m_orig = compute_metrics(sub[nwp_col], sub['ym风速'])
        m_impr = compute_metrics(sub[nwp_col], sub['ym风速_i'])

        if m_orig and m_impr:
            r_diff = m_impr['r'] - m_orig['r']
            rmse_diff = m_impr['RMSE'] - m_orig['RMSE']
            d_min = sub['场站距离_km'].min()
            rows.append({
                '台风': name, '最近距离km': d_min,
                'r_orig': m_orig['r'], 'r_impr': m_impr['r'], 'r_diff': r_diff,
                'RMSE_orig': m_orig['RMSE'], 'RMSE_impr': m_impr['RMSE'],
                'RMSE_diff': rmse_diff, 'N': m_orig['N'],
            })

    per_ty = pd.DataFrame(rows).sort_values('最近距离km')
    if len(per_ty) == 0:
        print("  无有效台风")
        return None

    print(f"\n{'台风':8s} {'最近距':>6s} {'r原':>7s} {'r改':>7s} {'Δr':>7s} "
          f"{'RMSE原':>8s} {'RMSE改':>8s} {'ΔRMSE':>8s} {'N':>5s}")
    print("-" * 75)
    for _, r in per_ty.iterrows():
        print(f"{r['台风']:8s} {r['最近距离km']:6.0f}km "
              f"{r['r_orig']:+7.4f} {r['r_impr']:+7.4f} {r['r_diff']:+7.4f} "
              f"{r['RMSE_orig']:8.3f} {r['RMSE_impr']:8.3f} {r['RMSE_diff']:+8.3f} "
              f"{r['N']:5d}")

    n_better = (per_ty['r_diff'] > 0).sum()
    n_total = len(per_ty)
    r_avg_orig = per_ty['r_orig'].mean()
    r_avg_impr = per_ty['r_impr'].mean()

    print(f"\n  相关性提升的台风: {n_better}/{n_total}")
    print(f"  平均 r: {r_avg_orig:+.4f} → {r_avg_impr:+.4f} "
          f"({(r_avg_impr - r_avg_orig):+.4f})")

    return per_ty


# ============================================================================
# 4. 增量信息分析 — 改进模型能否提供 NWP 之外的信息?
# ============================================================================

def incremental_value_analysis(df, nwp_col='nwp_wind'):
    """分析改进参数风是否能补充 NWP 未捕获的信息.

    方法:
      残差 = 实测功率 - NWP风×α (线性回归)
      如果参数风与残差显著相关, 说明参数风提供了 NWP 之外的信息.
    """
    print("\n" + "=" * 75)
    print("F. 增量信息分析 — 参数风能否补充 NWP 未解释的变异?")
    print("=" * 75)

    valid = df[[nwp_col, 'power']].dropna().copy()
    if 'ym风速' not in df.columns or 'ym风速_i' not in df.columns:
        print("  缺少参数风列")
        return

    valid['ym_orig'] = df.loc[valid.index, 'ym风速']
    valid['ym_impr'] = df.loc[valid.index, 'ym风速_i']
    valid = valid.dropna()

    if len(valid) < 100:
        print("  有效样本不足")
        return

    # Step 1: NWP 对功率的线性回归
    X_nwp = np.column_stack([np.ones(len(valid)), valid[nwp_col].values])
    y_pwr = valid['power'].values
    beta_nwp = np.linalg.lstsq(X_nwp, y_pwr, rcond=None)[0]
    pwr_pred_nwp = X_nwp @ beta_nwp
    resid_nwp = y_pwr - pwr_pred_nwp  # NWP 未能解释的残差

    r_nwp_vs_pwr = float(np.corrcoef(valid[nwp_col], y_pwr)[0, 1])
    print(f"  NWP vs Power: r={r_nwp_vs_pwr:.4f}")

    # Step 2: 参数风与 NWP残差的相关性
    r_orig_resid = float(np.corrcoef(valid['ym_orig'], resid_nwp)[0, 1])
    r_impr_resid = float(np.corrcoef(valid['ym_impr'], resid_nwp)[0, 1])

    print(f"  YM原 vs NWP残差: r={r_orig_resid:+7.4f} "
          f"{'(显著补充!)' if abs(r_orig_resid)>0.1 else '(弱)'}")
    print(f"  YM改 vs NWP残差: r={r_impr_resid:+7.4f} "
          f"{'(显著补充!)' if abs(r_impr_resid)>0.1 else '(弱)'}")

    # Step 3: 加入参数风后的回归 R² 提升
    X_comb_orig = np.column_stack([X_nwp, valid['ym_orig'].values])
    X_comb_impr = np.column_stack([X_nwp, valid['ym_impr'].values])

    beta_comb_orig = np.linalg.lstsq(X_comb_orig, y_pwr, rcond=None)[0]
    beta_comb_impr = np.linalg.lstsq(X_comb_impr, y_pwr, rcond=None)[0]

    pwr_pred_comb_orig = X_comb_orig @ beta_comb_orig
    pwr_pred_comb_impr = X_comb_impr @ beta_comb_impr

    # R²
    ss_tot = np.sum((y_pwr - y_pwr.mean())**2)
    r2_nwp = 1 - np.sum((y_pwr - pwr_pred_nwp)**2) / ss_tot
    r2_orig = 1 - np.sum((y_pwr - pwr_pred_comb_orig)**2) / ss_tot
    r2_impr = 1 - np.sum((y_pwr - pwr_pred_comb_impr)**2) / ss_tot

    print(f"\n  R2 (power ~ NWP only):       {r2_nwp:.4f}")
    print(f"  R2 (power ~ NWP + YM orig):  {r2_orig:.4f} (dR2={r2_orig-r2_nwp:+.4f})")
    print(f"  R2 (power ~ NWP + YM impr):  {r2_impr:.4f} (dR2={r2_impr-r2_nwp:+.4f})")

    if r2_impr > r2_nwp:
        print(f"\n  >>> Improved YM provides incremental info beyond NWP: "
              f"R2 gain {r2_impr-r2_nwp:+.4f}")

    # Step 4: 近距离 (d<200km) 的增量信息 — 参数模型最强区域
    close = valid.loc[df['场站距离_km'] < 200]
    if len(close) > 30:
        X_nwp_c = np.column_stack([np.ones(len(close)), close[nwp_col].values])
        y_pwr_c = close['power'].values
        beta_nwp_c = np.linalg.lstsq(X_nwp_c, y_pwr_c, rcond=None)[0]
        pwr_pred_nwp_c = X_nwp_c @ beta_nwp_c
        ss_tot_c = np.sum((y_pwr_c - y_pwr_c.mean())**2)

        X_comb_c = np.column_stack([X_nwp_c, close['ym_impr'].values])
        beta_comb_c = np.linalg.lstsq(X_comb_c, y_pwr_c, rcond=None)[0]
        pwr_pred_comb_c = X_comb_c @ beta_comb_c

        r2_nwp_c = 1 - np.sum((y_pwr_c - pwr_pred_nwp_c)**2) / ss_tot_c
        r2_comb_c = 1 - np.sum((y_pwr_c - pwr_pred_comb_c)**2) / ss_tot_c

        print(f"\n  Close range (d<200km, n={len(close)}):")
        print(f"    R2 (NWP only):    {r2_nwp_c:.4f}")
        print(f"    R2 (NWP + YM impr): {r2_comb_c:.4f} (dR2={r2_comb_c-r2_nwp_c:+.4f})")


# ============================================================================
# 5. 主流程
# ============================================================================

def main():
    print("=" * 75)
    print("改进参数台风风场模型 — 精度验证")
    print("=" * 75)

    ty, ghdt = load_data()
    df = merge_typhoon_ghdt(ty, ghdt)

    if len(df) == 0:
        print("错误: 台风数据与 GHDT 数据无时间重叠!")
        return

    # 检查必要列
    required = ['ym风速', 'ym风速_i', 'hollandvg', 'hollandvg_i',
                'hollandvg_az', 'nwp_wind', 'power', '场站距离_km']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"缺少列: {missing}")

    # 运行所有评估
    results = evaluate_all(df)
    per_ty = per_typhoon_summary(df)
    incremental_value_analysis(df)

    # ---- 最终结论 ----
    print("\n" + "=" * 75)
    print("结论摘要")
    print("=" * 75)

    # 汇总增量信息
    valid = df[['nwp_wind', 'power']].dropna().copy()
    if 'ym风速' in df.columns and 'ym风速_i' in df.columns:
        valid['ym_orig'] = df.loc[valid.index, 'ym风速']
        valid['ym_impr'] = df.loc[valid.index, 'ym风速_i']
        valid = valid.dropna()

        # NWP vs 功率
        r_nwp = float(np.corrcoef(valid['nwp_wind'], valid['power'])[0, 1])

        # 残差相关性
        X = np.column_stack([np.ones(len(valid)), valid['nwp_wind'].values])
        y = valid['power'].values
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        r_ym_resid = float(np.corrcoef(valid['ym_impr'], resid)[0, 1])

        print(f"\n  NWP风 vs 功率:  r = {r_nwp:.4f}")
        print(f"  改进YM风 vs NWP残差: r = {r_ym_resid:.4f}")
        if abs(r_ym_resid) > 0.05:
            print(f"  >>> 结论: 改进参数风提供了 NWP 之外的增量信息!")
        else:
            print(f"  >>> 注意: 当前距离下参数风的增量信息有限, "
                  f"近距离(d<200km)效果可能更显著")

    print(f"\n  总台风数: {df['台风编号'].nunique()}")
    print(f"  总匹配样本: {len(df)}")
    print()

    # 保存详细结果
    out_dir = os.path.join(BASE, 'results')
    os.makedirs(out_dir, exist_ok=True)
    if per_ty is not None:
        per_ty.to_csv(os.path.join(out_dir, 'per_typhoon_validation.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"逐台风结果已保存至: results/per_typhoon_validation.csv")


if __name__ == '__main__':
    main()
