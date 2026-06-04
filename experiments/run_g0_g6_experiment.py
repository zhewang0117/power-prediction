#!/usr/bin/env python3
"""
G0-G6 消融实验: GWO-ELM 校正器 LOTO 验证

G0: Base BiLSTM (无校正)
G1: 无非物理特征 (y_pred_mw, lead_norm) - 验证残差可校正性
G2: 强度+运动特征 (P_deficit, Vmax, typhoon_level, Vmove)
G3: 几何位置特征 (dist, D/Rmax, cos_rel_azimuth, sin_rel_azimuth)
G4: 局地参数风场 (V_yanmeng, Vtan, Vrad)
G5: 全部有效特征 (G2+G3+G4)
G6: 反证组 (B_param, Rmax_km, atten_w) - 弱相关特征
"""
import os, sys, glob, re, math, time, argparse, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_typhoon_gwo_elm_loto import (
    build_dataset, run_loto_feature, metrics, ELM, GWO,
    train_elm_simple, train_elm_gwo
)

OUT_DIR = 'results/g0_g6_experiment'
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON = 96
CAPACITY = 200.0
PN = 1013.3

# ============================================================
# 特征组定义
# ============================================================
FEATURE_GROUPS = {
    "G1_no_physics": {
        "features": ["y_pred_mw", "lead_norm"],
        "description": "No typhoon features - baseline correctability check",
        "question": "Can residuals be corrected at all?"
    },
    "G2_intensity_motion": {
        "features": ["y_pred_mw", "lead_norm",
                     "P_deficit_hPa", "Vmax_ms", "typhoon_level", "Vmove_ms"],
        "description": "Intensity + motion parameters from CMA best track",
        "question": "Do basic typhoon parameters help?"
    },
    "G3_geometry": {
        "features": ["y_pred_mw", "lead_norm",
                     "dist_D_km", "D_over_Rmax", "cos_rel_azimuth", "sin_rel_azimuth"],
        "description": "Relative geometric position of wind farm to typhoon",
        "question": "Does geometric position matter?"
    },
    "G4_parametric_wind": {
        "features": ["y_pred_mw", "lead_norm",
                     "V_yanmeng_ms", "Vtan_tangential", "Vrad_radial"],
        "description": "Local parametric wind field estimates",
        "question": "Does parametric model wind help?"
    },
    "G5_all_effective": {
        "features": ["y_pred_mw", "lead_norm",
                     "P_deficit_hPa", "Vmax_ms", "typhoon_level", "Vmove_ms",
                     "dist_D_km", "D_over_Rmax", "cos_rel_azimuth", "sin_rel_azimuth",
                     "V_yanmeng_ms", "Vtan_tangential", "Vrad_radial"],
        "description": "ALL effective features combined",
        "question": "Final method - does combining all help?"
    },
    "G6_weak_control": {
        "features": ["y_pred_mw", "lead_norm",
                     "B_param", "Rmax_km", "atten_w"],
        "description": "Negative control: structurally weak features only",
        "question": "Do weak features (|r|<0.1) provide any benefit?"
    },
}

# ============================================================
# 1. 加载数据并添加计算特征
# ============================================================
print("=" * 70)
print("[1] Loading and preparing data...")

PRED_GLOB = 'data/predictions/bilstm_pred_*.csv'
TY_CSV = 'data/typhoon/typhoon19-24complete_15min_improved.csv'

df = build_dataset(PRED_GLOB, TY_CSV)

# 加载原始台风数据 (用列索引获取中文列名内容)
ty_raw = pd.read_csv(TY_CSV, encoding='utf-8-sig')
ty_raw['target_time'] = pd.to_datetime(ty_raw['target_time'], errors='coerce')
# 每时刻取最近台风
ty_raw = ty_raw.sort_values(['target_time', ty_raw.columns[14]]).drop_duplicates(
    'target_time', keep='first')

print(f"  Dataset: {len(df)} rows, {df['台风编号'].nunique()} typhoon events")

# ============================================================
# 2. 添加衍生特征
# ============================================================
print("[2] Computing derived features...")

# Merge additional raw typhoon columns (using indices for Chinese-named columns)
# df already has typhoon columns merged in build_dataset
# We need: dist, rmax_i, b_i, atten_w, ym风速_i, azimuth, rel_azimuth, move_speed, move_dir, wind_speed, pressure, level

# From build_dataset, the typhoon columns are merged. Let's check what we have
ty_cols_in_df = [c for c in ty_raw.columns if c in df.columns]
print(f"  Typhoon columns in merged df: {len(ty_cols_in_df)}")

# 用列索引提取 (列名是中文, 终端编码有问题)
# [20]=场站距离_km, [21]=方位角, [22]=相对方位角, [23]=移动速度(m/s), [24]=移动方向(°)
# [16]=风速, [17]=气压, [15]=台风等级
# [33]=rmax_i, [34]=b_i, [44]=atten_w, [39]=ym风速_i, [41]=hollandvg_az
D          = df.iloc[:, 20].fillna(0).values.astype(float)
az_vals    = df.iloc[:, 21].fillna(0).values.astype(float)
rel_az_vals= df.iloc[:, 22].fillna(0).values.astype(float)
mvspd_vals = df.iloc[:, 23].fillna(0).values.astype(float)
mvdir_vals = df.iloc[:, 24].fillna(0).values.astype(float)
ws_vals    = df.iloc[:, 16].fillna(0).values.astype(float)
pres_vals  = df.iloc[:, 17].fillna(PN).values.astype(float)
level_vals = df.iloc[:, 15].fillna(0).values.astype(float)
rmax_vals  = df['rmax_i'].fillna(0).values.astype(float)
b_vals     = df['b_i'].fillna(0).values.astype(float)
atten_vals = df['atten_w'].fillna(0).values.astype(float)
ym_vals    = df.iloc[:, 39].fillna(0).values.astype(float)   # ym风速_i
holland_vals = df['hollandvg_az'].fillna(0).values.astype(float)

# Add all derived features to df
df['dist_D_km'] = D
df['D_over_Rmax'] = np.where(rmax_vals > 0.5, D / rmax_vals, 0.0)
df['cos_rel_azimuth'] = np.cos(np.radians(rel_az_vals))
df['sin_rel_azimuth'] = np.sin(np.radians(rel_az_vals))
df['cos_azimuth'] = np.cos(np.radians(az_vals))
df['sin_azimuth'] = np.sin(np.radians(az_vals))
df['Vmax_ms'] = ws_vals
df['P_deficit_hPa'] = PN - pres_vals
df['Rmax_km'] = rmax_vals
df['B_param'] = b_vals
df['atten_w'] = atten_vals
df['Vmove_ms'] = mvspd_vals
df['sin_move_dir'] = np.sin(np.radians(mvdir_vals))
df['cos_move_dir'] = np.cos(np.radians(mvdir_vals))
df['V_holland_ms'] = holland_vals
df['V_yanmeng_ms'] = ym_vals
df['Vtan_tangential'] = ym_vals * df['sin_rel_azimuth'].values
df['Vrad_radial'] = ym_vals * df['cos_rel_azimuth'].values
df['typhoon_level'] = level_vals

print("  Derived features added:", [c for c in df.columns if c in [
    'dist_D_km','D_over_Rmax','cos_rel_azimuth','sin_rel_azimuth',
    'Vmax_ms','P_deficit_hPa','Rmax_km','B_param','atten_w','Vmove_ms',
    'V_yanmeng_ms','Vtan_tangential','Vrad_radial','typhoon_level'
]])

# Verify features exist
for group_name, group_info in FEATURE_GROUPS.items():
    missing = [f for f in group_info['features'] if f not in df.columns]
    if missing:
        print(f"  WARNING: {group_name} missing features: {missing}")

# Fix: make sure V_yanmeng_ms exists even if ym was zero
if 'V_yanmeng_ms' not in df.columns:
    df['V_yanmeng_ms'] = 0.0

# ============================================================
# 3. G0: Base BiLSTM (no correction)
# ============================================================
print("\n" + "=" * 70)
print("[G0] Base BiLSTM baseline...")

# Compute base metrics per typhoon event
g0_rows = []
for event_id, grp in df.groupby('台风编号'):
    y_true = grp['y_true_mw'].values.astype(float)
    y_base = np.clip(grp['y_pred_mw'].values.astype(float), 0, CAPACITY)
    m = metrics(y_true, y_base, CAPACITY)
    m['test_event'] = event_id
    event_name = str(df.iloc[:, 8].dropna().iloc[0]) if df.iloc[:, 8].notna().any() else str(event_id)
    m['test_event_name'] = event_name
    m['rmse_improve_pct'] = 0.0  # no correction
    g0_rows.append(m)

df_g0 = pd.DataFrame(g0_rows)
g0_summary = {
    'group': 'G0_Base_BiLSTM',
    'base_RMSE_mean': df_g0['RMSE'].mean(),
    'base_RMSE_std': df_g0['RMSE'].std(),
    'base_MAE_mean': df_g0['MAE'].mean(),
    'base_MAE_std': df_g0['MAE'].std(),
    'corr_RMSE_mean': df_g0['RMSE'].mean(),  # same as base
    'corr_MAE_mean': df_g0['MAE'].mean(),
    'rmse_improve_pct_mean': 0.0,
    'rmse_win_rate': 0.0,
}
print(f"  Base RMSE: {g0_summary['base_RMSE_mean']:.2f} ± {g0_summary['base_RMSE_std']:.2f} MW")

# ============================================================
# 4. 运行 G1-G6
# ============================================================
print("\n" + "=" * 70)
print("[G1-G6] Running LOTO experiments...")

all_summaries = [g0_summary]
all_fold_dfs = [df_g0]

for group_name in ['G1_no_physics', 'G2_intensity_motion', 'G3_geometry',
                    'G4_parametric_wind', 'G5_all_effective', 'G6_weak_control']:
    group_info = FEATURE_GROUPS[group_name]
    feat_cols = group_info['features']

    # Check all features exist
    missing = [f for f in feat_cols if f not in df.columns]
    if missing:
        print(f"\n  SKIP {group_name}: missing features {missing}")
        continue

    print(f"\n{'='*70}")
    print(f"  [{group_name}] {group_info['description']}")
    print(f"  Question: {group_info['question']}")
    print(f"  Features ({len(feat_cols)}): {feat_cols}")
    print(f"{'='*70}")

    t_start = time.time()

    try:
        fold_df, summary_df = run_loto_feature(
            df, feat_cols, CAPACITY,
            two_stage=False,
            simple_elm=False,  # use GWO-ELM
            n_hidden=60,
            gamma=1.0,
            n_wolves=15,
            max_iter=20,
            n_trials=2,
            max_train_rows=15000,
            ridge=1.0,        # stronger regularization to prevent singular H
            model_dir=f'{OUT_DIR}/models_{group_name}' if True else '',
        )

        # Save fold results
        fold_df.to_csv(f'{OUT_DIR}/loto_fold_{group_name}.csv', index=False, encoding='utf-8-sig')

        summary = {
            'group': group_name,
            'description': group_info['description'],
            'question': group_info['question'],
            'n_features': len(feat_cols),
            'features': feat_cols,
            'base_RMSE_mean': summary_df['base_RMSE_mean'].values[0],
            'base_RMSE_std': summary_df['base_RMSE_std'].values[0],
            'corr_RMSE_mean': summary_df['corr_RMSE_mean'].values[0],
            'corr_RMSE_std': summary_df['corr_RMSE_std'].values[0],
            'base_MAE_mean': summary_df['base_MAE_mean'].values[0],
            'corr_MAE_mean': summary_df['corr_MAE_mean'].values[0],
            'rmse_improve_pct_mean': summary_df['rmse_improve_pct_mean'].values[0],
            'mae_improve_pct_mean': summary_df['mae_improve_pct_mean'].values[0],
            'rmse_win_rate': summary_df['rmse_win_rate'].values[0],
            'elapsed_sec': time.time() - t_start,
        }
        all_summaries.append(summary)
        all_fold_dfs.append(fold_df)

        imp = summary['rmse_improve_pct_mean']
        arrow = "↓" if imp > 0 else "↑"
        print(f"  >>> {group_name}: RMSE={summary['base_RMSE_mean']:.2f}->{summary['corr_RMSE_mean']:.2f}, "
              f"{arrow}{abs(imp):.2f}%, win_rate={summary['rmse_win_rate']*100:.0f}%")

    except Exception as e:
        print(f"  ERROR in {group_name}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# 5. 汇总结果
# ============================================================
print("\n" + "=" * 70)
print("  FINAL RESULTS TABLE")
print("=" * 70)

summary_df_all = pd.DataFrame(all_summaries)

cols_show = ['group', 'n_features', 'base_RMSE_mean', 'corr_RMSE_mean',
             'rmse_improve_pct_mean', 'rmse_win_rate']
print(f"\n{'Group':<25s} {'Feats':>5s} {'BaseRMSE':>10s} {'CorrRMSE':>10s} {'Imp%':>8s} {'WinRate':>8s}")
print("-" * 75)
for _, row in summary_df_all.iterrows():
    imp = row['rmse_improve_pct_mean']
    wr = row['rmse_win_rate'] if 'rmse_win_rate' in row else 0
    arrow = "↓" if imp > 0 else "↑"
    print(f"{row['group']:<25s} {row.get('n_features',0):5d} {row['base_RMSE_mean']:10.2f} "
          f"{row['corr_RMSE_mean']:10.2f} {arrow}{abs(imp):7.2f}% {wr*100:7.0f}%")

# Save
summary_df_all.to_csv(f'{OUT_DIR}/g0_g6_summary.csv', index=False, encoding='utf-8-sig')

# Also save as JSON for easier reading
results_json = []
for _, row in summary_df_all.iterrows():
    results_json.append({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                         for k, v in row.items() if pd.notna(v)})
with open(f'{OUT_DIR}/g0_g6_summary.json', 'w', encoding='utf-8') as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False, default=str)

print(f"\nResults saved to {OUT_DIR}/")
print("Done!")
