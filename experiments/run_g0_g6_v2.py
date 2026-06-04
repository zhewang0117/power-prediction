#!/usr/bin/env python3
"""
G0-G6 消融实验 V2: 纯特征组, G2-G6 不含 y_pred_mw + lead_norm

G0: Base BiLSTM (无校正)
G1: y_pred_mw + lead_norm                         — 纯预测特征基线
G2: P_deficit, Vmax, typhoon_level, Vmove         — 原始CMA参数
G3: dist, D/Rmax, cos_rel_azimuth, sin_rel_azimuth — 几何位置
G4: V_yanmeng, Vtan, Vrad                         — 参数风场
G5: G2+G3+G4全部物理特征                           — 全部
G6: B, Rmax, atten                                — 弱结构参数(反证)
"""
import os, sys, glob, re, time, math, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_typhoon_gwo_elm_loto import (
    build_dataset, run_loto_feature, metrics, ELM, GWO,
    train_elm_simple, train_elm_gwo
)

HORIZON = 96
CAPACITY = 200.0
PN = 1013.3
MODEL_DIR = 'results/g0_g6_v2'
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 特征组定义 (G2-G6 不含 y_pred_mw + lead_norm)
# ============================================================
FEATURE_GROUPS = {
    "G1_pred_only": {
        "features": ["y_pred_mw", "lead_norm"],
        "desc": "Prediction features only (baseline)",
    },
    "G2_cma_params": {
        "features": ["P_deficit_hPa", "Vmax_ms", "typhoon_level", "Vmove_ms"],
        "desc": "CMA best-track parameters only",
    },
    "G3_geometry": {
        "features": ["dist_D_km", "D_over_Rmax", "cos_rel_azimuth", "sin_rel_azimuth"],
        "desc": "Geometric position features only",
    },
    "G4_parametric_wind": {
        "features": ["V_yanmeng_ms", "Vtan_tangential", "Vrad_radial"],
        "desc": "Parametric wind field features only",
    },
    "G5_all_physics": {
        "features": ["P_deficit_hPa", "Vmax_ms", "typhoon_level", "Vmove_ms",
                     "dist_D_km", "D_over_Rmax", "cos_rel_azimuth", "sin_rel_azimuth",
                     "V_yanmeng_ms", "Vtan_tangential", "Vrad_radial"],
        "desc": "ALL physics features combined",
    },
    "G6_weak_structural": {
        "features": ["B_param", "Rmax_km", "atten_w"],
        "desc": "Weak structural params (negative control)",
    },
}

# ============================================================
# 1. 加载数据 + 衍生特征
# ============================================================
print("[1] Loading data...")
PRED_GLOB = 'data/predictions/bilstm_pred_*.csv'
TY_CSV = 'data/typhoon/typhoon19-24complete_15min_improved.csv'
df = build_dataset(PRED_GLOB, TY_CSV)

D = df.iloc[:, 20].fillna(0).values.astype(float)
rmax = df['rmax_i'].fillna(0).values.astype(float)
b = df['b_i'].fillna(0).values.astype(float)
atten = df['atten_w'].fillna(0).values.astype(float)
ym = df.iloc[:, 39].fillna(0).values.astype(float)
rel_az = df.iloc[:, 22].fillna(0).values.astype(float)
az = df.iloc[:, 21].fillna(0).values.astype(float)
ws = df.iloc[:, 16].fillna(0).values.astype(float)
pres = df.iloc[:, 17].fillna(PN).values.astype(float)
level = df.iloc[:, 15].fillna(0).values.astype(float)
mvspd = df.iloc[:, 23].fillna(0).values.astype(float)

df['dist_D_km'] = D
df['D_over_Rmax'] = np.where(rmax > 0.5, D / rmax, 0.0)
df['cos_rel_azimuth'] = np.cos(np.radians(rel_az))
df['sin_rel_azimuth'] = np.sin(np.radians(rel_az))
df['Vmax_ms'] = ws
df['P_deficit_hPa'] = PN - pres
df['Rmax_km'] = rmax
df['B_param'] = b
df['atten_w'] = atten
df['Vmove_ms'] = mvspd
df['V_yanmeng_ms'] = ym
df['Vtan_tangential'] = ym * np.sin(np.radians(rel_az))
df['Vrad_radial'] = ym * np.cos(np.radians(rel_az))
df['typhoon_level'] = level

print(f"  Dataset: {len(df)} rows, {df.iloc[:, 7].nunique()} typhoon events")

# ============================================================
# 2. G0: Base BiLSTM baseline
# ============================================================
print("\n[G0] Base BiLSTM...")
g0_rows = []
for ev, grp in df.groupby(df.columns[7]):
    y_true = grp['y_true_mw'].values.astype(float)
    y_base = np.clip(grp['y_pred_mw'].values.astype(float), 0, CAPACITY)
    m = metrics(y_true, y_base, CAPACITY)
    m['test_event'] = ev
    g0_rows.append(m)
df_g0 = pd.DataFrame(g0_rows)
base_rmse_mean = df_g0['RMSE'].mean()
base_rmse_std = df_g0['RMSE'].std()
print(f"  Base RMSE: {base_rmse_mean:.2f} ± {base_rmse_std:.2f} MW")

# ============================================================
# 3. Run G1-G6
# ============================================================
all_summaries = [{
    'group': 'G0_Base_BiLSTM', 'desc': 'No correction',
    'n_features': 0, 'features': '—',
    'base_RMSE_mean': base_rmse_mean, 'base_RMSE_std': base_rmse_std,
    'corr_RMSE_mean': base_rmse_mean, 'corr_RMSE_std': base_rmse_std,
    'rmse_improve_pct_mean': 0.0, 'rmse_win_rate': 0.0,
}]

for group_name in ['G1_pred_only', 'G2_cma_params', 'G3_geometry',
                    'G4_parametric_wind', 'G5_all_physics', 'G6_weak_structural']:
    info = FEATURE_GROUPS[group_name]
    feats = info['features']

    # Verify features exist
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"\n  SKIP {group_name}: missing {missing}")
        continue

    print(f"\n{'='*60}")
    print(f"  [{group_name}] {info['desc']}")
    print(f"  Features: {feats}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        fold_df, summary_df = run_loto_feature(
            df, feats, CAPACITY,
            two_stage=False, simple_elm=False,
            n_hidden=60, gamma=1.0,
            n_wolves=15, max_iter=20, n_trials=2,
            max_train_rows=15000, ridge=1.0,
            model_dir=f'{MODEL_DIR}/models_{group_name}',
        )
        fold_df.to_csv(f'{MODEL_DIR}/loto_fold_{group_name}.csv', index=False, encoding='utf-8-sig')

        summary = {
            'group': group_name, 'desc': info['desc'],
            'n_features': len(feats), 'features': feats,
            'base_RMSE_mean': summary_df['base_RMSE_mean'].values[0],
            'base_RMSE_std': summary_df['base_RMSE_std'].values[0],
            'corr_RMSE_mean': summary_df['corr_RMSE_mean'].values[0],
            'corr_RMSE_std': summary_df['corr_RMSE_std'].values[0],
            'rmse_improve_pct_mean': summary_df['rmse_improve_pct_mean'].values[0],
            'rmse_win_rate': summary_df['rmse_win_rate'].values[0],
            'elapsed_min': (time.time() - t0) / 60,
        }
        all_summaries.append(summary)
        imp = summary['rmse_improve_pct_mean']
        wr = summary['rmse_win_rate']
        arrow = "v" if imp > 0 else "^"
        print(f"  >>> {group_name}: RMSE={summary['base_RMSE_mean']:.2f}->{summary['corr_RMSE_mean']:.2f}, "
              f"{arrow}{abs(imp):.2f}%, win={wr*100:.0f}%")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

# ============================================================
# 4. Final Summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL RESULTS: G0-G6 V2 (G2-G6 = pure typhoon features, NO y_pred_mw)")
print("=" * 70)

df_summary = pd.DataFrame(all_summaries)
print(f"\n{'Group':<22s} {'Feats':>5s} {'BaseRMSE':>9s} {'CorrRMSE':>9s} {'Imp%':>8s} {'Win':>6s}")
print("-" * 65)
for _, row in df_summary.iterrows():
    imp = row['rmse_improve_pct_mean']
    wr = row.get('rmse_win_rate', 0)
    nf = row.get('n_features', 0)
    arrow = "v" if imp > 0 else "^"
    wr_str = f"{wr*100:.0f}%" if wr else "—"
    print(f"{row['group']:<22s} {nf:5d} {row['base_RMSE_mean']:9.2f} {row['corr_RMSE_mean']:9.2f} "
          f"{arrow}{abs(imp):7.2f}% {wr_str:>6s}")

df_summary.to_csv(f'{MODEL_DIR}/g0_g6_v2_summary.csv', index=False, encoding='utf-8-sig')
with open(f'{MODEL_DIR}/g0_g6_v2_summary.json', 'w', encoding='utf-8') as f:
    json.dump([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in row.items() if pd.notna(v)}
               for _, row in df_summary.iterrows()], f, indent=2, ensure_ascii=False, default=str)

print(f"\nResults saved to {MODEL_DIR}/")
print("Done!")
