#!/usr/bin/env python3
"""
V_yanmeng 条件校正: 只在有利条件下启用参数风场校正

Gate策略:
  B1: dist ∈ [100, 350] → G4, else → G1
  B2: dist ∈ [150, 300] → G4, else → G1
  B3: 平滑融合 weight=sigmoid(dist,150,300) * (1-sigmoid(dist,300,450))
  C1: |V_yanmeng| > 10 m/s → G4, else → G1
  D1: dist∈[100,350] AND |V_yanmeng|>5 → G4
"""
import os, sys, glob, time, math, json, inspect
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_typhoon_gwo_elm_loto import (
    build_dataset, metrics, train_elm_gwo
)

HORIZON = 96; CAPACITY = 200.0; PN = 1013.3
MODEL_DIR = 'results/yanmeng_conditional'
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print("[1] Loading data...")
PRED_GLOB = 'data/predictions/bilstm_pred_*.csv'
TY_CSV = 'data/typhoon/typhoon19-24complete_15min_improved.csv'
df = build_dataset(PRED_GLOB, TY_CSV)

D = df.iloc[:, 20].fillna(0).values.astype(float)
rmax = df['rmax_i'].fillna(0).values.astype(float)
ym = df.iloc[:, 39].fillna(0).values.astype(float)
rel_az = df.iloc[:, 22].fillna(0).values.astype(float)

df['dist_D_km'] = D
df['D_over_Rmax'] = np.where(rmax > 0.5, D / rmax, 0.0)
df['cos_rel_azimuth'] = np.cos(np.radians(rel_az))
df['sin_rel_azimuth'] = np.sin(np.radians(rel_az))
df['V_yanmeng_ms'] = ym
df['Vtan_tangential'] = ym * df['sin_rel_azimuth'].values
df['Vrad_radial'] = ym * df['cos_rel_azimuth'].values

G1_FEATS = ['y_pred_mw', 'lead_norm']
G4_FEATS = ['y_pred_mw', 'lead_norm', 'V_yanmeng_ms', 'Vtan_tangential', 'Vrad_radial']

events = df.groupby(df.columns[7])[df.columns[2]].min().sort_values().index.tolist()
n_events = len(events)
print(f"  {len(df)} rows, {n_events} typhoon events")

# ============================================================
# 2. Gate函数
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

def gate_B1(dist):
    """硬门控 [100, 350]"""
    return ((dist >= 100) & (dist <= 350)).astype(float)

def gate_B2(dist):
    """硬门控 [150, 300]"""
    return ((dist >= 150) & (dist <= 300)).astype(float)

def gate_B3(dist):
    """平滑门控: 100↗250↘400"""
    return sigmoid((dist - 100) / 30) * (1.0 - sigmoid((dist - 350) / 30))

def gate_C1(ym_vals):
    """V_yanmeng幅值门控"""
    return (np.abs(ym_vals) > 10).astype(float)

def gate_D1(dist, ym_vals):
    """组合门控"""
    return ((dist >= 100) & (dist <= 350) & (np.abs(ym_vals) > 5)).astype(float)

# Also: always use G1 as comparison, always use G4 as comparison
def gate_G1(dist): return np.zeros_like(dist)
def gate_G4(dist): return np.ones_like(dist)

strategies = {
    'G1_baseline': gate_G1,
    'G4_only': gate_G4,
    'B1_dist_100_350': gate_B1,
    'B2_dist_150_300': gate_B2,
    'B3_smooth_dist': gate_B3,
    'C1_vmag_gt10': gate_C1,
    'D1_dist_and_vmag': gate_D1,
}

# ============================================================
# 3. LOTO: 先训练所有fold, 再应用所有gate策略
# ============================================================
print("[2] LOTO: training G1 + G4 per fold...")

fold_data = []  # store per-fold predictions

for fold_i, test_event in enumerate(events, 1):
    remain = [e for e in events if e != test_event]
    remain_sorted = (
        df[df.iloc[:, 7].isin(remain)]
        .groupby(df.columns[7])[df.columns[2]].min().sort_values().index.tolist()
    )
    val_event = remain_sorted[-1]
    train_events = remain_sorted[:-1]

    train_df = df[df.iloc[:, 7].isin(train_events)].copy()
    val_df   = df[df.iloc[:, 7].isin([val_event])].copy()
    test_df  = df[df.iloc[:, 7].isin([test_event])].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        continue

    y_tr = train_df['delta_mw'].to_numpy(dtype=np.float32)
    y_va = val_df['delta_mw'].to_numpy(dtype=np.float32)

    # Train G1
    X_tr_g1 = train_df[G1_FEATS].to_numpy(dtype=np.float32)
    X_va_g1 = val_df[G1_FEATS].to_numpy(dtype=np.float32)
    X_te_g1 = test_df[G1_FEATS].to_numpy(dtype=np.float32)
    m1 = X_tr_g1.mean(axis=0); s1 = X_tr_g1.std(axis=0) + 1e-6
    g1_model, _ = train_elm_gwo(
        (X_tr_g1 - m1) / s1, y_tr, (X_va_g1 - m1) / s1, y_va,
        n_hidden=60, gamma=1.0, n_wolves=15, max_iter=20, n_trials=2, ridge=1.0)
    pred_g1 = g1_model.predict((X_te_g1 - m1) / s1)

    # Train G4
    X_tr_g4 = train_df[G4_FEATS].to_numpy(dtype=np.float32)
    X_va_g4 = val_df[G4_FEATS].to_numpy(dtype=np.float32)
    X_te_g4 = test_df[G4_FEATS].to_numpy(dtype=np.float32)
    m4 = X_tr_g4.mean(axis=0); s4 = X_tr_g4.std(axis=0) + 1e-6
    g4_model, _ = train_elm_gwo(
        (X_tr_g4 - m4) / s4, y_tr, (X_va_g4 - m4) / s4, y_va,
        n_hidden=60, gamma=1.0, n_wolves=15, max_iter=20, n_trials=2, ridge=1.0)
    pred_g4 = g4_model.predict((X_te_g4 - m4) / s4)

    base_te = np.clip(test_df['y_pred_mw'].to_numpy(dtype=np.float32), 0, CAPACITY)
    y_te = test_df['y_true_mw'].to_numpy(dtype=np.float32)

    fold_data.append({
        'event': test_event,
        'base': base_te,
        'y_true': y_te,
        'pred_g1': pred_g1,
        'pred_g4': pred_g4,
        'dist': test_df['dist_D_km'].values.astype(float),
        'ym': test_df['V_yanmeng_ms'].values.astype(float),
    })

    if fold_i % 5 == 0 or fold_i == 1:
        base_rmse = math.sqrt(np.mean((y_te - base_te)**2))
        g1_rmse = math.sqrt(np.mean((y_te - base_te - pred_g1)**2))
        g4_rmse = math.sqrt(np.mean((y_te - base_te - pred_g4)**2))
        print(f"    [{fold_i}/{n_events}] base={base_rmse:.1f} g1={g1_rmse:.1f} g4={g4_rmse:.1f}")

print(f"  Training done. {len(fold_data)} folds.")

# ============================================================
# 4. 对所有gate策略评估
# ============================================================
print("\n[3] Evaluating gate strategies...")

all_results = {}
for strat_name, gate_fn in strategies.items():
    fold_rows = []
    for fd in fold_data:
        pred_g1 = fd['pred_g1']
        pred_g4 = fd['pred_g4']
        d_test = fd['dist']
        ym_test = fd['ym']
        base_te = fd['base']
        y_te = fd['y_true']

        sig = inspect.signature(gate_fn)
        alpha = gate_fn(d_test) if len(sig.parameters) == 1 else gate_fn(d_test, ym_test)

        pred_blend = (1 - alpha) * pred_g1 + alpha * pred_g4
        corrected = base_te + pred_blend

        m = metrics(y_te, base_te, CAPACITY)
        m_corr = metrics(y_te, corrected, CAPACITY)
        imp = (m_corr['RMSE'] - m['RMSE']) / m['RMSE'] * 100

        fold_rows.append({
            'test_event': fd['event'],
            'base_RMSE': m['RMSE'], 'corr_RMSE': m_corr['RMSE'],
            'rmse_improve_pct': imp, 'gated_pct': float(alpha.mean() * 100),
        })

    fdf = pd.DataFrame(fold_rows)
    imp_m = fdf['rmse_improve_pct'].mean()
    imp_s = fdf['rmse_improve_pct'].std()
    win = (fdf['rmse_improve_pct'] > 0).mean() * 100

    all_results[strat_name] = {
        'base_RMSE': fdf['base_RMSE'].mean(), 'corr_RMSE': fdf['corr_RMSE'].mean(),
        'imp_mean': imp_m, 'imp_std': imp_s, 'win_rate': win,
        'gated_pct': fdf['gated_pct'].mean(),
    }
    fdf.to_csv(f'{MODEL_DIR}/fold_{strat_name}.csv', index=False, encoding='utf-8-sig')
    print(f"    {strat_name:<25s}: {all_results[strat_name]['base_RMSE']:.2f}->{all_results[strat_name]['corr_RMSE']:.2f} v{imp_m:.2f}% (±{imp_s:.1f}) win={win:.0f}% gate={all_results[strat_name]['gated_pct']:.0f}%")

# ============================================================
# 4. 汇总
# ============================================================
print("\n" + "=" * 70)
print("FINAL: V_yanmeng Conditional Correction")
print("=" * 70)
print(f"{'Strategy':<25s} {'RMSE':>8s} {'Imp%':>8s} {'Std':>6s} {'Win':>5s} {'Gate%':>6s}")
print("-" * 65)
best_imp = -999
best_name = ""
for name, r in all_results.items():
    arrow = "v" if r['imp_mean'] > 0 else "^"
    print(f"{name:<25s} {r['corr_RMSE']:8.2f} {arrow}{abs(r['imp_mean']):7.2f}% {r['imp_std']:6.1f} {r['win_rate']:5.0f}% {r['gated_pct']:6.0f}%")
    if r['imp_mean'] > best_imp:
        best_imp = r['imp_mean']
        best_name = name
print(f"\n  Best: {best_name} ({best_imp:+.2f}%)")

df_save = pd.DataFrame(all_results).T
df_save.to_csv(f'{MODEL_DIR}/summary.csv', encoding='utf-8-sig')
print(f"Results saved to {MODEL_DIR}/")
