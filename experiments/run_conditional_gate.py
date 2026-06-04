#!/usr/bin/env python3
"""
条件校正实验: 根据台风距离选择不同的GWO-ELM校正器

方案:
  A: G6 = G1 + B + Rmax + atten (已跑, ↓1.33% vs G1 ↓1.71%)
  B: 距离门控 G1/G4 — 150-300km用G4, 其余用G1
  C: V_yanmeng置信门控 — |V_yanmeng|大时用G4
  D: 平滑融合 — weight = f(dist, |V_yanmeng|)
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
MODEL_DIR = 'results/g0_g6_experiment'

# ============================================================
# 1. 加载数据集并添加衍生特征
# ============================================================
print("[1] Loading dataset...")
PRED_GLOB = 'data/predictions/bilstm_pred_*.csv'
TY_CSV = 'data/typhoon/typhoon19-24complete_15min_improved.csv'
df = build_dataset(PRED_GLOB, TY_CSV)

# Add derived features
D = df.iloc[:, 20].fillna(0).values.astype(float)
rmax = df['rmax_i'].fillna(0).values.astype(float)
ym = df.iloc[:, 39].fillna(0).values.astype(float)
rel_az = df.iloc[:, 22].fillna(0).values.astype(float)
b = df['b_i'].fillna(0).values.astype(float)
atten = df['atten_w'].fillna(0).values.astype(float)

df['dist_D_km'] = D
df['Rmax_km'] = rmax
df['B_param'] = b
df['atten_w'] = atten
df['V_yanmeng_ms'] = ym
df['Vtan_tangential'] = ym * np.sin(np.radians(rel_az))
df['Vrad_radial'] = ym * np.cos(np.radians(rel_az))

# ============================================================
# 2. 定义特征组
# ============================================================
G1_FEATS = ['y_pred_mw', 'lead_norm']
G4_FEATS = ['y_pred_mw', 'lead_norm', 'V_yanmeng_ms', 'Vtan_tangential', 'Vrad_radial']

# ============================================================
# 3. LOTO评估 with gating
# ============================================================
print("[2] Running LOTO with conditional gating...")

events = df.groupby(df.columns[7])[df.columns[2]].min().sort_values().index.tolist()
print(f"  {len(events)} typhoon events")

# Gating strategies
def gate_distance_hard(dist_vals, low=150, high=300):
    """硬门控: 距离在[low,high]时用G4, 否则用G1"""
    return ((dist_vals >= low) & (dist_vals <= high)).astype(float)

def gate_distance_smooth(dist_vals, center=225, width=75):
    """平滑门控: sigmoid过渡"""
    return 1.0 / (1.0 + np.exp(-(dist_vals - center + width/2) / (width/6))) * \
           (1.0 - 1.0 / (1.0 + np.exp(-(dist_vals - center - width/2) / (width/6))))

def gate_vmag(ym_vals, threshold=10):
    """V_yanmeng幅值门控: 风速>threshold时用G4"""
    return (np.abs(ym_vals) > threshold).astype(float)

def gate_combined(dist_vals, ym_vals, d_low=100, d_high=350, ym_thresh=5):
    """组合门控: 距离在范围 AND V_yanmeng足够大"""
    d_gate = ((dist_vals >= d_low) & (dist_vals <= d_high)).astype(float)
    v_gate = (np.abs(ym_vals) > ym_thresh).astype(float)
    return d_gate * v_gate

strategies = {
    'B_hard_gate_150_300': lambda d, ym: gate_distance_hard(d, 150, 300),
    'B_hard_gate_100_350': lambda d, ym: gate_distance_hard(d, 100, 350),
    'C_vmag_gate_10': lambda d, ym: gate_vmag(ym, 10),
    'C_vmag_gate_5': lambda d, ym: gate_vmag(ym, 5),
    'D_combined_d_v': lambda d, ym: gate_combined(d, ym, 100, 350, 5),
    'D_smooth_gate': lambda d, ym: gate_distance_smooth(d, 200, 200),
}

all_results = []

for strat_name, gate_fn in strategies.items():
    print(f"\n  --- {strat_name} ---")
    fold_rows = []
    t_start = time.time()

    for fold_i, test_event in enumerate(events, 1):
        remain = [e for e in events if e != test_event]
        remain_sorted = (
            df[df.iloc[:, 7].isin(remain)]
            .groupby(df.columns[7])[df.columns[2]].min().sort_values().index.tolist()
        )
        val_event = remain_sorted[-1]
        train_events = remain_sorted[:-1]

        train_df = df[df.iloc[:, 7].isin(train_events)].copy()
        val_df = df[df.iloc[:, 7].isin([val_event])].copy()
        test_df = df[df.iloc[:, 7].isin([test_event])].copy()

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            continue

        # Train G1 model
        X_tr_g1 = train_df[G1_FEATS].to_numpy(dtype=np.float32)
        y_tr = train_df['delta_mw'].to_numpy(dtype=np.float32)
        X_va_g1 = val_df[G1_FEATS].to_numpy(dtype=np.float32)
        y_va = val_df['delta_mw'].to_numpy(dtype=np.float32)
        X_te_g1 = test_df[G1_FEATS].to_numpy(dtype=np.float32)

        # Train G4 model
        X_tr_g4 = train_df[G4_FEATS].to_numpy(dtype=np.float32)
        X_va_g4 = val_df[G4_FEATS].to_numpy(dtype=np.float32)
        X_te_g4 = test_df[G4_FEATS].to_numpy(dtype=np.float32)

        # Standardize each
        g1_mean = X_tr_g1.mean(axis=0); g1_std = X_tr_g1.std(axis=0) + 1e-6
        g4_mean = X_tr_g4.mean(axis=0); g4_std = X_tr_g4.std(axis=0) + 1e-6

        X_tr_g1_n = (X_tr_g1 - g1_mean) / g1_std
        X_va_g1_n = (X_va_g1 - g1_mean) / g1_std
        X_te_g1_n = (X_te_g1 - g1_mean) / g1_std

        X_tr_g4_n = (X_tr_g4 - g4_mean) / g4_std
        X_va_g4_n = (X_va_g4 - g4_mean) / g4_std
        X_te_g4_n = (X_te_g4 - g4_mean) / g4_std

        # Train models
        g1_model, _ = train_elm_gwo(X_tr_g1_n, y_tr, X_va_g1_n, y_va,
                                     n_hidden=60, gamma=1.0, n_wolves=15, max_iter=20,
                                     n_trials=2, ridge=1.0)
        g4_model, _ = train_elm_gwo(X_tr_g4_n, y_tr, X_va_g4_n, y_va,
                                     n_hidden=60, gamma=1.0, n_wolves=15, max_iter=20,
                                     n_trials=2, ridge=1.0)

        # Get G1 and G4 predictions
        pred_g1 = g1_model.predict(X_te_g1_n)
        pred_g4 = g4_model.predict(X_te_g4_n)

        # Compute gate weights for each test sample
        d_test = test_df['dist_D_km'].values.astype(float)
        ym_test = test_df['V_yanmeng_ms'].values.astype(float)
        alpha = gate_fn(d_test, ym_test)  # 0=G1, 1=G4

        # Blended correction
        pred_blend = (1 - alpha) * pred_g1 + alpha * pred_g4

        # Apply correction
        base_te = np.clip(test_df['y_pred_mw'].to_numpy(dtype=np.float32), 0.0, CAPACITY)
        corrected = base_te + pred_blend
        y_te = test_df['y_true_mw'].to_numpy(dtype=np.float32)

        m = metrics(y_te, base_te, CAPACITY)
        m_corr = metrics(y_te, corrected, CAPACITY)
        imp = (m_corr['RMSE'] - m['RMSE']) / m['RMSE'] * 100
        gated_pct = alpha.mean() * 100

        fold_rows.append({
            'test_event': test_event,
            'base_RMSE': m['RMSE'],
            'corr_RMSE': m_corr['RMSE'],
            'base_MAE': m['MAE'],
            'corr_MAE': m_corr['MAE'],
            'rmse_improve_pct': imp,
            'gated_pct': gated_pct,
        })

        if fold_i % 5 == 0 or fold_i == 1:
            print(f"    [{fold_i}/{len(events)}] imp={imp:+.1f}% gated={gated_pct:.0f}% samples")

    # Summarize
    fold_df = pd.DataFrame(fold_rows)
    imp_mean = fold_df['rmse_improve_pct'].mean()
    imp_std = fold_df['rmse_improve_pct'].std()
    win_rate = (fold_df['rmse_improve_pct'] > 0).mean() * 100
    base_mean = fold_df['base_RMSE'].mean()
    corr_mean = fold_df['corr_RMSE'].mean()
    gated_mean = fold_df['gated_pct'].mean()

    result = {
        'strategy': strat_name,
        'base_RMSE': base_mean,
        'corr_RMSE': corr_mean,
        'imp_mean': imp_mean,
        'imp_std': imp_std,
        'win_rate': win_rate,
        'gated_pct_mean': gated_mean,
        'elapsed_min': (time.time() - t_start) / 60,
    }
    all_results.append(result)
    fold_df.to_csv(f'{MODEL_DIR}/conditional_{strat_name}.csv', index=False, encoding='utf-8-sig')

    print(f"    >>> {strat_name}: RMSE={base_mean:.2f}->{corr_mean:.2f}, "
          f"↓{imp_mean:.2f}% (±{imp_std:.1f}), win={win_rate:.0f}%, gated={gated_mean:.0f}%")

# ============================================================
# 4. 对比汇总
# ============================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

# G1 baseline from experiment
g1_folds = pd.read_csv(f'{MODEL_DIR}/loto_fold_G1_no_physics.csv', encoding='utf-8-sig')
g1_imp = g1_folds['rmse_improve_pct'].mean()
g1_win = (g1_folds['rmse_improve_pct'] > 0).mean() * 100

print(f"G1 (baseline):        ↓{g1_imp:.2f}%, win={g1_win:.0f}%")
for r in all_results:
    print(f"{r['strategy']:<25s}: ↓{r['imp_mean']:.2f}% (±{r['imp_std']:.1f}), win={r['win_rate']:.0f}%, gated={r['gated_pct_mean']:.0f}%")

# Save
df_summary = pd.DataFrame(all_results)
df_summary.to_csv(f'{MODEL_DIR}/conditional_gate_summary.csv', index=False, encoding='utf-8-sig')
print(f"\nResults saved to {MODEL_DIR}/")
print("Done!")
