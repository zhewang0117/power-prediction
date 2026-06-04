#!/usr/bin/env python3
"""
23/9 台风训练/测试拆分 + 多维指标评估

指标:
  1. Global RMSE/MAE    — 测试集全部样本
  2. Typhoon RMSE       — 仅台风影响时段
  3. Effective RMSE     — 有效区间 (dist 100-350km)
  4. Direction Accuracy  — 校正方向正确率
  5. Ramp RMSE          — 功率骤变时段 (|deltaP|>20%cap in 1h)

对比组:
  G0: Base BiLSTM (无校正)
  G1: y_pred + lead (基线校正)
  G2: G1 + V_yanmeng + Vtan + Vrad (参数风场)
  G3: G1 + Rmax + B + atten (结构参数)
"""

import os, sys, time, math, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_typhoon_gwo_elm_loto import build_dataset, metrics, train_elm_gwo

HORIZON = 96; CAPACITY = 200.0; PN = 1013.3
OUT_DIR = 'results/train_test_split'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载+衍生特征
# ============================================================
print("[1] Loading data...")
df = build_dataset('data/predictions/bilstm_pred_*.csv', 'data/typhoon/typhoon19-24complete_15min_improved.csv')

D = df.iloc[:, 20].fillna(0).values.astype(float)
rmax = df['rmax_i'].fillna(0).values.astype(float)
b = df['b_i'].fillna(0).values.astype(float)
atten = df['atten_w'].fillna(0).values.astype(float)
ym = df.iloc[:, 39].fillna(0).values.astype(float)
rel_az = df.iloc[:, 22].fillna(0).values.astype(float)

df['dist_D_km'] = D
df['D_over_Rmax'] = np.where(rmax > 0.5, D / rmax, 0.0)
df['Rmax_km'] = rmax
df['B_param'] = b
df['atten_w'] = atten
df['V_yanmeng_ms'] = ym
df['Vtan_tangential'] = ym * np.sin(np.radians(rel_az))
df['Vrad_radial'] = ym * np.cos(np.radians(rel_az))

# ============================================================
# 2. 台风事件分割: 按最早出现时间排序, 前23训练, 后9测试
# ============================================================
print("[2] Splitting 23 train / 9 test...")
event_times = df.groupby(df.columns[7])[df.columns[2]].min().sort_values()
event_ids = event_times.index.tolist()
train_ids = event_ids[:23]
test_ids = event_ids[23:]

train_df = df[df.iloc[:, 7].isin(train_ids)].copy()
# Valid: use last train typhoon for val
val_id = train_ids[-1]
train_no_val = [e for e in train_ids if e != val_id]
val_df = df[df.iloc[:, 7].isin([val_id])].copy()
train_sub = df[df.iloc[:, 7].isin(train_no_val)].copy()

test_df = df[df.iloc[:, 7].isin(test_ids)].copy()

print(f"  Train: {len(train_sub)} rows, {len(train_no_val)} typhoons")
print(f"  Val:   {len(val_df)} rows, 1 typhoon")
print(f"  Test:  {len(test_df)} rows, {len(test_ids)} typhoons")
test_names = [str(n) for n in test_df.iloc[:, 8].dropna().unique()]
print(f"  Test typhoons: {test_names}")

# ============================================================
# 3. 定义特征组
# ============================================================
FEAT_GROUPS = {
    "G1_pred_baseline":  ["y_pred_mw", "lead_norm"],
    "G2_param_wind":     ["y_pred_mw", "lead_norm", "V_yanmeng_ms", "Vtan_tangential", "Vrad_radial"],
    "G3_structural":     ["y_pred_mw", "lead_norm", "Rmax_km", "B_param", "atten_w"],
}

# ============================================================
# 4. 训练+评估
# ============================================================
print("\n[3] Training and evaluating...")

def compute_ramp_mask(power_vals, threshold_pct=0.2):
    """标记功率骤变样本: 1小时内变化超过threshold_pct*CAPACITY"""
    ramp = np.zeros(len(power_vals), dtype=bool)
    for i in range(4, len(power_vals)):  # 1h=4steps
        delta = abs(power_vals[i] - power_vals[i-4])
        if delta > threshold_pct * CAPACITY:
            ramp[i] = True
    return ramp

def evaluate_all(y_true, y_base, y_corr, dist_vals, lead_vals, group_name):
    """计算所有多维指标"""
    e_base = y_true - y_base
    e_corr = y_true - y_corr

    # 1. Global
    g_rmse_b = math.sqrt(np.mean(e_base**2))
    g_rmse_c = math.sqrt(np.mean(e_corr**2))
    g_mae_b = float(np.mean(np.abs(e_base)))
    g_mae_c = float(np.mean(np.abs(e_corr)))

    # 2. Typhoon window (all test data = typhoon by construction)
    ty_rmse_b = g_rmse_b
    ty_rmse_c = g_rmse_c

    # 3. Effective zone: distance 100-350km
    ez_mask = (dist_vals >= 100) & (dist_vals <= 350)
    if ez_mask.sum() > 10:
        ez_rmse_b = math.sqrt(np.mean(e_base[ez_mask]**2))
        ez_rmse_c = math.sqrt(np.mean(e_corr[ez_mask]**2))
    else:
        ez_rmse_b = ez_rmse_c = np.nan

    # 4. Direction accuracy: does the correction move in the right direction?
    # No correction → no direction. Only for corrected models.
    if group_name != "G0_Base":
        # Correction delta applied: delta = y_corr - y_base
        delta = y_corr - y_base
        # Correct direction: sign(delta) == sign(e_base)
        correct = (np.sign(delta) == np.sign(e_base)) & (np.abs(delta) > 1e-6)
        dir_acc = float(correct.mean())
    else:
        dir_acc = np.nan

    # 5. Ramp error: power changes > 20% capacity in 1h
    ramp_mask = compute_ramp_mask(y_true)
    if ramp_mask.sum() > 10:
        ramp_rmse_b = math.sqrt(np.mean(e_base[ramp_mask]**2))
        ramp_rmse_c = math.sqrt(np.mean(e_corr[ramp_mask]**2))
    else:
        ramp_rmse_b = ramp_rmse_c = np.nan

    return {
        'group': group_name,
        'n_samples': len(y_true),
        'g_rmse_b': g_rmse_b, 'g_rmse_c': g_rmse_c,
        'g_mae_b': g_mae_b, 'g_mae_c': g_mae_c,
        'ty_rmse_b': ty_rmse_b, 'ty_rmse_c': ty_rmse_c,
        'ez_rmse_b': ez_rmse_b, 'ez_rmse_c': ez_rmse_c,
        'ez_n': int(ez_mask.sum()),
        'dir_acc': dir_acc,
        'ramp_rmse_b': ramp_rmse_b, 'ramp_rmse_c': ramp_rmse_c,
        'ramp_n': int(ramp_mask.sum()),
        'g_imp': (g_rmse_c - g_rmse_b) / g_rmse_b * 100,
        'ty_imp': (ty_rmse_c - ty_rmse_b) / ty_rmse_b * 100,
        'ez_imp': (ez_rmse_c - ez_rmse_b) / ez_rmse_b * 100 if not np.isnan(ez_rmse_b) else np.nan,
        'ramp_imp': (ramp_rmse_c - ramp_rmse_b) / ramp_rmse_b * 100 if not np.isnan(ramp_rmse_b) else np.nan,
    }

results = []

# G0: Base BiLSTM (no correction)
y_true = test_df['y_true_mw'].to_numpy(dtype=np.float32)
y_base = np.clip(test_df['y_pred_mw'].to_numpy(dtype=np.float32), 0, CAPACITY)
dist_test = test_df['dist_D_km'].values.astype(float)
lead_test = test_df['lead_norm'].values.astype(float)
results.append(evaluate_all(y_true, y_base, y_base, dist_test, lead_test, "G0_Base"))

# G1-G3: Train correctors
for gname, feats in FEAT_GROUPS.items():
    print(f"\n  {gname}: {feats}")
    t0 = time.time()

    # Prepare data
    X_tr = train_sub[feats].to_numpy(dtype=np.float32)
    y_tr = train_sub['delta_mw'].to_numpy(dtype=np.float32)
    X_va = val_df[feats].to_numpy(dtype=np.float32)
    y_va = val_df['delta_mw'].to_numpy(dtype=np.float32)
    X_te = test_df[feats].to_numpy(dtype=np.float32)

    # Standardize
    m = X_tr.mean(axis=0); s = X_tr.std(axis=0) + 1e-6
    X_tr_n = (X_tr - m) / s; X_va_n = (X_va - m) / s; X_te_n = (X_te - m) / s

    # Train GWO-ELM
    model, _ = train_elm_gwo(X_tr_n, y_tr, X_va_n, y_va,
                              n_hidden=80, gamma=1.0,
                              n_wolves=20, max_iter=25, n_trials=3, ridge=1.0)

    # Predict
    delta_pred = model.predict(X_te_n)
    y_corr = y_base + delta_pred
    y_corr = np.clip(y_corr, 0, CAPACITY)

    r = evaluate_all(y_true, y_base, y_corr, dist_test, lead_test, gname)
    r['elapsed_s'] = time.time() - t0
    results.append(r)

    print(f"    Global:  {r['g_rmse_b']:.2f} -> {r['g_rmse_c']:.2f} ({r['g_imp']:+.2f}%)")
    print(f"    Typhoon: {r['ty_rmse_b']:.2f} -> {r['ty_rmse_c']:.2f} ({r['ty_imp']:+.2f}%)")
    print(f"    EffZ 100-350km: {r['ez_rmse_b']:.2f} -> {r['ez_rmse_c']:.2f} ({r['ez_imp']:+.2f}%) (n={r['ez_n']})")
    print(f"    DirAcc: {r['dir_acc']*100:.1f}%")
    if not np.isnan(r.get('ramp_rmse_b', np.nan)):
        print(f"    Ramp:   {r['ramp_rmse_b']:.2f} -> {r['ramp_rmse_c']:.2f} ({r['ramp_imp']:+.2f}%) (n={r['ramp_n']})")

# ============================================================
# 5. 汇总表
# ============================================================
print("\n" + "=" * 90)
print("FINAL: 23-train / 9-test Multi-Metric Comparison")
print("=" * 90)

df_r = pd.DataFrame(results)
for metric_name, col_b, col_c in [
    ("Global RMSE", "g_rmse_b", "g_rmse_c"),
    ("Typhoon RMSE", "ty_rmse_b", "ty_rmse_c"),
    ("EffZone RMSE (100-350km)", "ez_rmse_b", "ez_rmse_c"),
    ("Ramp RMSE", "ramp_rmse_b", "ramp_rmse_c"),
]:
    print(f"\n--- {metric_name} ---")
    print(f"  {'Group':<22s} {'Base':>8s} {'Corr':>8s} {'Improve':>8s}")
    print(f"  {'-'*46}")
    for _, row in df_r.iterrows():
        b = row[col_b]; c = row[col_c]
        if pd.isna(b) or pd.isna(c): continue
        imp = (c - b) / b * 100
        arrow = "v" if imp < 0 else "^"
        print(f"  {row['group']:<22s} {b:8.2f} {c:8.2f} {arrow}{abs(imp):7.2f}%")

print(f"\n--- Direction Accuracy ---")
for _, row in df_r.iterrows():
    if pd.isna(row.get('dir_acc')): continue
    print(f"  {row['group']:<22s} {row['dir_acc']*100:5.1f}%")

df_r.to_csv(f'{OUT_DIR}/multi_metric_results.csv', index=False, encoding='utf-8-sig')
print(f"\nSaved to {OUT_DIR}/")
