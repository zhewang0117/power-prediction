#!/usr/bin/env python3
"""Distance-based hybrid KRR: G3 for FAR, G2/G5 for CORE"""
import numpy as np, pandas as pd, math, sys
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, '.')
from evaluate_typhoon_gwo_elm_loto import build_dataset

HORIZON = 96; CAPACITY = 200.0; PN = 1013.3
np.random.seed(42)

df = build_dataset('data/predictions/bilstm_pred_*.csv', 'data/typhoon/typhoon19-24complete_15min_improved.csv')
D = df.iloc[:, 20].fillna(0).values.astype(float)
rmax = df['rmax_i'].fillna(0).values.astype(float); b = df['b_i'].fillna(0).values.astype(float)
atten = df['atten_w'].fillna(0).values.astype(float)
ym = df.iloc[:, 39].fillna(0).values.astype(float); rel_az = df.iloc[:, 22].fillna(0).values.astype(float)
ws = df.iloc[:, 16].fillna(0).values.astype(float); pres = df.iloc[:, 17].fillna(PN).values.astype(float)
level = df.iloc[:, 15].fillna(0).values.astype(float); mvspd = df.iloc[:, 23].fillna(0).values.astype(float)

df['dist_D_km'] = D; df['Rmax_km'] = rmax; df['B_param'] = b; df['atten_w'] = atten
df['V_yanmeng_ms'] = ym
df['Vtan_tangential'] = ym * np.sin(np.radians(rel_az))
df['Vrad_radial'] = ym * np.cos(np.radians(rel_az))
df['cos_rel_azimuth'] = np.cos(np.radians(rel_az)); df['sin_rel_azimuth'] = np.sin(np.radians(rel_az))
df['Vmax_ms'] = ws; df['P_deficit_hPa'] = PN - pres
df['typhoon_level'] = level; df['Vmove_ms'] = mvspd

G3_F = ['y_pred_mw', 'lead_norm', 'Rmax_km', 'B_param', 'atten_w']
G2_F = ['y_pred_mw', 'lead_norm', 'V_yanmeng_ms', 'Vtan_tangential', 'Vrad_radial']
G5_F = ['y_pred_mw', 'lead_norm',
        'V_yanmeng_ms', 'Vtan_tangential', 'Vrad_radial',
        'Rmax_km', 'B_param', 'atten_w',
        'dist_D_km', 'D_over_Rmax', 'cos_rel_azimuth', 'sin_rel_azimuth',
        'Vmax_ms', 'P_deficit_hPa', 'typhoon_level', 'Vmove_ms']

HYBRIDS = {
    'Hyb_G3FAR_G2CORE': {'FAR': G3_F, 'CORE': G2_F, 'CLOSE': G3_F},
    'Hyb_G3FAR_G5CORE': {'FAR': G3_F, 'CORE': G5_F, 'CLOSE': G3_F},
}
STAGES = {'FAR': (350, 99999), 'CORE': (100, 350), 'CLOSE': (0, 100)}

all_events = df.groupby(df.columns[7])[df.columns[2]].min().sort_values().index.tolist()

def train_krr(tr_df, va_df, te_df, feats):
    X_tr = tr_df[feats].to_numpy(dtype=np.float32); y_tr = tr_df['delta_mw'].to_numpy(dtype=np.float32)
    X_va = va_df[feats].to_numpy(dtype=np.float32); y_va = va_df['delta_mw'].to_numpy(dtype=np.float32)
    X_te = te_df[feats].to_numpy(dtype=np.float32)
    scl = StandardScaler(); X_tr_s = scl.fit_transform(X_tr); X_va_s = scl.transform(X_va); X_te_s = scl.transform(X_te)
    if len(X_tr_s) > 5000:
        idx = np.random.choice(len(X_tr_s), 5000, replace=False); X_tr_k = X_tr_s[idx]; y_tr_k = y_tr[idx]
    else: X_tr_k = X_tr_s; y_tr_k = y_tr
    best_a = 1.0; best_v = 1e9
    for a in [0.1, 1.0, 10.0]:
        m = KernelRidge(kernel='rbf', alpha=a, gamma=1.0 / len(feats)); m.fit(X_tr_k, y_tr_k)
        v = np.mean((m.predict(X_va_s) - y_va) ** 2)
        if v < best_v: best_v = v; best_a = a
    model = KernelRidge(kernel='rbf', alpha=best_a, gamma=1.0 / len(feats)); model.fit(X_tr_k, y_tr_k)
    return scl, best_a, model.predict(X_te_s)

def rmse(y_true, y_pred, mask=None):
    if mask is not None: return math.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
    return math.sqrt(np.mean((y_true - y_pred) ** 2))

# Also run baselines for comparison
BASELINES = {
    'G2_KRR': G2_F,
    'G3_KRR': G3_F,
    'G5_KRR': G5_F,
}

print('Seed  Method                  GLOBAL   FAR(>350)  CORE(100-350) CLOSE(<100)')
print('-' * 80)

for seed in [42, 123, 456, 789, 1024]:
    np.random.seed(seed)
    shuffled = np.random.permutation(all_events)
    tr_ids = list(shuffled[:23]); te_ids = list(shuffled[23:])
    val_id = tr_ids[-1]; tr_no_val = [e for e in tr_ids if e != val_id]
    tr_df = df[df.iloc[:, 7].isin(tr_no_val)]
    va_df = df[df.iloc[:, 7].isin([val_id])]
    te_df = df[df.iloc[:, 7].isin(te_ids)]
    y_true = te_df['y_true_mw'].to_numpy(dtype=np.float32)
    y_base = np.clip(te_df['y_pred_mw'].to_numpy(dtype=np.float32), 0, CAPACITY)
    d_test = te_df['dist_D_km'].values.astype(float)
    g0_rmse = rmse(y_true, y_base)

    # Train all single models
    all_preds = {}
    all_models = {**BASELINES}
    for hname, stage_feats in HYBRIDS.items():
        for sname in ['FAR', 'CORE', 'CLOSE']:
            all_models[f'{hname}_{sname}'] = stage_feats[sname]

    deltas = {}
    for mname, feats in all_models.items():
        _, _, delta = train_krr(tr_df, va_df, te_df, feats)
        deltas[mname] = np.clip(y_base + delta, 0, CAPACITY)

    # Build hybrid predictions
    for hname in HYBRIDS:
        y_hyb = y_base.copy()
        for sname, (lo, hi) in STAGES.items():
            mask = (d_test >= lo) & (d_test < hi)
            y_hyb[mask] = deltas[f'{hname}_{sname}'][mask]
        # Smooth transition: blend 50km overlap
        for (s1, s2), (lo, hi) in [(('CORE','FAR'), (330, 370)), (('CLOSE','CORE'), (80, 120))]:
            mask = (d_test >= lo) & (d_test < hi)
            if mask.sum() > 0:
                w = np.clip((d_test[mask] - lo) / (hi - lo), 0, 1)
                y_hyb[mask] = (1 - w) * deltas[f'{hname}_{s1}'][mask] + w * deltas[f'{hname}_{s2}'][mask]
        deltas[hname] = y_hyb

    row_fmt = '%-5d %-22s %7.2f %10.2f %13.2f %13.2f'
    print(row_fmt % (seed, 'G0_Base', g0_rmse, rmse(y_true, y_base, d_test>=350),
                     rmse(y_true, y_base, (d_test>=100)&(d_test<=350)), rmse(y_true, y_base, d_test<100)))

    for mname in ['G2_KRR', 'G3_KRR', 'G5_KRR'] + list(HYBRIDS.keys()):
        yp = deltas[mname]
        print(row_fmt % (seed, mname, rmse(y_true, yp), rmse(y_true, yp, d_test>=350),
                         rmse(y_true, yp, (d_test>=100)&(d_test<=350)), rmse(y_true, yp, d_test<100)))
    print()

print('=' * 80)
print('5-seed MEAN summary:')

# Collect and average
all_methods = ['G0_Base', 'G2_KRR', 'G3_KRR', 'G5_KRR'] + list(HYBRIDS.keys())
means = {m: {'g': [], 'far': [], 'core': [], 'close': []} for m in all_methods}
# Re-run to collect means (lazy: just aggregate from above)
# Actually let me do a proper aggregation run
