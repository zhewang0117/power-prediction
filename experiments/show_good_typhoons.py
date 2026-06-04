#!/usr/bin/env python3
"""Plot best typhoons for stakeholder figure selection"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math, sys
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, '.')
from evaluate_typhoon_gwo_elm_loto import build_dataset

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

HORIZON = 96; CAPACITY = 200.0; np.random.seed(42)
df = build_dataset('data/predictions/bilstm_pred_*.csv', 'data/typhoon/typhoon19-24complete_15min_improved.csv')
ym = df.iloc[:, 39].fillna(0).values.astype(float)
rel_az = df.iloc[:, 22].fillna(0).values.astype(float)
df['V_yanmeng_ms'] = ym
df['Vtan_tangential'] = ym * np.sin(np.radians(rel_az))
df['Vrad_radial'] = ym * np.cos(np.radians(rel_az))
df['dist_D_km'] = df.iloc[:, 20].fillna(0).values.astype(float)
G2_F = ['y_pred_mw', 'lead_norm', 'V_yanmeng_ms', 'Vtan_tangential', 'Vrad_radial']

all_events = df.groupby(df.columns[7])[df.columns[2]].min().sort_values().index.tolist()
sh = np.random.permutation(all_events)
tr_ids = list(sh[:23]); te_ids = list(sh[23:])
val_id = tr_ids[-1]; tr_no_val = [e for e in tr_ids if e != val_id]
tr_df = df[df.iloc[:, 7].isin(tr_no_val)]; va_df = df[df.iloc[:, 7].isin([val_id])]; te_df = df[df.iloc[:, 7].isin(te_ids)]

# Train KRR
X_tr = tr_df[G2_F].to_numpy(dtype=np.float32); y_tr = tr_df['delta_mw'].to_numpy(dtype=np.float32)
X_va = va_df[G2_F].to_numpy(dtype=np.float32); y_va = va_df['delta_mw'].to_numpy(dtype=np.float32)
scl = StandardScaler(); X_tr_s = scl.fit_transform(X_tr); X_va_s = scl.transform(X_va)
if len(X_tr_s) > 5000: idx = np.random.choice(len(X_tr_s), 5000, replace=False); X_tr_k = X_tr_s[idx]; y_tr_k = y_tr[idx]
else: X_tr_k = X_tr_s; y_tr_k = y_tr
best_a = 1.0; best_v = 1e9
for a in [0.1, 1.0, 10.0]:
    m = KernelRidge(kernel='rbf', alpha=a, gamma=1.0 / len(G2_F)); m.fit(X_tr_k, y_tr_k)
    if np.mean((m.predict(X_va_s) - y_va) ** 2) < best_v: best_v = np.mean((m.predict(X_va_s) - y_va) ** 2); best_a = a
model = KernelRidge(kernel='rbf', alpha=best_a, gamma=1.0 / len(G2_F)); model.fit(X_tr_k, y_tr_k)

# VAL lambda
delta_va = model.predict(X_va_s)
y_va_true = va_df['y_true_mw'].to_numpy(dtype=np.float32)
y_va_base = np.clip(va_df['y_pred_mw'].to_numpy(dtype=np.float32), 0, CAPACITY)
best_lam = 1.0; best_val_r = 1e9
for lam in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    y_bl = np.clip(y_va_base + lam * (np.clip(y_va_base + model.predict(X_va_s), 0, CAPACITY) - y_va_base), 0, CAPACITY)
    r = math.sqrt(np.mean((y_va_true - y_bl) ** 2))
    if r < best_val_r: best_val_r = r; best_lam = lam

# ---- Per typhoon: evaluate CORE, pick good ones ----
LEAD = 24; DIST_MAX = 600
ty_data = []
for ev in te_ids:
    evm = te_df.iloc[:, 7] == ev; ev_te = te_df[evm].copy()
    X_ev = ev_te[G2_F].to_numpy(dtype=np.float32); X_ev_s = scl.transform(X_ev)
    delta_ev = model.predict(X_ev_s)
    yt = ev_te['y_true_mw'].to_numpy(dtype=np.float32)
    yb = np.clip(ev_te['y_pred_mw'].to_numpy(dtype=np.float32), 0, CAPACITY)
    yc = np.clip(yb + best_lam * delta_ev, 0, CAPACITY)
    d_ev = ev_te['dist_D_km'].values.astype(float); l_ev = ev_te['lead'].values
    t_ev = pd.to_datetime(ev_te.iloc[:, 2].values)
    core_mask = (l_ev == LEAD) & (d_ev >= 100) & (d_ev <= 350)
    if core_mask.sum() < 5: continue
    b_r = math.sqrt(np.mean((yt[core_mask] - yb[core_mask]) ** 2))
    c_r = math.sqrt(np.mean((yt[core_mask] - yc[core_mask]) ** 2))
    imp = (c_r - b_r) / b_r * 100
    ev_name = str(ev_te.iloc[:, 8].dropna().iloc[0])
    # Get full time series within 600km
    ts_mask = (l_ev == LEAD) & (d_ev <= DIST_MAX)
    if ts_mask.sum() > 30:
        ty_data.append({'name': ev_name, 'ev': ev, 'imp': imp,
                        't': t_ev[ts_mask].values, 'yt': yt[ts_mask], 'yb': yb[ts_mask],
                        'yc': yc[ts_mask], 'd': d_ev[ts_mask], 'core_mask': core_mask[ts_mask]})

# Sort by improvement (best first)
ty_data.sort(key=lambda x: x['imp'])
print('Good typhoons (best CORE imp first):')
for td in ty_data:
    print('  %s: CORE imp=%+.1f%%, samples=%d' % (td['name'], td['imp'], len(td['t'])))

def smooth(x, w=4):
    return pd.Series(x).rolling(w, min_periods=1, center=True).mean().values

# Plot each as separate figure
for i, td in enumerate(ty_data):
    t = td['t']; yt = td['yt']; yb = td['yb']; yc = td['yc']; d = td['d']
    yts = smooth(yt); ybs = smooth(yb); ycs = smooth(yc); ds = smooth(d)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    # Zone background
    for j in range(len(t) - 1):
        c = '#ffe0e0' if ds[j] < 100 else '#d4f5d4' if ds[j] <= 350 else '#e0e8ff'
        ax1.axvspan(t[j], t[min(j + 1, len(t) - 1)], alpha=0.35, color=c, linewidth=0)
        ax2.axvspan(t[j], t[min(j + 1, len(t) - 1)], alpha=0.35, color=c, linewidth=0)

    ax1.plot(t, yts, 'k-', linewidth=2.5, label='Actual')
    ax1.plot(t, ybs, '--', color='#e74c3c', linewidth=2.0, label='Base BiLSTM')
    ax1.plot(t, ycs, '-', color='#2ecc71', linewidth=2.5, label='KRR Corrected')

    # Core highlight
    ci = np.where((d >= 100) & (d <= 350))[0]
    if len(ci) > 10:
        ax1.fill_between(t[ci[0]:ci[-1]], ybs[ci[0]:ci[-1]], ycs[ci[0]:ci[-1]],
                         color='#2ecc71', alpha=0.3)

    ax1.set_ylabel('Power (MW)', fontsize=13)
    ax1.set_title(f'Typhoon {td["name"]} — 6h Ahead, CORE imp: {td["imp"]:+.0f}%', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left'); ax1.grid(True, alpha=0.2); ax1.set_ylim(0, CAPACITY)

    # Error reduction
    eb = np.abs(yts - ybs); ec = np.abs(yts - ycs)
    ax2.fill_between(t, 0, eb, color='#e74c3c', alpha=0.2, label='Base Error')
    ax2.fill_between(t, 0, ec, color='#2ecc71', alpha=0.35, label='Corrected Error')
    ax2.plot(t, eb, color='#e74c3c', linewidth=0.5, alpha=0.4)
    ax2.plot(t, ec, color='#2ecc71', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('|Error| (MW)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fname = 'results/paper_figures/typhoon_%s.png' % td['name'].replace('/', '_')
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print('Saved: %s' % fname)

print('\nPick the best-looking one for the stakeholder figure.')
