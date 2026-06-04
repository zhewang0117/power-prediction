#!/usr/bin/env python3
"""甲方图: 台风期间 实际 vs 原始 vs KRR校正 功率对比"""
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

HORIZON = 96; CAPACITY = 200.0; PN = 1013.3; np.random.seed(42)
df = build_dataset('data/predictions/bilstm_pred_*.csv', 'data/typhoon/typhoon19-24complete_15min_improved.csv')

# Derived features using column indices to avoid encoding issues
ym = df.iloc[:, 39].fillna(0).values.astype(float)
rel_az = df.iloc[:, 22].fillna(0).values.astype(float)
df['V_yanmeng_ms'] = ym
df['Vtan_tangential'] = ym * np.sin(np.radians(rel_az))
df['Vrad_radial'] = ym * np.cos(np.radians(rel_az))
df['dist_D_km'] = df.iloc[:, 20].fillna(0).values.astype(float)

G2_F = ['y_pred_mw', 'lead_norm', 'V_yanmeng_ms', 'Vtan_tangential', 'Vrad_radial']

# Split 23/9
all_events = df.groupby(df.columns[7])[df.columns[2]].min().sort_values().index.tolist()
shuffled = np.random.permutation(all_events)
tr_ids = list(shuffled[:23]); te_ids = list(shuffled[23:])
val_id = tr_ids[-1]; tr_no_val = [e for e in tr_ids if e != val_id]
tr_df = df[df.iloc[:, 7].isin(tr_no_val)]
va_df = df[df.iloc[:, 7].isin([val_id])]
te_df = df[df.iloc[:, 7].isin(te_ids)]

# Train KRR G2
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

# ---- Select typhoon: 贝碧嘉 (202413) ----
ev = 202413.0
ev_mask = te_df.iloc[:, 7] == ev
ev_te = te_df[ev_mask].copy()
X_te = ev_te[G2_F].to_numpy(dtype=np.float32); X_te_s = scl.transform(X_te)
delta = model.predict(X_te_s)
y_true = ev_te['y_true_mw'].to_numpy(dtype=np.float32)
y_base = np.clip(ev_te['y_pred_mw'].to_numpy(dtype=np.float32), 0, CAPACITY)
y_corr = np.clip(y_base + 0.7 * delta, 0, CAPACITY)  # fixed lambda
d_test = ev_te['dist_D_km'].values.astype(float)
times = pd.to_datetime(ev_te.iloc[:, 2].values)
leads = ev_te['lead'].values
ty_name = str(ev_te.iloc[:, 8].dropna().iloc[0])

# 6h ahead showcase
LEAD_SHOW = 24
mask = leads == LEAD_SHOW
t = times[mask].values
yt = y_true[mask]; yb = y_base[mask]; yc = y_corr[mask]; ds = d_test[mask]

def smooth(x, w=4):
    return pd.Series(x).rolling(w, min_periods=1, center=True).mean().values

yt_s = smooth(yt); yb_s = smooth(yb); yc_s = smooth(yc); ds_s = smooth(ds)

core_mask = (ds >= 100) & (ds <= 350)
base_core_rmse = math.sqrt(np.mean((yt[core_mask] - yb[core_mask]) ** 2))
corr_core_rmse = math.sqrt(np.mean((yt[core_mask] - yc[core_mask]) ** 2))
core_imp = (corr_core_rmse - base_core_rmse) / base_core_rmse * 100
far_mask = ds >= 350
base_far_rmse = math.sqrt(np.mean((yt[far_mask] - yb[far_mask]) ** 2))
corr_far_rmse = math.sqrt(np.mean((yt[far_mask] - yc[far_mask]) ** 2))

print(f"Typhoon: {ty_name}, CORE imp: {core_imp:.1f}%, FAR: base={base_far_rmse:.1f} corr={corr_far_rmse:.1f}")

# ==== FIGURE ====
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(4, 1, height_ratios=[3.5, 1.2, 1.2, 0.8], hspace=0.06)

# ---- Panel A: Power prediction ----
ax = fig.add_subplot(gs[0])

# Zone backgrounds
for i in range(len(t) - 1):
    if ds_s[i] < 100: c = '#ffe0e0'
    elif ds_s[i] <= 350: c = '#d4f5d4'
    else: c = '#e0e8ff'
    ax.axvspan(t[i], t[min(i+1, len(t)-1)], alpha=0.4, color=c, linewidth=0)

# Power curves
ax.plot(t, yt_s, 'k-', linewidth=2.5, label='Actual Power (实际功率)', alpha=0.9, zorder=10)
ax.plot(t, yb_s, '--', color='#e74c3c', linewidth=2.0, label='Base BiLSTM (原始预测)', alpha=0.8, zorder=8)
ax.plot(t, yc_s, '-', color='#2ecc71', linewidth=2.5, label='KRR Corrected (校正后)', alpha=0.9, zorder=9)

# Fill: where correction helps in CORE
ci = np.where(core_mask)[0]
if len(ci) > 0:
    ax.fill_between(t[ci[0]:ci[-1]], yb_s[ci[0]:ci[-1]], yc_s[ci[0]:ci[-1]],
                    color='#2ecc71', alpha=0.2, label='Core Zone Improvement (核心区改善)')

# Annotate
core_start_idx = np.where(core_mask)[0]
if len(core_start_idx) > 0:
    mid = (core_start_idx[0] + core_start_idx[-1]) // 2
    ax.annotate(f'CORE ZONE\n核心影响区\nRMSE {abs(core_imp):.0f}%',
                xy=(t[mid], CAPACITY * 0.82), fontsize=13, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#d4f5d4', edgecolor='#27ae60', alpha=0.92, pad=1))

ax.set_ylabel('Power / 功率 (MW)', fontsize=13)
ax.set_title(f'Typhoon {ty_name} / 台风{ty_name} — 6h Ahead Power Prediction / 6小时超前预测',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower left', ncol=4, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_ylim(0, CAPACITY)

# ---- Panel B: Error comparison ----
ax2 = fig.add_subplot(gs[1], sharex=ax)
eb = np.abs(yt_s - yb_s); ec = np.abs(yt_s - yc_s)
ax2.fill_between(t, 0, eb, color='#e74c3c', alpha=0.25, label='Base Error (原始误差)')
ax2.fill_between(t, 0, ec, color='#2ecc71', alpha=0.35, label='Corrected Error (校正后误差)')
ax2.plot(t, eb, color='#e74c3c', linewidth=0.6, alpha=0.5)
ax2.plot(t, ec, color='#2ecc71', linewidth=1.0, alpha=0.7)
for i in range(len(t) - 1):
    if ds_s[i] < 100: c = '#ffe0e0'
    elif ds_s[i] <= 350: c = '#d4f5d4'
    else: c = '#e0e8ff'
    ax2.axvspan(t[i], t[min(i+1, len(t)-1)], alpha=0.3, color=c, linewidth=0)
ax2.set_ylabel('|Error| (MW)', fontsize=12)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.2)

# ---- Panel C: Error reduction (base_err - corr_err) ----
ax3 = fig.add_subplot(gs[2], sharex=ax)
reduction = eb - ec  # positive = correction helped
ax3.fill_between(t, 0, reduction, where=reduction>0, color='#2ecc71', alpha=0.4, label='Correction Improved (校正改善)')
ax3.fill_between(t, 0, reduction, where=reduction<=0, color='#e74c3c', alpha=0.3, label='Correction Degraded (校正变差)')
ax3.axhline(0, color='gray', linewidth=0.5)
for i in range(len(t) - 1):
    if ds_s[i] < 100: c = '#ffe0e0'
    elif ds_s[i] <= 350: c = '#d4f5d4'
    else: c = '#e0e8ff'
    ax3.axvspan(t[i], t[min(i+1, len(t)-1)], alpha=0.3, color=c, linewidth=0)
ax3.set_ylabel('Error Reduction\n误差减少量 (MW)', fontsize=12)
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.2)

# ---- Panel D: Distance ----
ax4 = fig.add_subplot(gs[3], sharex=ax)
ax4.fill_between(t, 0, ds_s, color='#3498db', alpha=0.3)
ax4.plot(t, ds_s, 'b-', linewidth=1.8)
ax4.axhline(100, color='orange', linestyle='--', linewidth=1.2)
ax4.axhline(350, color='orange', linestyle='--', linewidth=1.2)
ax4.text(t[len(t)//2], 120, 'Core Zone / 核心区 (100-350km)', fontsize=10, color='orange', ha='center', fontweight='bold')
ax4.set_ylabel('Distance\n距离 (km)', fontsize=12)
ax4.set_xlabel('Time / 时间', fontsize=13)
ax4.grid(True, alpha=0.2)
ax4.invert_yaxis()

# Legend for zones
from matplotlib.patches import Patch
leg = [Patch(facecolor='#ffe0e0', alpha=0.5, label='Close (<100km)'),
       Patch(facecolor='#d4f5d4', alpha=0.5, label='Core (100-350km)'),
       Patch(facecolor='#e0e8ff', alpha=0.5, label='Far (>350km)')]
fig.legend(handles=leg, loc='lower center', ncol=3, fontsize=10, framealpha=0.9)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig('results/paper_figures/stakeholder_figure.png', dpi=300)
plt.close(fig)
print("Saved: results/paper_figures/stakeholder_figure.png")
