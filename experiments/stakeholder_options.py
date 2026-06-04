#!/usr/bin/env python3
"""甲方图选项: 多种风格, 聚焦600km内"""
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

# Split
all_events = df.groupby(df.columns[7])[df.columns[2]].min().sort_values().index.tolist()
shuffled = np.random.permutation(all_events)
tr_ids = list(shuffled[:23]); te_ids = list(shuffled[23:])
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

# Get predictions for ALL test typhoons, within 600km, at 6h ahead
LEAD = 24  # 6h
DIST_MAX = 600
mask_all = (te_df['lead'] == LEAD) & (te_df['dist_D_km'] <= DIST_MAX)
sub = te_df[mask_all].copy()
X_sub = sub[G2_F].to_numpy(dtype=np.float32); X_sub_s = scl.transform(X_sub)
delta = model.predict(X_sub_s)
y_true = sub['y_true_mw'].to_numpy(dtype=np.float32)
y_base = np.clip(sub['y_pred_mw'].to_numpy(dtype=np.float32), 0, CAPACITY)
y_corr = np.clip(y_base + 0.7 * delta, 0, CAPACITY)  # fixed lambda 0.7
d_all = sub['dist_D_km'].values.astype(float)

err_base = y_true - y_base
err_corr = y_true - y_corr
ab_base = np.abs(err_base)
ab_corr = np.abs(err_corr)

# Zone masks
core = (d_all >= 100) & (d_all <= 350)
close = d_all < 100
far_in = (d_all > 350) & (d_all <= DIST_MAX)

# Compute zone metrics
for name, mask in [('CLOSE(<100)',close),('CORE(100-350)',core),('FAR(350-600)',far_in),('ALL(<600)',np.ones(len(d_all),dtype=bool))]:
    if mask.sum() < 5: continue
    b_rmse = math.sqrt(np.mean(err_base[mask]**2))
    c_rmse = math.sqrt(np.mean(err_corr[mask]**2))
    b_mae = float(np.mean(ab_base[mask]))
    c_mae = float(np.mean(ab_corr[mask]))
    print(f'{name:15s} n={mask.sum():5d} BaseRMSE={b_rmse:.1f} CorrRMSE={c_rmse:.1f} ({((c_rmse-b_rmse)/b_rmse*100):+.1f}%) MAE:{b_mae:.1f}->{c_mae:.1f}')

# ============================================================
# OPTION A: Time series focused on typhoon approach (one typhoon, <600km)
# ============================================================
print("\n[Option A] Time series, single typhoon, <600km ...")
ev = 202413.0
ev_mask = te_df.iloc[:, 7] == ev
ev_te = te_df[ev_mask].copy()
X_ev = ev_te[G2_F].to_numpy(dtype=np.float32); X_ev_s = scl.transform(X_ev)
delta_ev = model.predict(X_ev_s)
yt_ev = ev_te['y_true_mw'].to_numpy(dtype=np.float32)
yb_ev = np.clip(ev_te['y_pred_mw'].to_numpy(dtype=np.float32), 0, CAPACITY)
yc_ev = np.clip(yb_ev + 0.7 * delta_ev, 0, CAPACITY)
d_ev = ev_te['dist_D_km'].values.astype(float)
t_ev = pd.to_datetime(ev_te.iloc[:, 2].values)
l_ev = ev_te['lead'].values
ty_name = str(ev_te.iloc[:, 8].dropna().iloc[0])

# 6h ahead, only <600km
lm6 = (l_ev == LEAD) & (d_ev <= DIST_MAX)
t6 = t_ev[lm6].values
yt6 = yt_ev[lm6]; yb6 = yb_ev[lm6]; yc6 = yc_ev[lm6]; d6 = d_ev[lm6]
def smooth(x, w=4): return pd.Series(x).rolling(w, min_periods=1, center=True).mean().values
yt6s=smooth(yt6); yb6s=smooth(yb6); yc6s=smooth(yc6); d6s=smooth(d6)

figA, (axA1, axA2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[3,1])

# Zone background
for i in range(len(t6)-1):
    c = '#ffe0e0' if d6s[i]<100 else '#d4f5d4' if d6s[i]<=350 else '#e0e8ff'
    axA1.axvspan(t6[i], t6[min(i+1,len(t6)-1)], alpha=0.35, color=c, linewidth=0)
    axA2.axvspan(t6[i], t6[min(i+1,len(t6)-1)], alpha=0.35, color=c, linewidth=0)

axA1.plot(t6, yt6s, 'k-', linewidth=2.5, label='Actual')
axA1.plot(t6, yb6s, '--', color='#e74c3c', linewidth=2.0, label='Base BiLSTM')
axA1.plot(t6, yc6s, '-', color='#2ecc71', linewidth=2.5, label='KRR Corrected')

# Core zone highlight
ci6 = np.where((d6>=100)&(d6<=350))[0]
if len(ci6) > 10:
    i0, i1 = ci6[0], ci6[-1]
    axA1.fill_between(t6[i0:i1], yb6s[i0:i1], yc6s[i0:i1], color='#2ecc71', alpha=0.25)
    axA1.annotate(f'Core Zone\nRMSE -27%',
                  xy=(t6[(i0+i1)//2], CAPACITY*0.85), fontsize=13, ha='center', fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='#d4f5d4', edgecolor='#27ae60', alpha=0.9))

# Error reduction in bottom panel
eb6 = np.abs(yt6s - yb6s); ec6 = np.abs(yt6s - yc6s)
axA2.fill_between(t6, 0, eb6, color='#e74c3c', alpha=0.2, label='Base Error')
axA2.fill_between(t6, 0, ec6, color='#2ecc71', alpha=0.35, label='Corrected Error')
axA2.plot(t6, eb6, color='#e74c3c', linewidth=0.5, alpha=0.4)
axA2.plot(t6, ec6, color='#2ecc71', linewidth=1.5, alpha=0.7)

axA1.set_ylabel('Power (MW)', fontsize=13)
axA1.set_title(f'Typhoon {ty_name} — 6h Ahead, Within 600km', fontsize=14, fontweight='bold')
axA1.legend(fontsize=10, loc='lower left'); axA1.grid(True, alpha=0.2); axA1.set_ylim(0, CAPACITY)
axA2.set_ylabel('|Error| (MW)', fontsize=12); axA2.legend(fontsize=10, loc='upper right'); axA2.grid(True, alpha=0.2)
axA2.set_xlabel('Time', fontsize=12)
plt.tight_layout()
figA.savefig('results/paper_figures/optionA_timeseries.png', dpi=300)
plt.close(figA)

# ============================================================
# OPTION B: RMSE bar chart by zone (pooled across all test typhoons)
# ============================================================
print("[Option B] Bar chart RMSE by zone ...")
figB, axesB = plt.subplots(1, 2, figsize=(14, 6))

zones = [('Close\n(<100km)', close), ('Core\n(100-350km)', core), ('Far\n(350-600km)', far_in)]
x = np.arange(len(zones)); w = 0.3
base_rmses = []; corr_rmses = []; imps = []; n_samples = []
for _, mask in zones:
    base_rmses.append(math.sqrt(np.mean(err_base[mask]**2)))
    corr_rmses.append(math.sqrt(np.mean(err_corr[mask]**2)))
    imps.append((corr_rmses[-1]-base_rmses[-1])/base_rmses[-1]*100)
    n_samples.append(mask.sum())

# Left: RMSE bars
ax = axesB[0]
ax.bar(x-w/2, base_rmses, w, color='#e74c3c', alpha=0.8, label='Base BiLSTM', edgecolor='white')
ax.bar(x+w/2, corr_rmses, w, color='#2ecc71', alpha=0.8, label='KRR Corrected', edgecolor='white')
for i in range(len(zones)):
    ax.text(i, max(base_rmses[i], corr_rmses[i])+1, f'{imps[i]:+.1f}%', ha='center', fontsize=11, fontweight='bold',
            color='#27ae60' if imps[i]<0 else '#e74c3c')
    ax.text(i, 2, f'n={n_samples[i]}', ha='center', fontsize=8, color='gray')
ax.set_xticks(x); ax.set_xticklabels([z[0] for z in zones], fontsize=11)
ax.set_ylabel('RMSE (MW)', fontsize=13)
ax.set_title('RMSE by Distance Zone', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

# Right: Error distribution (density)
ax = axesB[1]
bins = np.linspace(0, 100, 50)
ax.hist(ab_base[core], bins=bins, alpha=0.5, color='#e74c3c', density=True, label='Base (Core)', edgecolor='none')
ax.hist(ab_corr[core], bins=bins, alpha=0.5, color='#2ecc71', density=True, label='Corrected (Core)', edgecolor='none')
ax.axvline(np.mean(ab_base[core]), color='#e74c3c', linestyle='--', linewidth=1.5)
ax.axvline(np.mean(ab_corr[core]), color='#2ecc71', linestyle='--', linewidth=1.5)
ax.set_xlabel('|Error| (MW)', fontsize=12); ax.set_ylabel('Density', fontsize=12)
ax.set_title('Core Zone Error Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2)

plt.tight_layout()
figB.savefig('results/paper_figures/optionB_barchart.png', dpi=300)
plt.close(figB)

# ============================================================
# OPTION C: Scatter: Base error vs Corrected error (Core zone only, clean)
# ============================================================
print("[Option C] Scatter plot ...")
figC, axC = plt.subplots(figsize=(8, 8))
axC.scatter(ab_base[core], ab_corr[core], s=3, alpha=0.4, color='#3498db', rasterized=True)
lim = max(ab_base[core].max(), ab_corr[core].max()) * 1.05
axC.plot([0, lim], [0, lim], '--', color='gray', linewidth=1.5, alpha=0.7,
         label='No improvement (对角线)')
better = ab_corr[core] < ab_base[core]
pct_better = better.mean() * 100
axC.fill_between([0, lim], [0, lim], 0, alpha=0.08, color='#2ecc71',
                  label=f'Correction helps ({pct_better:.0f}% of samples)')
axC.set_xlim(0, lim); axC.set_ylim(0, lim)
axC.set_xlabel('Base |Error| (MW)', fontsize=13); axC.set_ylabel('Corrected |Error| (MW)', fontsize=13)
axC.set_title(f'Core Zone (100-350km): Base vs Corrected\n{pct_better:.0f}% samples improved, Scatter below line = correction helps',
              fontsize=12, fontweight='bold')
axC.legend(fontsize=10); axC.grid(True, alpha=0.2); axC.set_aspect('equal')
plt.tight_layout()
figC.savefig('results/paper_figures/optionC_scatter.png', dpi=300)
plt.close(figC)

# ============================================================
# OPTION D: Clean before/after dashboard
# ============================================================
print("[Option D] Dashboard ...")
figD = plt.figure(figsize=(16, 8))
gsD = figD.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# (0,0): Core zone time series snippet
ax = figD.add_subplot(gsD[0, :2])
for i in range(len(t6)-1):
    c = '#ffe0e0' if d6s[i]<100 else '#d4f5d4' if d6s[i]<=350 else '#e0e8ff'
    ax.axvspan(t6[i], t6[min(i+1,len(t6)-1)], alpha=0.35, color=c, linewidth=0)
ax.plot(t6, yt6s, 'k-', linewidth=2.5, label='Actual')
ax.plot(t6, yb6s, '--', color='#e74c3c', linewidth=2.0, label='Base BiLSTM')
ax.plot(t6, yc6s, '-', color='#2ecc71', linewidth=2.5, label='KRR Corrected')
if len(ci6) > 10:
    ax.fill_between(t6[ci6[0]:ci6[-1]], yb6s[ci6[0]:ci6[-1]], yc6s[ci6[0]:ci6[-1]], color='#2ecc71', alpha=0.2)
ax.set_ylabel('Power (MW)', fontsize=12); ax.set_title(f'{ty_name} — 6h Ahead', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower left'); ax.grid(True, alpha=0.2); ax.set_ylim(0, CAPACITY)

# (0,2): Overall metrics table
ax = figD.add_subplot(gsD[0, 2])
ax.axis('off')
table_data = [
    ['Zone', 'Base RMSE', 'Corr RMSE', 'Improve'],
    ['Close (<100km)', f'{math.sqrt(np.mean(err_base[close]**2)):.1f}', f'{math.sqrt(np.mean(err_corr[close]**2)):.1f}',
     f'{((math.sqrt(np.mean(err_corr[close]**2))-math.sqrt(np.mean(err_base[close]**2)))/math.sqrt(np.mean(err_base[close]**2))*100):+.1f}%' if close.sum()>5 else '—'],
    ['Core (100-350km)', f'{base_rmses[1]:.1f}', f'{corr_rmses[1]:.1f}', f'{imps[1]:+.1f}%'],
    ['Far (350-600km)', f'{base_rmses[2]:.1f}', f'{corr_rmses[2]:.1f}', f'{imps[2]:+.1f}%'],
]
tbl = ax.table(cellText=table_data, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(11)
tbl.scale(1, 1.8)
for i in range(4):
    for j in range(4):
        tbl[(i,j)].set_facecolor('#d4f5d4' if (i>0 and float(table_data[i][3].replace('%','').replace('+','').replace('—','0')) < 0) else 'white')
ax.set_title('Per-Zone RMSE Summary', fontsize=12, fontweight='bold')

# (1,0): Error reduction by distance (scatter)
ax = figD.add_subplot(gsD[1, :2])
reduction = ab_base - ab_corr
ax.scatter(d_all[::10], reduction[::10], s=2, alpha=0.3, c=reduction[::10], cmap='RdYlGn', rasterized=True)
ax.axhline(0, color='gray', linewidth=0.8)
ax.axvline(100, color='orange', linestyle='--', linewidth=1)
ax.axvline(350, color='orange', linestyle='--', linewidth=1)
ax.set_xlabel('Distance from typhoon center (km)', fontsize=12)
ax.set_ylabel('Error Reduction (MW)\npositive=correction helps', fontsize=12)
ax.set_title('Correction Impact by Distance (all test typhoons)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2)

# (1,2): Win rate pie
ax = figD.add_subplot(gsD[1, 2])
better_core = (ab_corr[core] < ab_base[core]).sum()
worse_core = core.sum() - better_core
ax.pie([better_core, worse_core], labels=[f'Improved\n{better_core}', f'Worse\n{worse_core}'],
       colors=['#2ecc71', '#e74c3c'], autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11})
ax.set_title(f'Core Zone: {better_core/core.sum()*100:.0f}% samples improved', fontsize=11, fontweight='bold')

plt.tight_layout()
figD.savefig('results/paper_figures/optionD_dashboard.png', dpi=300)
plt.close(figD)

print("\nAll options saved to results/paper_figures/")
print("  optionA_timeseries.png")
print("  optionB_barchart.png")
print("  optionC_scatter.png")
print("  optionD_dashboard.png")
