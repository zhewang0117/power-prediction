#!/usr/bin/env python3
"""图1 实测NWP风-功相关性 vs 台风距离 — PPT用"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans SC'
plt.rcParams['axes.unicode_minus'] = False

labels = ['非台风期', '<100 km', '100~200 km', '200~500 km', '>500 km']
r_vals = [0.774, 0.344, 0.455, 0.797, 0.807]
r2_vals = [0.60, 0.12, 0.21, 0.64, 0.65]
N_vals = [191897, 53, 302, 2291, 15886]
colors = ['#4472C4', '#C00000', '#C00000', '#007030', '#007030']

fig, ax1 = plt.subplots(figsize=(9.5, 5.5))

x = np.arange(len(labels))
bars = ax1.bar(x, r_vals, 0.5, color=colors, edgecolor='white', linewidth=0.5)

# r值标注
for i, (bar, r) in enumerate(zip(bars, r_vals)):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f'r = {r:.3f}', ha='center', fontsize=12, fontweight='bold', color=colors[i])

# R2标注在柱内
for i, (bar, r2) in enumerate(zip(bars, r2_vals)):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
             f'R2={r2:.2f}', ha='center', fontsize=11, fontweight='bold', color='white')

# 样本数
for i, n in enumerate(N_vals):
    ax1.text(i, 0.05, f'N={n}', ha='center', fontsize=8, color='#666')

# 失效区/有效区标注
ax1.axvspan(-0.4, 2.4, alpha=0.08, color='red')
ax1.axvspan(2.4, 4.4, alpha=0.08, color='green')
ax1.annotate('NWP 失效区', xy=(1.0, 0.88), fontsize=13, ha='center', color='#C00000', fontweight='bold')
ax1.annotate('NWP 正常区', xy=(3.5, 0.88), fontsize=13, ha='center', color='#007030', fontweight='bold')

ax1.set_ylabel('Pearson r (NWP风速 vs 实测功率)', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12)
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', alpha=0.2, linestyle='--')
ax1.set_title('实测数据中 NWP 风速与功率的相关性随台风距离的变化', fontsize=15, fontweight='bold', pad=12)

# 结论
ax1.text(2.0, 0.95, 'R2从0.60→0.12，NWP解释力在台风内核区崩溃',
         ha='center', fontsize=11, color='#C00000', fontweight='bold',
         bbox=dict(boxstyle='round', fc='#FFF5F5', ec='#C00000', alpha=0.8))

plt.tight_layout()
plt.savefig('docs/fig1_nwp_correlation_ppt.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: docs/fig1_nwp_correlation_ppt.png')
