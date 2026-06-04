#!/usr/bin/env python3
"""图1 NWP风速在不同台风距离区间的贡献 — PPT用"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans SC'
plt.rcParams['axes.unicode_minus'] = False

labels = ['<100 km', '100~200 km', '200~500 km', '>500 km']
pwr  = [38.69, 39.48, 47.70, 44.47]   # 仅功率
nwp  = [42.34, 51.03, 35.42, 32.59]   # 功率+NWP
N    = [108, 225, 527, 3204]           # 预测窗口数

x = np.arange(len(labels))
w = 0.32

fig, ax = plt.subplots(figsize=(9.5, 5.5))

b1 = ax.bar(x - w/2, pwr, w, label='仅历史功率 (M1)', color='#4472C4', edgecolor='white', linewidth=0.5)
b2 = ax.bar(x + w/2, nwp, w, label='历史功率 + NWP (M2)', color='#ED7D31', edgecolor='white', linewidth=0.5)

# 数值标签
for b in b1:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.2, f'{b.get_height():.1f}',
            ha='center', fontsize=11, fontweight='bold', color='#4472C4')
for b in b2:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.2, f'{b.get_height():.1f}',
            ha='center', fontsize=11, fontweight='bold', color='#ED7D31')

# 上方标注 NWP 贡献（恶化/改善）
for i in range(len(labels)):
    delta = (nwp[i] - pwr[i]) / pwr[i] * 100
    if delta > 0:
        txt = f'NWP恶化 {delta:+.1f}%'
        color = '#C00000'
    else:
        txt = f'NWP改善 {delta:+.1f}%'
        color = '#007030'
    ax.annotate(txt, xy=(x[i], max(pwr[i], nwp[i]) + 6),
                ha='center', fontsize=11, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color, lw=1.2))

# 样本数标注
for i in range(len(labels)):
    ax.text(x[i], 5, f'N={N[i]}', ha='center', fontsize=8, color='#888')

ax.set_ylabel('RMSE (MW)', fontsize=14)
ax.set_xlabel('台风中心至风电场平均距离', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
ax.set_ylim(0, max(nwp) + 15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_title('NWP风速在台风不同距离区间的预测贡献', fontsize=16, fontweight='bold', pad=12)

# 结论标注
ax.annotate('NWP 有效区', xy=(3.2, 33), fontsize=12, ha='center', color='#007030', fontweight='bold',
            bbox=dict(boxstyle='round', fc='#E8F5E9', ec='#007030', alpha=0.7))
ax.annotate('NWP 失效区\n(需参数模型补充)', xy=(0.6, 53), fontsize=11, ha='center', color='#C00000', fontweight='bold',
            bbox=dict(boxstyle='round', fc='#FFEBEE', ec='#C00000', alpha=0.7))

plt.tight_layout()
plt.savefig('docs/fig1_nwp_contribution_ppt.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: docs/fig1_nwp_contribution_ppt.png')
