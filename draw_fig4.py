#!/usr/bin/env python3
"""图4 软切出功率曲线示意"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans SC'
plt.rcParams['axes.unicode_minus'] = False

V_co, P_rated, alpha = 25.0, 200.0, 0.5
V_ci = 20.0
V = np.linspace(0, 40, 400)

# 标准功率曲线 (硬切出)
P_std = np.zeros_like(V)
cutin, rated_speed = 3, 14
for i, v in enumerate(V):
    if v < cutin: P_std[i] = 0
    elif v < rated_speed: P_std[i] = P_rated * ((v - cutin) / (rated_speed - cutin))**3
    elif v < V_co: P_std[i] = P_rated
    else: P_std[i] = 0

# 软切出曲线
P_soft = np.zeros_like(V)
for i, v in enumerate(V):
    if v < cutin: P_soft[i] = 0
    elif v < rated_speed: P_soft[i] = P_rated * ((v - cutin) / (rated_speed - cutin))**3
    elif v < V_co: P_soft[i] = P_rated
    else: P_soft[i] = P_rated * (1 - np.tanh(alpha * (v - V_co))) / 2

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(V, P_std, 'b-', linewidth=2.5, label='硬切出 (标准功率曲线)')
ax.plot(V, P_soft, 'r-', linewidth=2.5, label='软切出 (双曲正切模型)')

# 标注切出风速
ax.axvline(x=V_co, color='#555', linewidth=1.2, linestyle='--', alpha=0.7)
ax.annotate('V_co = 25 m/s\n(切出风速)', xy=(V_co, 20), fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='#555'))
ax.annotate('', xy=(V_co+1.5, 50), xytext=(V_co, 20),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1))

# 标注再切入风速
ax.axvline(x=V_ci, color='#888', linewidth=1, linestyle=':', alpha=0.6)
ax.annotate('V_ci = 20 m/s\n(再切入风速)', xy=(V_ci, 140), fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='#888'))

# 标注衰减速率
x_alpha = V_co + 2.5
y_alpha = P_soft[int((V_co+2.5)*10)]
ax.annotate('衰减速率 α = 0.5\nα 越大, 曲线越陡', xy=(x_alpha, y_alpha),
            xytext=(x_alpha+6, y_alpha+40), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', alpha=0.9, ec='#C00000'))

# 标注死区 (切出-再切入之间的区域)
ax.annotate('', xy=(V_ci-1, 10), xytext=(V_co+1, 10),
            arrowprops=dict(arrowstyle='<->', color='#E65100', lw=1.5))
ax.text((V_ci+V_co)/2, 18, '死区区间\nT_dead控制\n再切入延迟', fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3E0', alpha=0.9, ec='#E65100'))

# 坐标轴
ax.set_xlabel('轮毂高度风速 (m/s)', fontsize=13)
ax.set_ylabel('功率 (MW)', fontsize=13)
ax.set_xlim(0, 40)
ax.set_ylim(-5, 220)
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.25)
ax.set_title('图4 风电机组软切出功率曲线示意', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('docs/fig4_soft_cutout.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: docs/fig4_soft_cutout.png')
