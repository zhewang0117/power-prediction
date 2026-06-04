#!/usr/bin/env python3
"""图5 风机状态机及三层递进预测框架"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans SC'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

def box(x, y, w, h, txt, c='#D6E8F7', ec='#2E75B6', fs=8, fw='normal'):
    r = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.2', fc=c, ec=ec, lw=1.3)
    ax.add_patch(r)
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, fontweight=fw, linespacing=1.15, color='black')

def arr(x1, y1, x2, y2, c='#555', lw=1.2, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=c, lw=lw))

def arr_curved(x1, y1, x2, y2, rad=0.3, c='#555', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, connectionstyle=f'arc3,rad={rad}'))

# ===== 标题 =====
ax.text(9, 11.6, '图5  风机状态机及三层递进预测框架', ha='center', fontsize=14, fontweight='bold')

# ========== 左侧: 状态机 ==========
ax.text(4, 10.8, '风机运行状态机', ha='center', fontsize=11, fontweight='bold')

# 三个状态圆
r = 0.9
state_positions = {
    'N': (2.2, 7.5, 'Normal\n正常发电', '#E8F5E9', '#388E3C'),
    'C': (5.8, 7.5, 'Cutout\n切出停机', '#FFEBEE', '#C62828'),
    'R': (4.0, 5.2, 'Restarting\n再切入恢复', '#FFF8E1', '#F57F17'),
}

for state, (cx, cy, label, fc, ec) in state_positions.items():
    circle = Circle((cx, cy), r, facecolor=fc, edgecolor=ec, lw=2.5)
    ax.add_patch(circle)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=8, fontweight='bold')

# 转换箭头
# N -> C (上弧线)
pmid = ((2.2+5.8)/2 + 1.2, 9.0)
arr_curved(3.0, 8.2, 5.0, 8.2, rad=0.4, c='#C62828', lw=2.0)
ax.text(4.0, 9.5, 'V ≥ V_co', fontsize=8.5, ha='center', color='#C62828', fontweight='bold')

# C -> R (右侧下弧线)
arr_curved(5.8, 6.6, 4.9, 5.8, rad=-0.4, c='#F57F17', lw=2.0)
ax.text(6.2, 5.8, 'V < V_ci\n且 τ > T_dead', fontsize=7.5, ha='center', color='#F57F17', fontweight='bold')

# R -> N (左侧下弧线)
arr_curved(3.1, 5.6, 2.2, 6.6, rad=-0.4, c='#388E3C', lw=2.0)
ax.text(1.5, 5.6, 'P ≥ P_curve(V)', fontsize=8, ha='center', color='#388E3C', fontweight='bold')

# 虚分隔线
ax.axvline(x=9.5, ymin=0.04, ymax=0.96, color='#BBB', lw=1.5, ls='--')

# ========== 右侧: 三层框架 ==========
ax.text(13.5, 10.8, '三层递进预测框架', ha='center', fontsize=11, fontweight='bold')

# 第一层
ax.text(13.5, 9.8, '第一层：距离门控NWP风速修正', ha='center', fontsize=9, fontweight='bold', color='#1565C0',
        bbox=dict(boxstyle='round', fc='#E3F2FD', ec='#1565C0', alpha=0.5, pad=0.3))
box(13.5, 8.8, 4.8, 0.55, '台风数据 → 改进参数化模型 → V_YM\n→ 距离门控 w(d) → V_NWP*', '#E3F2FD', '#1565C0', 7.5)
arr(13.5, 8.52, 13.5, 7.85)

# 第二层
ax.text(13.5, 7.3, '第二层：GWO-ELM结构参数校正', ha='center', fontsize=9, fontweight='bold', color='#E65100',
        bbox=dict(boxstyle='round', fc='#FFF3E0', ec='#E65100', alpha=0.5, pad=0.3))
box(13.5, 6.3, 5.5, 0.7, 'BiLSTM → P_base(t)\n+ [Rmax_i, B_i, atten_w]\n→ GWO-ELM → P_physics(t)', '#FFF3E0', '#E65100', 7.5)
arr(13.5, 5.95, 13.5, 5.25)

# 第三层
ax.text(13.5, 4.7, '第三层：本文方法 控制策略约束', ha='center', fontsize=9, fontweight='bold', color='#2E7D32',
        bbox=dict(boxstyle='round', fc='#E8F5E9', ec='#2E7D32', alpha=0.5, pad=0.3))
box(13.5, 3.6, 5.8, 0.8, 'V_pred(t) + P_physics(t)\n→ 状态机 式(1)-(3)\nP_final(t) 由 式(10)-(12) 输出', '#E8F5E9', '#2E7D32', 7.5)

# 左侧箭头: 状态机 → 三层框架
arr_curved(9.0, 6.0, 10.2, 3.6, rad=0.3, c='#888', lw=1.5)
ax.text(9.5, 4.8, '控制\n逻辑\n嵌入', fontsize=7, ha='center', color='#666')

plt.tight_layout(pad=0.5)
plt.savefig('docs/fig5_state_machine.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: docs/fig5_state_machine.png')
