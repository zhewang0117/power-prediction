#!/usr/bin/env python3
"""论文方法论框架图"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Rectangle
import numpy as np

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(22, 16))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# ===== Color scheme =====
C_INPUT = '#3498db'      # 输入数据
C_FEAT = '#f39c12'       # 特征工程
C_MODEL = '#2ecc71'      # 模型
C_OUTPUT = '#9b59b6'     # 输出
C_EVAL = '#e74c3c'       # 评估
C_ARROW = '#2c3e50'      # 箭头
C_BG_BOX = '#ecf0f1'     # 背景框

def draw_box(ax, x, y, w, h, color, text, fontsize=11, fontweight='bold', text_color='white', edgewidth=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3', facecolor=color,
                          edgecolor='#2c3e50', linewidth=edgewidth, alpha=0.92)
    ax.add_patch(box)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.8 - i * 1.5, line, ha='center', va='top',
                fontsize=fontsize, fontweight=fontweight, color=text_color)

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2.5, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, connectionstyle='arc3,rad=0'))

def draw_sub_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

def draw_section_bg(ax, x, y, w, h, label, color):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5', facecolor=color,
                           edgecolor='none', alpha=0.12)
    ax.add_patch(rect)
    ax.text(x + 0.5, y + h - 0.5, label, fontsize=12, fontweight='bold', color='#2c3e50', va='top')

# ============================================================
# SECTION 1: 数据输入 (左列)
# ============================================================
draw_section_bg(ax, 1, 62, 24, 36, '数据输入', C_INPUT)

draw_box(ax, 2, 88, 22, 7, C_INPUT,
         'NWP数值天气预报\n(ECMWF, 15min分辨率)\n风速·风向·气温·气压', fontsize=9, text_color='white', edgewidth=1.5)

draw_box(ax, 2, 75, 22, 7, C_INPUT,
         'CMA台风最佳路径数据\n(6h间隔)\n经度·纬度·中心气压·最大风速', fontsize=9, text_color='white', edgewidth=1.5)

draw_box(ax, 2, 64, 22, 5, C_INPUT,
         '风电场历史功率\n(15min, SCADA)', fontsize=9, text_color='white', edgewidth=1.5)

# ============================================================
# SECTION 2: 特征工程 (中左)
# ============================================================
draw_section_bg(ax, 27, 38, 28, 60, '台风参数模型特征工程', C_FEAT)

# 参数模型盒子
draw_box(ax, 28, 82, 26, 13, C_FEAT,
         '改进台风参数模型\n─────────────────\nHolland气压场 + Batts梯度风\n闫孟边界层风速模型\nWilloughby Rmax · Powell B · Ning衰减',
         fontsize=9, text_color='white', edgewidth=1.5)

# 分支特征组
y_feat_start = 72
feat_groups = [
    ('结构参数', 'Rmax_km  最大风速半径\nB_param  Holland B参数\natten_w  移速衰减系数', '#e67e22'),
    ('局地风场', 'V_yanmeng  闫孟模型风速\nVtan  切向风分量\nVrad  径向风分量', '#27ae60'),
    ('几何位置', 'dist_D_km  风机-台风距离\nD/Rmax  归一化距离\ncos/sin 相对方位角', '#2980b9'),
]

for i, (title, desc, fcolor) in enumerate(feat_groups):
    xb = 28.5 + i * 8.8
    draw_box(ax, xb, 59, 8.2, 12, fcolor,
             f'{title}\n──────\n{desc}', fontsize=7.5, text_color='white', edgewidth=1)

# ============================================================
# SECTION 3: 校正模型 (中右)
# ============================================================
draw_section_bg(ax, 57, 38, 22, 60, '偏差校正模型', C_MODEL)

# KRR模型
draw_box(ax, 58, 76, 20, 10, C_MODEL,
         'Kernel Ridge Regression (KRR)\n─────────────────────\nRBF核函数  非线性学习\n特征与误差的隐式映射\n\n核函数: K(xi,xj) = exp(-gamma * d2)',
         fontsize=9, text_color='white', edgewidth=1.5)

# 保守融合
draw_box(ax, 58, 62, 20, 6, '#1abc9c',
         '保守融合策略\nP_corrected = P_base + alpha * delta\n(alpha = 0.6, 验证集选定)',
         fontsize=9, text_color='white', edgewidth=1.5)

# 训练策略
draw_box(ax, 58, 50, 20, 7, '#16a085',
         '训练策略\n──────\n23台风训练 + 1台风验证\n-> 验证集选alpha -> 9台风测试',
         fontsize=8.5, text_color='white', edgewidth=1.5)

# ============================================================
# SECTION 4: 输出与评估 (右列)
# ============================================================
draw_section_bg(ax, 81, 38, 18, 60, '输出与评估', C_OUTPUT)

# 输出
draw_box(ax, 82, 88, 16, 7, C_OUTPUT,
         '校正后功率预测\nP_corrected(t+1..t+96)\n96步超前 (24h)',
         fontsize=9, text_color='white', edgewidth=1.5)

# 评估指标
draw_box(ax, 82, 73, 16, 12, C_EVAL,
         '多维评估指标\n────────\n① 全局RMSE/MAE\n② 台风窗口RMSE\n③ 核心区RMSE (100-350km)\n④ 功率骤变RMSE\n⑤ 校正方向准确率',
         fontsize=8.5, text_color='white', edgewidth=1.5)

# 消融实验
draw_box(ax, 82, 54, 16, 16, '#e74c3c',
         '消融实验设计\n────────\nG0: BiLSTM基线\nG1: 纯预测特征\nG2: +参数风场\nG3: +结构参数\nG4: +几何位置\nG5: 全部物理特征\nG6: 弱特征反证',
         fontsize=8, text_color='white', edgewidth=1.5)

# ============================================================
# 箭头连接
# ============================================================

# 数据 → 特征
draw_arrow(ax, 24, 91, 27.5, 88.5)
draw_arrow(ax, 24, 78, 27.5, 88)
draw_arrow(ax, 24, 66, 27.5, 87.5)

# 特征内部箭头
draw_arrow(ax, 41, 84.5, 57.5, 81, C_FEAT, 2.5)
draw_arrow(ax, 41, 84, 31, 72, C_FEAT, 1.5)
draw_arrow(ax, 31, 71, 36.5, 59.5, C_FEAT, 1.5)
draw_arrow(ax, 39.5, 71, 40.5, 59.5, C_FEAT, 1.5)
draw_arrow(ax, 48.5, 71, 44, 59.5, C_FEAT, 1.5)

# 特征 → KRR
for xi in [33, 41.5, 50]:
    draw_arrow(ax, xi, 59, 58.5, 80.5, C_FEAT, 1.5)

# KRR → 融合 → 输出
draw_arrow(ax, 68, 76, 68, 68.5, C_MODEL, 2.5)
draw_arrow(ax, 68, 62, 82, 90, C_MODEL, 2.5)

# 输出 → 评估
draw_arrow(ax, 90, 88, 90, 85.5, C_OUTPUT, 2.5)

# ============================================================
# 阶段标注 (顶部时间线)
# ============================================================
stages = [
    (1, '阶段一', '数据准备', C_INPUT),
    (28.5, '阶段二', '台风参数模型特征工程', C_FEAT),
    (58, '阶段三', 'KRR-RBF偏差校正', C_MODEL),
    (83, '阶段四', '输出与多维评估', C_OUTPUT),
]

for x_start, title, desc, color in stages:
    ax.text(x_start + 0.5, 99, f'{title}\n{desc}', fontsize=10, fontweight='bold', color=color,
            ha='center', va='top')
    ax.plot([x_start, x_start + 11], [98.5, 98.5], '-', color=color, linewidth=4, alpha=0.5)

# ============================================================
# BiLSTM 底座标注
# ============================================================
draw_box(ax, 40, 40, 38, 4, '#34495e',
         '基于 BiLSTM 的风电功率基础预测模型 (96步超前, 双向LSTM, 输入NWP+历史功率)',
         fontsize=9, text_color='white', edgewidth=1.5)

draw_arrow(ax, 59, 48, 67, 50, '#34495e', 2)
draw_arrow(ax, 59, 44, 67, 44, '#34495e', 2)

# ============================================================
# 标题
# ============================================================
ax.text(50, 101, '融合台风参数模型信息的KRR风电功率偏差校正方法 — 整体框架',
        fontsize=18, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2c3e50', linewidth=2))

plt.tight_layout()
fig.savefig('results/paper_figures/methodology_framework.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved: results/paper_figures/methodology_framework.png')
