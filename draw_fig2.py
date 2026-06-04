#!/usr/bin/env python3
"""图2 距离门控融合预测框架 — 全部硬编码坐标, 统一宽度2.2"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'Noto Sans SC'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(9, 10))
ax.set_xlim(0, 9)
ax.set_ylim(4.5, 22.5)
ax.axis('off')

def box(x, y, w, h, txt, c='#D6E8F7', ec='#2E75B6', fs=8, fw='bold', color='black'):
    r = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.2', fc=c, ec=ec, lw=1.3)
    ax.add_patch(r)
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, fontweight=fw, linespacing=1.15, color=color)

def arr(x1,y1,x2,y2,c='black',lw=1.1):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=c,lw=lw))

def arr_c(x1,y1,x2,y2,rad=0.3,c='black',lw=1.1):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color=c,lw=lw,connectionstyle=f'arc3,rad={rad}'))

W = 2.2         # 中轴模块宽度
SW = 1.5        # 侧面模块宽度
CX = 4.5        # 中轴 x

# ===== 标题 =====
ax.text(CX, 22.1, '图2  距离门控融合风功率预测整体框架', ha='center', fontsize=11, fontweight='bold', color='black')

# ===== 上部模块标签 =====
ax.text(CX, 21.2, '距离门控融合模块', ha='center', fontsize=9, fontweight='bold', color='black',
        bbox=dict(boxstyle='round', fc='#D6E8F7', ec='#2E75B6', alpha=0.6, pad=0.3))

# 以下垂直坐标恢复为拉伸版的间距
box(CX, 20.0, W, 0.65, '台风最佳路径数据\n(中心气压、风速、移速、经纬度)', '#FFF3E0', '#F57C00', 7.5)
arr(CX, 19.68, CX, 19.08)

box(CX, 18.7, W, 0.60, '改进 Holland-YanMeng\n参数化风场模型', '#D6E8F7', '#2E75B6', 8)
arr(CX, 18.40, CX, 17.85)

box(CX, 17.5, W, 0.50, '地表风速 V_YM 及结构参数', '#E8F5E9', '#388E3C', 8)
arr(CX, 17.25, CX, 16.70)

box(CX, 16.35, W, 0.50, '台风-风电场距离 d (Haversine 公式)', '#FFF8E1', '#F9A825', 7.5)
arr(CX, 16.10, CX, 15.55)

box(CX, 15.20, W, 0.50, '门控函数 w(d)=exp(−d/d0)', '#FFF8E1', '#F9A825', 8)

# NWP 左侧汇入
box(1.0, 14.30, SW, 0.45, '原始 NWP 风速\nV_NWP', '#ECEFF1', '#607D8B', 7)
ax.text(1.0, 14.85, 'NWP输入', ha='center', fontsize=7, color='black')
arr(1.75, 14.30, CX-W/2+0.1, 14.30)

arr(CX, 14.95, CX, 14.45)
box(CX, 14.05, W, 0.60, '单向残差修正\nV_NWP* = V_NWP\n− w(d)·max(0, V_NWP−V_YM)', '#C8E6C9', '#2E7D32', 7.5)

# ===== 分隔线 =====
ax.axhline(y=13.15, xmin=0.04, xmax=0.96, color='#999', lw=1, ls='--')

# ===== 下部模块标签 =====
ax.text(CX, 12.75, 'BiLSTM 预测模块', ha='center', fontsize=9, fontweight='bold', color='black',
        bbox=dict(boxstyle='round', fc='#D6E8F7', ec='#2E75B6', alpha=0.6, pad=0.3))

box(CX, 11.50, W, 0.65, '输入特征\n历史7天功率\n+ V_NWP* + NWP风向', '#FFF3E0', '#F57C00', 7.5)
arr(CX, 11.18, CX, 10.63)

box(CX, 10.30, W, 0.50, 'BiLSTM 编码器\n(2层双向, 隐层96)', '#D6E8F7', '#2E75B6', 8)
arr(CX, 10.05, CX, 9.50)

box(CX, 9.20, W, 0.40, '上下文向量', '#E8F5E9', '#388E3C', 8)

# 未来 NWP 右侧汇入
box(7.3, 9.60, SW, 0.45, '未来 24h\nNWP 风向', '#ECEFF1', '#607D8B', 7)
ax.text(7.3, 10.15, 'NWP输入', ha='center', fontsize=7, color='black')
arr(7.3-SW/2+0.05, 9.60, CX+W/2-0.05, 9.20)

arr(CX, 9.00, CX, 8.40)
box(CX, 8.10, W, 0.55, '外源特征融合\n+ MLP 解码器', '#D6E8F7', '#2E75B6', 8)
arr(CX, 7.82, CX, 7.27)

box(CX, 7.00, W, 0.55, '输出\n未来 24h 功率预测\n(96步 × 15分钟)', '#C8E6C9', '#2E7D32', 7.5)

# ===== 跨模块红色箭头 =====
arr_c(0.55, 14.05, 0.55, 11.50, rad=-0.55, c='#C00000', lw=2.0)
ax.text(0.18, 12.78, '替换\nNWP\n风速\n通道', fontsize=6, color='#C00000', fontweight='bold', ha='center')

plt.tight_layout(pad=0.3)
plt.savefig('docs/fig2_framework.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: docs/fig2_framework.png')
