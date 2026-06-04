#!/usr/bin/env python3
"""Draw Figure: Three-Layer Progressive Embedding Framework"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(22, 10))
ax.set_xlim(0, 22)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

C1 = '#2C6B9E'   # work1 blue
C2 = '#D4402E'   # work2 red
C3 = '#2E8B57'   # work3 green
C4 = '#6B3FA0'   # output purple

def rounded_box(ax, x, y, w, h, color, label, sub_labels=None):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                                    facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.95)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.32, label, ha='center', va='top',
            fontsize=10, fontweight='bold', color='white')
    if sub_labels:
        for i, sl in enumerate(sub_labels):
            ax.text(x + w/2, y + h - 0.72 - i*0.35, sl, ha='center', va='top',
                    fontsize=8, color='white', alpha=0.85)

def data_box(ax, x, y, w, h, label, color='#D6E4F0'):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                    facecolor=color, edgecolor='#999', linewidth=1, ls='--')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=9, color='#555')

def draw_arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2.5))
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 + 0.2
        ax.text(mx, my, label, ha='center', va='bottom', fontsize=9, color='#555', style='italic')

def sm_arrow(ax, x1, y1, x2, y2, label):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='white', lw=2, connectionstyle='arc3,rad=0.3'))
    ax.text((x1+x2)/2, (y1+y2)/2 + 0.35, label, ha='center', va='bottom',
            fontsize=7.5, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.05', facecolor=C3, edgecolor='none'))

# ==================== DATA SOURCES ====================
data_box(ax, 0.3, 0.3, 2.5, 1.2, 'NWP Wind\nV_NWP')
data_box(ax, 3.3, 0.3, 2.5, 1.2, 'Typhoon Best Track\n(lat, lon, Pc, Vmax)')
data_box(ax, 6.3, 0.3, 2.5, 1.2, 'Historical Power\nP_true')
data_box(ax, 9.3, 0.3, 2.5, 1.2, 'Structure Params\nRmax, B, atten')

# ==================== WORK 1 ====================
rounded_box(ax, 0.5, 2.5, 3.5, 2.8, C1,
    'Work 1: Distance-Gated NWP Correction',
    ['Parametric Model: V_YM (Holland-YanMeng)',
     'Weight: w(d) = exp(-d/d0)',
     'V_NWP* = V_NWP - w * max(0, V_NWP - V_YM)',
     'Output: V_NWP* (corrected wind)'])

draw_arrow(ax, 1.65, 1.5, 1.65, 2.5, 'V_NWP')
draw_arrow(ax, 4.55, 0.9, 2.25, 2.5, 'best track')

# ==================== WORK 2 ====================
rounded_box(ax, 6.0, 2.5, 4.0, 2.8, C2,
    'Work 2: GWO-ELM Bias Correction',
    ['Pre-trained BiLSTM -> P_base',
     'Input: [P_base, lead_norm, Rmax, B, atten]',
     'ELM: beta = H^+ Y (analytical solution)',
     'GWO: optimize {W, b} (n_wolves=10, iter=10)',
     'Output: P_physics = P_base + delta_pred'])

draw_arrow(ax, 4.0, 3.9, 6.0, 3.9, 'P_base')
draw_arrow(ax, 2.25, 5.3, 6.0, 5.3, 'V_NWP*')
draw_arrow(ax, 4.55, 0.9, 10.75, 3.9, 'Rmax, B, atten')

# ==================== WORK 3 ====================
rect3 = mpatches.FancyBboxPatch((0.5, 6.0), 9.5, 3.5, boxstyle='round,pad=0.2',
                                 facecolor=C3, edgecolor='#333', linewidth=2.5, alpha=0.92)
ax.add_patch(rect3)
ax.text(5.25, 9.25, 'Work 3: Control Strategy Constraint', ha='center', va='top',
        fontsize=12, fontweight='bold', color='white')

# state machine sub-boxes
sm_y = 6.4
sm_h = 1.5

# Normal
rect_n = mpatches.FancyBboxPatch((0.8, sm_y), 2.8, sm_h, boxstyle='round,pad=0.1',
                                   facecolor='#3CB371', edgecolor='white', linewidth=1.5)
ax.add_patch(rect_n)
ax.text(2.2, sm_y + sm_h/2 + 0.15, 'NORMAL', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(2.2, sm_y + sm_h/2 - 0.3, 'P_final = P_physics', ha='center', va='center',
        fontsize=8.5, color='white', alpha=0.9)

# Cutout
rect_c = mpatches.FancyBboxPatch((4.2, sm_y), 2.8, sm_h, boxstyle='round,pad=0.1',
                                   facecolor='#DC143C', edgecolor='white', linewidth=1.5)
ax.add_patch(rect_c)
ax.text(5.6, sm_y + sm_h/2 + 0.15, 'CUTOUT', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(5.6, sm_y + sm_h/2 - 0.3, 'P_final = P_soft(V_pred)', ha='center', va='center',
        fontsize=8.5, color='white', alpha=0.9)

# Restarting
rect_r = mpatches.FancyBboxPatch((7.6, sm_y), 2.2, sm_h, boxstyle='round,pad=0.1',
                                   facecolor='#FF8C00', edgecolor='white', linewidth=1.5)
ax.add_patch(rect_r)
ax.text(8.7, sm_y + sm_h/2 + 0.15, 'RESTART', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(8.7, sm_y + sm_h/2 - 0.25, 'P = min(P(t-1)', ha='center', va='center',
        fontsize=8.5, color='white', alpha=0.9)
ax.text(8.7, sm_y + sm_h/2 - 0.55, '+ R*dt, P_phy)', ha='center', va='center',
        fontsize=8.5, color='white', alpha=0.9)

# state transitions
sm_arrow(ax, 3.6, sm_y + sm_h - 0.3, 4.2, sm_y + sm_h - 0.3, 'V >= V_co')

ax.annotate('', xy=(7.1, sm_y + 0.3), xytext=(5.6, sm_y + 0.3),
            arrowprops=dict(arrowstyle='->', color='white', lw=2, connectionstyle='arc3,rad=-0.3'))
ax.text(6.8, sm_y + 0.0, 'V < Vci & tau > Tdead', ha='center', va='top',
        fontsize=7.5, color='white', fontweight='bold')

ax.annotate('', xy=(7.6, sm_y + sm_h - 0.3), xytext=(6.0, sm_y + sm_h - 0.3),
            arrowprops=dict(arrowstyle='->', color='white', lw=2, connectionstyle='arc3,rad=-0.3'))
ax.text(6.5, sm_y + sm_h + 0.1, 'P >= P_curve(V)', ha='center', va='bottom',
        fontsize=7.5, color='white', fontweight='bold')

# soft cutout formula
ax.text(5.25, 6.15, 'P_soft(V) = P_rated * [1 - tanh(alpha(V-V_co))] / 2',
        ha='center', va='center', fontsize=8.5, color='white', alpha=0.8, style='italic')

# arrows to work3
draw_arrow(ax, 8.0, 5.3, 8.0, 6.0, 'P_physics')
draw_arrow(ax, 2.25, 5.3, 2.25, 6.0)

ax.text(2.25, 5.65, 'V_pred = V_NWP*', ha='center', va='center',
        fontsize=8, color='#555', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none'))
ax.text(5.0, 5.65, 'P_physics', ha='center', va='center',
        fontsize=8, color='#555', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none'))

# ==================== OUTPUT ====================
rect_out = mpatches.FancyBboxPatch((12.0, 7.0), 3.0, 1.5, boxstyle='round,pad=0.2',
                                     facecolor=C4, edgecolor='#333', linewidth=2.5)
ax.add_patch(rect_out)
ax.text(13.5, 7.75, 'P_final(t)', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')

draw_arrow(ax, 10.0, 7.75, 12.0, 7.75)

# ==================== ANNOTATION BRACKETS ====================
ax.annotate('', xy=(0.5, 3.6), xytext=(4.0, 3.6),
            arrowprops=dict(arrowstyle='<->', color=C1, lw=2, linestyle=':'))
ax.text(2.25, 3.8, 'Input-side correction', ha='center', va='bottom',
        fontsize=8, color=C1, fontweight='bold')

ax.annotate('', xy=(6.0, 3.6), xytext=(10.0, 3.6),
            arrowprops=dict(arrowstyle='<->', color=C2, lw=2, linestyle=':'))
ax.text(8.0, 3.8, 'Output-side correction', ha='center', va='bottom',
        fontsize=8, color=C2, fontweight='bold')

# ==================== LEGEND ====================
legend_items = [
    mpatches.Patch(color=C1, alpha=0.8, label='Work 1: Wind correction'),
    mpatches.Patch(color=C2, alpha=0.8, label='Work 2: Bias correction'),
    mpatches.Patch(color=C3, alpha=0.8, label='Work 3: Control constraint'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=9,
          framealpha=0.9, edgecolor='#ccc')

ax.set_title('Three-Layer Progressive Embedding Framework',
             fontsize=15, fontweight='bold', pad=8, color='#333')

plt.tight_layout()
out_path = 'docs/fig_three_layer_framework.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved to {out_path}")
