#!/usr/bin/env python3
"""
基于LOTO的台风分类分析: 哪些台风校正有效?
直接用台风编号匹配, 避免中文列名编码问题.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.size': 11,
})
OUT_DIR = 'results/paper_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# 1. LOTO结果
loto = pd.read_csv('results/loto_results_gwo_elm/loto_fold_paper_style.csv')

# 2. 台风数据
ty = pd.read_csv('data/typhoon/typhoon19-24complete_15min_improved.csv', encoding='utf-8-sig')
ty['target_time'] = pd.to_datetime(ty['target_time'], errors='coerce')

# 获取列名的索引位置 (避免直接使用中文列名)
col_names = list(ty.columns)
print(f"Total columns: {len(col_names)}")
for name in ['台风编号', '场站距离_km', 'rmax_i', 'b_i', 'atten_w', '台风等级', '移动速度(m/s)']:
    if name in col_names:
        print(f"  Found: {name} at index {col_names.index(name)}")

# 3. 对每个LOTO台风, 从原始数据提取特征统计
def get_typhoon_stats(event_id):
    """获取指定台风事件的气象特征统计"""
    mask = ty.iloc[:, 1] == event_id  # 台风编号列 (index 1)
    sub = ty[mask]
    if len(sub) == 0:
        return {}
    return {
        'min_dist_km': sub['场站距离_km'].min(),
        'mean_dist_km': sub['场站距离_km'].mean(),
        'mean_rmax': sub['rmax_i'].mean(),
        'mean_b': sub['b_i'].mean(),
        'mean_atten': sub['atten_w'].mean(),
        'max_intensity': sub['台风等级'].max(),
        'mean_speed': sub['移动速度(m/s)'].mean(),
    }

rows = []
for _, row in loto.iterrows():
    event_id = row['test_event']
    stats = get_typhoon_stats(event_id)
    rows.append({**row.to_dict(), **stats})

df = pd.DataFrame(rows)

# 4. 分类函数
def categorize(row):
    cats = {}
    d = row.get('min_dist_km', np.nan)
    if not pd.isna(d):
        if d < 100: cats['dist'] = 'Direct hit (<100km)'
        elif d < 250: cats['dist'] = 'Close (100-250km)'
        elif d < 500: cats['dist'] = 'Medium (250-500km)'
        else: cats['dist'] = 'Distant (>500km)'

    r = row.get('mean_rmax', np.nan)
    if not pd.isna(r):
        if r < 30: cats['rmax'] = 'Compact Rmax (<30km)'
        elif r < 55: cats['rmax'] = 'Medium Rmax (30-55km)'
        else: cats['rmax'] = 'Large Rmax (>55km)'

    i = row.get('max_intensity', np.nan)
    if not pd.isna(i):
        if i <= 5: cats['intensity'] = 'TS~TY'
        elif i <= 8: cats['intensity'] = 'STY'
        else: cats['intensity'] = 'Super TY'

    s = row.get('mean_speed', np.nan)
    if not pd.isna(s):
        if s < 5: cats['speed'] = 'Slow (<5m/s)'
        elif s < 8: cats['speed'] = 'Medium (5-8m/s)'
        else: cats['speed'] = 'Fast (>8m/s)'

    b = row.get('mean_b', np.nan)
    if not pd.isna(b):
        if b < 1.2: cats['b'] = 'Low B (<1.2)'
        elif b < 1.8: cats['b'] = 'Medium B (1.2-1.8)'
        else: cats['b'] = 'High B (>1.8)'

    return cats

for idx, row in df.iterrows():
    cats = categorize(row)
    for k, v in cats.items():
        df.loc[idx, f'cat_{k}'] = v

# 5. 统计
print(f"\nLOTO overall: {len(df)} typhoons, mean improvement = {df['rmse_improve_pct'].mean():.2f}%")
print(f"Win rate: {(df['rmse_improve_pct']>0).sum()}/{len(df)} = {(df['rmse_improve_pct']>0).mean()*100:.0f}%")

for cat_name in ['cat_dist', 'cat_rmax', 'cat_intensity', 'cat_speed', 'cat_b']:
    groups = df.groupby(cat_name)['rmse_improve_pct']
    print(f"\n--- {cat_name} ---")
    for name, grp in groups:
        win = (grp > 0).sum()
        print(f"  {name:25s} n={len(grp):2d} mean={grp.mean():+.2f}% win={win}/{len(grp)} ({win/len(grp)*100:.0f}%)")

# 6. 画图
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
ax_map = {
    'cat_dist': axes[0, 0], 'cat_rmax': axes[0, 1],
    'cat_intensity': axes[0, 2], 'cat_speed': axes[1, 0],
    'cat_b': axes[1, 1],
}
axes[1, 2].axis('off')

for cat_name, ax in ax_map.items():
    sub = df[[cat_name, 'rmse_improve_pct', 'test_event_name']].dropna(subset=[cat_name]).copy()
    if len(sub) == 0:
        ax.set_visible(False); continue
    cats_ordered = sub[cat_name].unique()
    for ci, cat_val in enumerate(cats_ordered):
        vals = sub[sub[cat_name] == cat_val]['rmse_improve_pct'].values
        jitter = np.random.default_rng(ci).uniform(-0.15, 0.15, len(vals))
        ax.scatter([ci + jitter[i] for i in range(len(vals))], vals,
                   s=60, alpha=0.6, color='#3498db', edgecolors='black', linewidth=0.5, zorder=3)
        ax.bar(ci, vals.mean(), width=0.4, color='#e74c3c' if vals.mean() < 0 else '#2ecc71',
               alpha=0.3, edgecolor='none')

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(range(len(cats_ordered)))
    ax.set_xticklabels(cats_ordered, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('RMSE Improvement (%)', fontsize=11)
    ax.set_title(cat_name.replace('cat_', 'By '), fontsize=12)
    ax.grid(True, alpha=0.2, axis='y')
    n_pos = (sub['rmse_improve_pct'] > 0).sum()
    ax.text(0.98, 0.95, f'Win: {n_pos}/{len(sub)} ({n_pos/len(sub)*100:.0f}%)',
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('LOTO: Correction Performance by Typhoon Category', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/loto_category_analysis.png')
plt.close(fig)
print(f"\n-> {OUT_DIR}/loto_category_analysis.png")

# 7. 最佳/最差台风
print("\n=== Best 5 ===")
print(df.nlargest(5, 'rmse_improve_pct')[
    ['test_event_name', 'rmse_improve_pct', 'min_dist_km', 'mean_rmax', 'max_intensity']
].to_string(index=False))

print("\n=== Worst 5 ===")
print(df.nsmallest(5, 'rmse_improve_pct')[
    ['test_event_name', 'rmse_improve_pct', 'min_dist_km', 'mean_rmax', 'max_intensity']
].to_string(index=False))
