#!/usr/bin/env python3
"""
台风分类分析: 校正器在不同类别台风上的表现

分类维度:
  1) 距离/Rmax 比值: 内侧vs外侧
  2) 绝对距离: 近距离vs远距离
  3) 台风强度: 台风vs强台风vs超强台风
  4) Rmax大小: 紧凑型vs中型vs大型

输出: 每类台风上的 base RMSE vs corrected RMSE
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, glob, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_smallnet_corrector import SmallCorrector, build_wide_dataset, HORIZON, SCALE_MAP

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.size': 11,
})
OUT_DIR = 'results/paper_figures'
os.makedirs(OUT_DIR, exist_ok=True)

PRED_GLOB = 'data/predictions/bilstm_pred_*.csv'
TY_CSV = 'data/typhoon/typhoon19-24complete_15min_improved.csv'

# 1. 加载数据集 (build_wide_dataset会按距离排序合并, 取最近的台风)
FEAT_ALL = ["rmax_i", "b_i", "atten_w", "场站距离_km", "hollandvg_az", "ym风速_i"]
X_ref, R_ref, is_ty_ref, times_ref = build_wide_dataset(PRED_GLOB, TY_CSV, FEAT_ALL)

n_total = len(X_ref)
n_train = int(n_total * 0.85)
test_ty_idx = np.where(is_ty_ref[n_train:])[0]

X_test = X_ref[n_train:][test_ty_idx]
R_test = R_ref[n_train:][test_ty_idx]
test_times = times_ref[n_train:][test_ty_idx]

# 2. 加载台风原始数据获取分类信息 (与build_wide_dataset一致的合并逻辑)
ty_raw = pd.read_csv(TY_CSV, encoding='utf-8-sig')
ty_raw['target_time'] = pd.to_datetime(ty_raw['target_time'], errors='coerce')
ty_nearest = ty_raw.sort_values(['target_time', '场站距离_km']).drop_duplicates('target_time', keep='first')

# 合并到测试集
test_df = pd.DataFrame({'target_time': pd.to_datetime(test_times)})
test_df = test_df.merge(
    ty_nearest[['target_time', '场站距离_km', '台风等级', '台风中文名称',
                '台风强度', '风速', '气压', '移动速度(m/s)']],
    on='target_time', how='left'
)

# 3. 反归一化特征: 距离直接从X_ref取 (与model使用的feature一致)
SCALE_VALS = [SCALE_MAP.get(f, 1.0) for f in FEAT_ALL]
rmax_vals = X_test[:, 0] * SCALE_VALS[0]  # km
b_vals = X_test[:, 1] * SCALE_VALS[1]
atten_vals = X_test[:, 2] * SCALE_VALS[2]
dist_vals_raw = X_test[:, 3] * SCALE_VALS[3]  # 从特征反推距离(km), 与model对齐

# 4. 计算分类标签
n_test = len(X_test)
categories = {
    'dist_r_max': np.full(n_test, 'unknown', dtype=object),
    'dist_bin': np.full(n_test, 'unknown', dtype=object),
    'intensity': np.full(n_test, 'unknown', dtype=object),
    'rmax_bin': np.full(n_test, 'unknown', dtype=object),
}

for i in range(n_test):
    d = dist_vals_raw[i] if not pd.isna(dist_vals_raw[i]) else 1000
    rm = rmax_vals[i]
    b = b_vals[i]

    # 1) distance/Rmax ratio
    if rm > 1 and d < 1000:
        ratio = d / rm
        if ratio < 0.8:
            categories['dist_r_max'][i] = 'Inside Rmax (core)'
        elif ratio < 1.5:
            categories['dist_r_max'][i] = 'Near Rmax (eyewall)'
        elif ratio < 3:
            categories['dist_r_max'][i] = 'Outer region'
        else:
            categories['dist_r_max'][i] = 'Far outer'
    else:
        categories['dist_r_max'][i] = 'Far / no data'

    # 2) distance
    if d < 150:
        categories['dist_bin'][i] = '<150km (close)'
    elif d < 350:
        categories['dist_bin'][i] = '150-350km'
    elif d < 600:
        categories['dist_bin'][i] = '350-600km'
    else:
        categories['dist_bin'][i] = '>600km (far)'

    # 3) Rmax size
    if rm > 1:
        if rm < 30:
            categories['rmax_bin'][i] = 'Compact (<30km)'
        elif rm < 60:
            categories['rmax_bin'][i] = 'Medium (30-60km)'
        else:
            categories['rmax_bin'][i] = 'Large (>60km)'
    else:
        categories['rmax_bin'][i] = 'Unknown'

# 台风等级: 从raw数据提取
for i in range(n_test):
    lv = test_df.iloc[i].get('台风等级', None)
    if pd.isna(lv) or lv == 0:
        categories['intensity'][i] = 'No typhoon label'
    elif lv <= 2:
        categories['intensity'][i] = 'TS/STS (tropical storm)'
    elif lv <= 5:
        categories['intensity'][i] = 'TY (typhoon)'
    elif lv <= 8:
        categories['intensity'][i] = 'STY (severe typhoon)'
    else:
        categories['intensity'][i] = 'Super TY'

# 5. 评估校正器在各分类上的表现
model = SmallCorrector(3)  # IMPR_STRUCT = rmax_i, b_i, atten_w
model.load_state_dict(torch.load('_smallnet_IMPR_STRUCT.pt', map_location='cpu', weights_only=True))
model.eval()

# P_base
pred_full = pd.concat([pd.read_csv(fp) for fp in sorted(glob.glob(PRED_GLOB))], ignore_index=True)
pred_cols = [f'y_pred_t+{i+1}_mw' for i in range(HORIZON)]
P_b = np.nan_to_num(pred_full[pred_cols].values.astype(np.float32), nan=0.0)
P_test = P_b[n_train:][test_ty_idx]
T_test = P_test + R_test

# 批处理校正
n_te = len(X_test)
print(f"Test samples: {n_te}")
parts = []
for h in range(HORIZON):
    ln = np.full((n_te, 1), h / 95.0, dtype=np.float32)
    parts.append(np.column_stack([X_test[:, :3], ln]))  # 只用IMPR_STRUCT
X_all = np.concatenate(parts, axis=0)

P_corr = P_test.copy()
model.eval()
with torch.no_grad():
    all_deltas = []
    bs = 4096
    for i in range(0, len(X_all), bs):
        inp_t = torch.tensor(X_all[i:i+bs], dtype=torch.float32)
        all_deltas.append(model(inp_t).cpu().numpy()[:, 0])
deltas = np.concatenate(all_deltas).reshape(HORIZON, n_te).T
P_corr += deltas

# 6. 在每个分类上计算RMSE
def eval_category(mask, name):
    if mask.sum() < 5:
        return None
    t = T_test[mask]
    b = P_test[mask]
    c = P_corr[mask]
    rmse_b = float(np.sqrt(np.mean((t - b) ** 2)))
    rmse_c = float(np.sqrt(np.mean((t - c) ** 2)))
    imp = (rmse_c - rmse_b) / rmse_b * 100
    return {'category': name, 'n': int(mask.sum()), 'base_rmse': rmse_b, 'corr_rmse': rmse_c, 'imp': imp}

all_results = []
for dim_name, cat_map in categories.items():
    unique_cats = np.unique(cat_map)
    for cat in unique_cats:
        mask = cat_map == cat
        r = eval_category(mask, f"[{dim_name}] {cat}")
        if r:
            all_results.append(r)

df_res = pd.DataFrame(all_results)
print(df_res.to_string(index=False))

# 对比 LOTO 结果: 逐台风表现
loto = pd.read_csv('results/loto_results_gwo_elm/loto_fold_paper_style.csv')
print(f"\nLOTO best 5: {loto.nlargest(5, 'rmse_improve_pct')[['test_event_name', 'rmse_improve_pct']].to_string(index=False)}")
print(f"LOTO worst 5: {loto.nsmallest(5, 'rmse_improve_pct')[['test_event_name', 'rmse_improve_pct']].to_string(index=False)}")

# ============ 画图 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, dim_name, title in zip(
    axes.flatten(),
    ['dist_r_max', 'dist_bin', 'rmax_bin', 'intensity'],
    ['By dist/Rmax ratio', 'By absolute distance', 'By Rmax size', 'By typhoon intensity']
):
    sub = df_res[df_res['category'].str.contains(f'\\[{dim_name}\\]')].copy()
    if len(sub) == 0:
        ax.set_visible(False)
        continue
    sub['short'] = sub['category'].str.replace(f'\\[{dim_name}\\] ', '', regex=True)

    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in sub['imp'].values]
    bars = ax.barh(range(len(sub)), sub['imp'].values, color=colors, edgecolor='white', height=0.6)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub['short'].values, fontsize=9)
    ax.set_xlabel('RMSE Improvement (%)', fontsize=11)
    ax.set_title(title, fontsize=12)

    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row['imp'] + 0.3 if row['imp'] >= 0 else row['imp'] - 0.3, i,
                f'{row["imp"]:+.1f}% (n={row["n"]})', va='center',
                fontsize=8, ha='left' if row['imp'] >= 0 else 'right',
                color='#27ae60' if row['imp'] < 0 else '#c0392b')
    ax.grid(True, alpha=0.2, axis='x')

plt.suptitle('GWO-ELM Corrector Performance by Typhoon Category', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/typhoon_category_analysis.png')
plt.close(fig)
print(f"\n-> {OUT_DIR}/typhoon_category_analysis.png")

# ============ 具体台风事件表 ============
print("\n=== Per-typhoon test set summary (temporal split) ===")
# 看看测试集包含了哪些台风
test_ty_names = test_df['台风中文名称'].dropna().unique()
print(f"Typhoons in test set: {test_ty_names}")
