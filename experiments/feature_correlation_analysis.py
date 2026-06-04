#!/usr/bin/env python3
"""
全面特征相关性分析: 台风期间BiLSTM预测误差 vs 所有候选台风特征

特征类别:
  几何: 距离D, D/Rmax, 相对方位角(sin/cos), 象限, ΔD
  强度: 压差(Pn-Pc), 最大风速Vmax
  尺度: Rmax
  运动: Vmove
  参数模型: 局地风速Vpm, 局地风速变化率ΔVpm, 切向/径向风速
  结构: D/Rmax-1, exp(-D/Rmax)
  差异: NWP风速-参数风速差, 风向差, 变化率差

输出:
  - 每特征与时序误差的相关性
  - 分类(几何/强度/结构/差异)热力图
  - Top-N 特征推荐列表
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, glob
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.size': 9,
})
OUT_DIR = 'results/paper_figures'
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON = 96
PRED_GLOB = 'data/predictions/bilstm_pred_*.csv'
TY_CSV = 'data/typhoon/typhoon19-24complete_15min_improved.csv'

# ============ 1. 加载数据 ============
print("[1] Loading data...")

# 预测数据
pred_dfs = []
for fp in sorted(glob.glob(PRED_GLOB)):
    pred_dfs.append(pd.read_csv(fp))
pred = pd.concat(pred_dfs, ignore_index=True)
pred['target_start_time'] = pd.to_datetime(pred['target_start_time'])

# 台风数据
ty = pd.read_csv(TY_CSV, encoding='utf-8-sig')
ty['target_time'] = pd.to_datetime(ty['target_time'], errors='coerce')

# 合并: 取最接近的台风记录 (与build_wide_dataset一致)
ty_sorted = ty.sort_values(['target_time', '场站距离_km']).drop_duplicates('target_time', keep='first')
merged = pred.merge(ty_sorted, left_on='target_start_time', right_on='target_time', how='inner')
print(f"  Merged typhoon samples: {len(merged)}")

# 区分台风/非台风
merged['is_ty'] = merged['target_time'].notna().astype(int)
ty_samples = merged[merged['is_ty'] == 1].copy()
print(f"  Typhoon period samples: {len(ty_samples)}")

# ============ 2. 计算误差(残差) ============
print("[2] Computing prediction errors...")

true_cols = [f'y_true_t+{i+1}_mw' for i in range(HORIZON)]
pred_cols = [f'y_pred_t+{i+1}_mw' for i in range(HORIZON)]

# 取几个代表性horizon: 1h, 4h, 8h, 12h, 24h
lead_steps = [3, 15, 31, 47, 95]   # 1h, 4h, 8h, 12h, 24h (15min steps)
lead_labels = ['1h', '4h', '8h', '12h', '24h']

errors = {}
for h, label in zip(lead_steps, lead_labels):
    truth = ty_samples[true_cols[h]].values.astype(float)
    preds = ty_samples[pred_cols[h]].values.astype(float)
    errors[label] = preds - truth  # positive = over-prediction
    # Also compute absolute error
    errors[f'{label}_abs'] = np.abs(preds - truth)

# Overall MAE across all horizons
all_truth = ty_samples[true_cols].values.astype(float)
all_preds = ty_samples[pred_cols].values.astype(float)
overall_abs_err = np.mean(np.abs(all_preds - all_truth), axis=1)

# ============ 3. 构建特征 ============
print("[3] Building feature matrix...")

CAPACITY = 200.0
PN = 1013.3  # ambient pressure (hPa)
SCALE_MAP = {'场站距离_km': 1000.0, 'hollandvg_az': 50.0, 'ym风速_i': 50.0,
             'rmax_i': 100.0, 'b_i': 3.0, 'atten_w': 1.0}

def safe_val(series, default=0.0):
    return series.fillna(default).values.astype(float)

features = {}

# --- 几何特征 ---
D = safe_val(ty_samples['场站距离_km'])  # distance in km
features['距离 D (km)'] = D
features['D/Rmax'] = D / (safe_val(ty_samples['rmax_i']) + 0.1)  # normalized distance
features['exp(-D/Rmax)'] = np.exp(-D / (safe_val(ty_samples['rmax_i']) + 1.0))

# 相对方位角
az = safe_val(ty_samples.get('相对方位角', ty_samples.get('方位角', 0)))
features['sin(相对方位角)'] = np.sin(np.radians(az))
features['cos(相对方位角)'] = np.cos(np.radians(az))

# 象限
az_mod = az % 360
quadrant = np.select(
    [(az_mod >= 0) & (az_mod < 90), (az_mod >= 90) & (az_mod < 180),
     (az_mod >= 180) & (az_mod < 270), (az_mod >= 270) & (az_mod < 360)],
    [0, 1, 2, 3], default=0
)
for qi, qname in enumerate(['RF(右前)', 'LF(左前)', 'LR(左后)', 'RR(右后)']):
    features[qname] = (quadrant == qi).astype(float)

# 距离变化率 (15min内)
target_times = ty_samples['target_start_time'].values
dist_series = ty_samples['场站距离_km'].values
delta_D = np.zeros(len(ty_samples))
for i in range(1, len(ty_samples)):
    if target_times[i] != target_times[i-1]:
        dt_hours = (pd.Timestamp(target_times[i]) - pd.Timestamp(target_times[i-1])).total_seconds() / 3600
        if 0 < dt_hours < 2:
            delta_D[i] = (dist_series[i] - dist_series[i-1]) / max(dt_hours, 0.01)
features['ΔD (km/h)'] = delta_D

# --- 强度特征 ---
features['Vmax (m/s)'] = safe_val(ty_samples['风速'])
Pc = safe_val(ty_samples['气压'])
features['压差 Pn-Pc (hPa)'] = PN - Pc

# --- 尺度特征 ---
features['Rmax (km)'] = safe_val(ty_samples['rmax_i'])
features['B'] = safe_val(ty_samples['b_i'])

# --- 运动特征 ---
features['Vmove (m/s)'] = safe_val(ty_samples.get('移动速度(m/s)', ty_samples.get('移动速度', 0)))

# 移动方向
move_dir = safe_val(ty_samples.get('移动方向(°)', ty_samples.get('移动方向', 0)))
features['sin(移动方向)'] = np.sin(np.radians(move_dir))
features['cos(移动方向)'] = np.cos(np.radians(move_dir))

# --- 参数模型特征 ---
features['V_holland (m/s)'] = safe_val(ty_samples['hollandvg_az'])
features['V_yanmeng (m/s)'] = safe_val(ty_samples['ym风速_i'])
features['atten_w'] = safe_val(ty_samples['atten_w'])

# 局地风速变化率 (15min内)
v_pm = safe_val(ty_samples['ym风速_i'])
delta_Vpm = np.zeros(len(ty_samples))
for i in range(1, len(ty_samples)):
    if target_times[i] != target_times[i-1]:
        dt_hours = (pd.Timestamp(target_times[i]) - pd.Timestamp(target_times[i-1])).total_seconds() / 3600
        if 0 < dt_hours < 2:
            delta_Vpm[i] = (v_pm[i] - v_pm[i-1]) / max(dt_hours, 0.01)
features['ΔVpm (m/s/h)'] = delta_Vpm

# 切向/径向风速近似 (使用方位+参数风速)
v_param = safe_val(ty_samples['ym风速_i'])
sin_az = features['sin(相对方位角)']
cos_az = features['cos(相对方位角)']
features['Vtan (切向)'] = v_param * sin_az  # 切向分量
features['Vrad (径向)'] = v_param * cos_az  # 径向分量

# --- 结构特征 ---
features['D/Rmax - 1'] = D / (safe_val(ty_samples['rmax_i']) + 0.1) - 1.0

# --- 差异特征: NWP vs 参数模型 ---
# NWP风速可以从预测功率推断 (P=0.5*rho*A*Cp*V^3, 但这里直接用NWP特征)
# BiLSTM的输入包含NWP风速, 需要从构建数据集获取
# 这里用预测CSV中的y_pred_mw和y_true_mw的差值来代替NWP-param风区的差异

# 参数模型预测风速 (轮毂高度近似)
Vpm = safe_val(ty_samples['hollandvg_az'])  # Holland梯度风速代表参数模型风速

# NWP风速需要从训练数据获取. 这里用功率残差/装机容量作为风的差异代理
# 更精确的做法: 从bilstm训练数据中提取NWP风速序列
p_base_mw = ty_samples[pred_cols[47]].values.astype(float)  # 12h horizon
p_true_mw = ty_samples[true_cols[47]].values.astype(float)
# 功率归一化差异作为 NWP-参数 差异的代理
# V_nwp_proxy = capacity_factor = P_true / CAPACITY
# V_param_proxy = Vpm / max(Vpm)
features['NWP-param 功率差'] = (p_base_mw - p_true_mw) / CAPACITY

# --- 增加: NWP风速特征(从预测CSV能拿到的) ---
# bilstm预测残差(12h ahead)
features['残差(12h)'] = p_base_mw - p_true_mw

# ============ 4. 相关性计算 ============
print("[4] Computing correlations...")

# 清理NaN/Inf
for k in list(features.keys()):
    vals = features[k]
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.nan_to_num(vals, nan=0.0)
    features[k] = vals

# 目标变量: 绝对误差
target = overall_abs_err

# 计算 Pearson 和 Spearman
corr_results = []
for feat_name, feat_vals in features.items():
    if feat_vals.std() < 1e-8:
        continue
    # Pearson
    try:
        r_p, p_p = pearsonr(feat_vals, target)
    except:
        r_p, p_p = 0, 1
    # Spearman
    try:
        r_s, p_s = spearmanr(feat_vals, target)
    except:
        r_s, p_s = 0, 1
    corr_results.append({
        'feature': feat_name,
        'pearson_r': r_p, 'pearson_p': p_p,
        'spearman_r': r_s, 'spearman_p': p_s,
        'abs_r': abs(r_s),  # use Spearman for ranking (nonlinear)
        'mean': feat_vals.mean(), 'std': feat_vals.std(),
    })

df_corr = pd.DataFrame(corr_results).sort_values('abs_r', ascending=False)
print(df_corr[['feature', 'spearman_r', 'abs_r', 'pearson_r', 'pearson_p']].to_string(index=False))

# ============ 5. 画图 ============
print("[5] Plotting...")

# (a) 相关性排名图
fig, ax = plt.subplots(figsize=(12, 8))
top30 = df_corr.head(30)
colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in top30['spearman_r'].values]
bars = ax.barh(range(len(top30)), top30['abs_r'].values, color=colors, alpha=0.7, edgecolor='none')
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30['feature'].values, fontsize=9)
ax.set_xlabel('|Spearman ρ|', fontsize=12)
ax.set_title('Feature Correlation with BiLSTM Absolute Error (Typhoon Periods)', fontsize=13)

for i, (_, row) in enumerate(top30.iterrows()):
    ax.text(row['abs_r'] + 0.002, i, f'ρ={row["spearman_r"]:.3f}',
            va='center', fontsize=8, color='#2c3e50')

ax.grid(True, alpha=0.2, axis='x')
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/feature_correlation_ranking.png')
plt.close(fig)
print(f"  -> {OUT_DIR}/feature_correlation_ranking.png")

# (b) 按类别分组的相关性热力图
categories = {
    'Geometry': ['距离 D (km)', 'D/Rmax', 'exp(-D/Rmax)', 'sin(相对方位角)', 'cos(相对方位角)', 'ΔD (km/h)'],
    'Intensity': ['Vmax (m/s)', '压差 Pn-Pc (hPa)'],
    'Structure': ['Rmax (km)', 'B', 'D/Rmax - 1', 'atten_w'],
    'Parametric': ['V_holland (m/s)', 'V_yanmeng (m/s)', 'Vtan (切向)', 'Vrad (径向)', 'ΔVpm (m/s/h)'],
    'Difference': ['NWP-param 功率差', '残差(12h)'],
}

cat_feats = []
cat_names = []
for cat_name, feat_list in categories.items():
    for f in feat_list:
        if f in df_corr['feature'].values:
            cat_feats.append(f)
            cat_names.append(cat_name)

cat_data = df_corr.set_index('feature').loc[[f for f in cat_feats if f in df_corr['feature'].values]]
cat_df = cat_data[['spearman_r']].copy()
cat_df['category'] = [cat_names[cat_feats.index(i)] for i in cat_df.index if i in cat_feats]
cat_df = cat_df.reset_index()
cat_df = cat_df.drop_duplicates('feature')

# 按类别排序
cat_order = list(categories.keys())
cat_df['cat_order'] = cat_df['category'].apply(lambda x: cat_order.index(x) if x in cat_order else 999)
cat_df = cat_df.sort_values(['cat_order', 'spearman_r'])

fig, ax = plt.subplots(figsize=(10, 6))
feat_names = cat_df['feature'].values
spear_vals = cat_df['spearman_r'].values
bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
cat_to_color = {c: bar_colors[i%5] for i, c in enumerate(cat_order)}

for i, (fname, rho, cat) in enumerate(zip(feat_names, spear_vals, cat_df['category'].values)):
    color = cat_to_color[cat] if rho > 0 else '#e74c3c'
    ax.barh(i, abs(rho), color=color, alpha=0.5 + abs(rho) * 2, edgecolor=color, height=0.6)
    sign = '+' if rho > 0 else '-'
    ax.text(abs(rho) + 0.005, i, f'{cat}', va='center', fontsize=7, color='#7f8c8d')

ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels(feat_names, fontsize=9)
ax.set_xlabel('|Spearman ρ|', fontsize=12)
ax.set_title('Feature-Error Correlation by Category', fontsize=13)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cat_to_color[c], alpha=0.7, label=c) for c in cat_order]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
ax.grid(True, alpha=0.2, axis='x')
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/feature_correlation_by_category.png')
plt.close(fig)
print(f"  -> {OUT_DIR}/feature_correlation_by_category.png")

# (c) 不同 horizon 的关键特征表现
fig, ax = plt.subplots(figsize=(10, 5))
lead_labels_plot = ['1h', '4h', '8h', '12h', '24h']
top_features_for_heat = df_corr.head(8)['feature'].values

horizon_corrs = np.zeros((len(top_features_for_heat), len(lead_labels_plot)))
for fi, feat_name in enumerate(top_features_for_heat):
    feat_vals = features[feat_name]
    for hi, ll in enumerate(lead_labels_plot):
        try:
            r, _ = spearmanr(feat_vals, np.abs(errors[ll]))
            horizon_corrs[fi, hi] = r
        except:
            horizon_corrs[fi, hi] = 0

im = ax.imshow(horizon_corrs, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)

# Truncate feature names
short_names = [f[:25] for f in top_features_for_heat]
ax.set_yticks(range(len(short_names)))
ax.set_yticklabels(short_names, fontsize=9)
ax.set_xticks(range(len(lead_labels_plot)))
ax.set_xticklabels(lead_labels_plot, fontsize=10)
ax.set_xlabel('Lead Time', fontsize=12)
ax.set_title('Feature-Error Correlation by Horizon', fontsize=13)
fig.colorbar(im, ax=ax, label='Spearman ρ')
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/feature_correlation_by_horizon.png')
plt.close(fig)
print(f"  -> {OUT_DIR}/feature_correlation_by_horizon.png")

# ============ 6. 总结 ============
print("\n" + "="*60)
print("  TOP 10 Features for Corrector")
print("="*60)
top10 = df_corr.head(10)
for i, (_, row) in enumerate(top10.iterrows()):
    print(f"  {i+1}. {row['feature']:<25s}  Spearman ρ={row['spearman_r']:+.4f}  p={row['spearman_p']:.2e}")

# 建议校正器特征组合
print("\n  Recommended Corrector Features (based on correlation):")
struct_feats = [f for f in top10['feature'].values if any(
    k in f for k in ['Rmax', 'D/', '/Rmax', 'atten', 'B', 'exp', '压差'])]
param_feats = [f for f in top10['feature'].values if 'V_' in f or 'Vtan' in f or 'Vrad' in f]
geo_feats = [f for f in top10['feature'].values if 'ΔD' in f or 'sin' in f or 'cos' in f]
print(f"  Top structure: {struct_feats}")
print(f"  Top parametric: {param_feats}")
print(f"  Top geometry: {geo_feats}")
print(f"\n  Suggested combo (3-5 features):")
suggested = struct_feats[:2] + param_feats[:1] + geo_feats[:1]
print(f"  {suggested}")
