#!/usr/bin/env python3
"""
严格验证: 台风特征 vs BiLSTM预测误差的相关性
每一步都打印中间结果, 确保没有数据合并错误.
"""
import numpy as np
import pandas as pd
import sys, glob

sys.path.insert(0, '.')
from scipy.stats import spearmanr

HORIZON = 96
PRED_GLOB = 'data/predictions/bilstm_pred_*.csv'
TY_CSV = 'data/typhoon/typhoon19-24complete_15min_improved.csv'
PN = 1013.3  # ambient pressure

# ============================================================
# STEP 1: 加载预测数据
# ============================================================
print("=" * 70)
print("STEP 1: Load prediction CSVs")
pred_dfs = []
for fp in sorted(glob.glob(PRED_GLOB)):
    df = pd.read_csv(fp)
    pred_dfs.append(df)
    print(f"  {fp.split('/')[-1]:35s}  {len(df)} rows")
pred = pd.concat(pred_dfs, ignore_index=True)
pred['target_start_time'] = pd.to_datetime(pred['target_start_time'])
print(f"  TOTAL: {len(pred)} rows, time range: {pred['target_start_time'].min()} ~ {pred['target_start_time'].max()}")

# ============================================================
# STEP 2: 加载台风数据, 每时刻取最近台风
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Load typhoon data, keep closest per timestamp")

ty = pd.read_csv(TY_CSV, encoding='utf-8-sig')
ty['target_time'] = pd.to_datetime(ty['target_time'], errors='coerce')
print(f"  Raw typhoon rows: {len(ty)}")

# Sort by time then distance, keep first (= closest typhoon per time)
ty_sorted = ty.sort_values([ty.columns[15], ty.columns[14]])  # target_time, distance_km
ty_closest = ty_sorted.drop_duplicates(ty.columns[15], keep='first')
print(f"  After dedup (closest per time): {len(ty_closest)} rows")
print(f"  Typhoon time range: {ty_closest['target_time'].min()} ~ {ty_closest['target_time'].max()}")

# ============================================================
# STEP 3: 合并 - 预测起点匹配台风时间
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Merge predictions with typhoon features")

merged = pred.merge(ty_closest, left_on='target_start_time', right_on='target_time', how='inner')
print(f"  Merged (inner join): {len(merged)} rows")

# Check what fraction matched
total_pred = len(pred)
typhoon_matched = len(merged)
print(f"  Match rate: {typhoon_matched}/{total_pred} = {typhoon_matched/total_pred*100:.1f}%")

# Only keep rows where we have valid typhoon features (non-zero distance)
has_valid = merged.iloc[:, 14].notna() & (merged.iloc[:, 14] > 0)
merged_valid = merged[has_valid].copy()
print(f"  With valid distance (>0): {len(merged_valid)} rows")
print(f"  Unique typhoon events: {merged_valid.iloc[:, 1].nunique()}")

# ============================================================
# STEP 4: 计算目标变量 - BiLSTM预测误差
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Compute BiLSTM prediction errors")

true_cols = [f'y_true_t+{i+1}_mw' for i in range(HORIZON)]
pred_cols = [f'y_pred_t+{i+1}_mw' for i in range(HORIZON)]

T_all = merged_valid[true_cols].values.astype(float)
P_all = merged_valid[pred_cols].values.astype(float)

# Target: mean absolute error across all 96 horizons per sample
target_mae = np.mean(np.abs(P_all - T_all), axis=1)
print(f"  Target MAE: mean={target_mae.mean():.2f}, std={target_mae.std():.2f}, "
      f"min={target_mae.min():.2f}, max={target_mae.max():.2f}")

# Also compute for specific horizons
target_1h  = np.abs(P_all[:, 3] - T_all[:, 3])
target_6h  = np.abs(P_all[:, 23] - T_all[:, 23])
target_12h = np.abs(P_all[:, 47] - T_all[:, 47])
target_24h = np.abs(P_all[:, 95] - T_all[:, 95])

# ============================================================
# STEP 5: 提取并构建所有特征
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Build feature matrix")

n = len(merged_valid)
feat = {}

# --- Raw extractions (using column indices for Chinese-named columns) ---
D      = merged_valid.iloc[:, 14].values.astype(float)   # distance_km
az     = merged_valid.iloc[:, 16].values.astype(float)   # azimuth
rel_az = merged_valid.iloc[:, 17].values.astype(float)   # relative_azimuth
mvspd  = merged_valid.iloc[:, 18].values.astype(float)   # move_speed_ms
mvdir  = merged_valid.iloc[:, 19].values.astype(float)   # move_direction_deg
ws_val = merged_valid.iloc[:, 10].values.astype(float)   # wind_speed_ms
pres   = merged_valid.iloc[:, 11].values.astype(float)   # pressure_hPa
level  = merged_valid.iloc[:, 9].values.astype(float)    # typhoon_level
lat    = merged_valid.iloc[:, 7].values.astype(float)    # latitude
lon    = merged_valid.iloc[:, 6].values.astype(float)    # longitude

rmax_i   = merged_valid['rmax_i'].fillna(0).values.astype(float)
b_i      = merged_valid['b_i'].fillna(0).values.astype(float)
holland  = merged_valid['hollandvg_az'].fillna(0).values.astype(float)
atten_w  = merged_valid['atten_w'].fillna(0).values.astype(float)
ym_wind  = merged_valid.iloc[:, 34].fillna(0).values.astype(float)  # ym风速_i

print(f"  distance range:  [{D.min():.1f}, {D.max():.1f}] km")
print(f"  rmax_i range:    [{rmax_i[rmax_i>0].min():.1f}, {rmax_i.max():.1f}] km")
print(f"  b_i range:       [{b_i[b_i>0].min():.2f}, {b_i.max():.2f}]")
print(f"  atten_w range:   [{atten_w[atten_w>0].min():.4f}, {atten_w.max():.4f}]")
print(f"  ym_wind range:   [{ym_wind[ym_wind>0].min():.1f}, {ym_wind.max():.1f}]")
print(f"  holland range:   [{holland[holland>0].min():.1f}, {holland.max():.1f}]")
print(f"  Vmax range:      [{ws_val.min():.1f}, {ws_val.max():.1f}]")
print(f"  P_deficit range: [{PN-pres.max():.1f}, {PN-pres.min():.1f}]")

# === Geometry ===
feat['01_dist_D_km'] = D
feat['02_D_over_Rmax'] = np.where(rmax_i > 0.5, D / rmax_i, 0.0)
feat['03_exp_neg_D_over_Rmax'] = np.exp(-D / np.maximum(rmax_i, 1.0))
feat['04_sin_rel_azimuth'] = np.sin(np.radians(rel_az))
feat['05_cos_rel_azimuth'] = np.cos(np.radians(rel_az))
feat['06_sin_azimuth'] = np.sin(np.radians(az))
feat['07_cos_azimuth'] = np.cos(np.radians(az))

# Quadrant
az_mod = rel_az % 360
feat['08_quad_RF'] = ((az_mod >= 0)   & (az_mod < 90)).astype(float)
feat['09_quad_LF'] = ((az_mod >= 90)  & (az_mod < 180)).astype(float)
feat['10_quad_LR'] = ((az_mod >= 180) & (az_mod < 270)).astype(float)
feat['11_quad_RR'] = ((az_mod >= 270) & (az_mod < 360)).astype(float)

# Delta D: compute distance change rate (km/h) between consecutive 15-min steps
dD = np.zeros(n)
for i in range(1, n):
    dt_h = (merged_valid['target_start_time'].iloc[i] - merged_valid['target_start_time'].iloc[i-1]).total_seconds() / 3600.0
    if 0 < dt_h < 2.0:
        dD[i] = (D[i] - D[i-1]) / dt_h
feat['12_delta_D_km_per_h'] = dD

# === Intensity ===
feat['13_Vmax_ms'] = ws_val
feat['14_P_deficit_hPa'] = PN - pres  # positive = stronger typhoon

# === Scale / Structure ===
feat['15_Rmax_km'] = rmax_i
feat['16_B_param'] = b_i
feat['17_atten_w'] = atten_w
feat['18_D_over_Rmax_minus_1'] = np.where(rmax_i > 0.5, D / rmax_i - 1.0, 0.0)

# === Motion ===
feat['19_Vmove_ms'] = mvspd
feat['20_sin_move_dir'] = np.sin(np.radians(mvdir))
feat['21_cos_move_dir'] = np.cos(np.radians(mvdir))

# === Parametric winds ===
feat['22_V_holland_ms'] = holland
feat['23_V_yanmeng_ms'] = ym_wind
feat['24_Vtan_tangential'] = ym_wind * feat['04_sin_rel_azimuth']
feat['25_Vrad_radial'] = ym_wind * feat['05_cos_rel_azimuth']

# Delta Vpm
dV = np.zeros(n)
for i in range(1, n):
    dt_h = (merged_valid['target_start_time'].iloc[i] - merged_valid['target_start_time'].iloc[i-1]).total_seconds() / 3600.0
    if 0 < dt_h < 2.0:
        dV[i] = (ym_wind[i] - ym_wind[i-1]) / dt_h
feat['26_delta_Vpm_ms_per_h'] = dV

# === Difference NWP vs Parametric ===
p_base_12h = P_all[:, 47]
p_true_12h = T_all[:, 47]
feat['27_NWP_param_power_diff'] = (p_base_12h - p_true_12h) / 200.0
feat['28_residual_12h_MW'] = p_base_12h - p_true_12h

# === Misc ===
feat['29_typhoon_level'] = level
feat['30_latitude'] = lat
feat['31_longitude'] = lon

# ============================================================
# STEP 6: 清理NaN/Inf
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Clean features")

for k in feat:
    v = feat[k]
    nan_count = np.isnan(v).sum()
    inf_count = np.isinf(v).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  {k}: {nan_count} NaN, {inf_count} Inf -> replaced with 0")
    feat[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================
# STEP 7: Spearman 相关
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Spearman rank correlation")
print("  Target: mean |error| over 96 horizons")
print(f"  N = {n}")

results = []
for name, vals in feat.items():
    if np.std(vals) < 1e-8:
        print(f"  SKIP {name}: constant")
        continue
    r, p = spearmanr(vals, target_mae)
    results.append({
        'feature': name,
        'spearman_r': r,
        'abs_r': abs(r),
        'p_value': p,
        'sig': '***' if p < 1e-100 else '**' if p < 1e-10 else '*' if p < 0.01 else 'ns'
    })

df = pd.DataFrame(results).sort_values('abs_r', ascending=False)
df['rank'] = range(1, len(df) + 1)

pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 150)
pd.set_option('display.float_format', '{:.4f}'.format)

print(f"\n{'rank':>4s} {'feature':<28s} {'spearman_r':>10s} {'abs_r':>8s} {'p_value':>12s} {'sig':>4s}")
print("-" * 70)
for _, row in df.iterrows():
    print(f"{row['rank']:4d} {row['feature']:<28s} {row['spearman_r']:10.4f} {row['abs_r']:8.4f} {row['p_value']:12.2e} {row['sig']:>4s}")

# ============================================================
# STEP 8: 按误差类型检查
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Correlation by horizon (sanity check)")
print(f"{'Feature':<28s} {'1h_MAE':>8s} {'6h_MAE':>8s} {'12h_MAE':>8s} {'24h_MAE':>8s}")
print("-" * 65)
top12 = df.head(12)['feature'].values
targets = {'1h': target_1h, '6h': target_6h, '12h': target_12h, '24h': target_24h}
for fname in top12:
    vals = feat[fname]
    row_str = f"{fname:<28s}"
    for tname, tvals in targets.items():
        r, _ = spearmanr(vals, tvals)
        row_str += f" {r:8.4f}"
    print(row_str)

# ============================================================
# STEP 9: 验证我们之前用的特征 (IMPR_STRUCT)
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: OUR CURRENT FEATURES (IMPR_STRUCT) verification")
our_feats = ['15_Rmax_km', '16_B_param', '17_atten_w']
for fname in our_feats:
    r, p = spearmanr(feat[fname], target_mae)
    print(f"  {fname}: spearman_r={r:.4f}, p={p:.2e}")

# 偏相关: 控制距离后
from scipy.stats import rankdata
print("\n  Partial rank correlation (controlling for distance):")
dist_rank = rankdata(feat['01_dist_D_km'])
mae_rank  = rankdata(target_mae)
for fname in our_feats:
    feat_rank = rankdata(feat[fname])
    # Simple: correlation of residuals after regressing out distance
    # Use rank-based partial: corr(resid_feat, resid_mae) where resid = rank - pred_from_dist
    # Approximation: corr(feat_rank - dist_rank * slope_feat, mae_rank - dist_rank * slope_mae)
    from scipy.stats import linregress
    _, _, _, _, std_f = linregress(dist_rank, feat_rank)
    _, _, _, _, std_m = linregress(dist_rank, mae_rank)
    slope_f, intercept_f, _, _, _ = linregress(dist_rank, feat_rank)
    slope_m, intercept_m, _, _, _ = linregress(dist_rank, mae_rank)
    resid_f = feat_rank - (slope_f * dist_rank + intercept_f)
    resid_m = mae_rank - (slope_m * dist_rank + intercept_m)
    r_partial, p_partial = spearmanr(resid_f, resid_m)
    print(f"  {fname} | dist: spearman_r_partial={r_partial:.4f}, p={p_partial:.2e}")

print("\nDone.")
