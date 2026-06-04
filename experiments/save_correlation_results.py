#!/usr/bin/env python3
"""
保存所有特征相关性分析结果, 供论文使用.
输出目录: results/correlation_analysis/
"""
import numpy as np, pandas as pd, glob, os, json
from datetime import datetime
from scipy.stats import spearmanr

HORIZON = 96
PN = 1013.3
PRED_GLOB = 'data/predictions/bilstm_pred_*.csv'
TY_CSV = 'data/typhoon/typhoon19-24complete_15min_improved.csv'

OUT_DIR = 'results/correlation_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
pred = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(PRED_GLOB))], ignore_index=True)
pred['t0'] = pd.to_datetime(pred['target_start_time'])

ty = pd.read_csv(TY_CSV, encoding='utf-8-sig')
ty['tt'] = pd.to_datetime(ty['target_time'], errors='coerce')
ty_c = ty.sort_values([ty.columns[15], ty.columns[14]]).drop_duplicates(ty.columns[15], keep='first')
m = pred.merge(ty_c, left_on='t0', right_on='tt', how='inner')
m = m[m.iloc[:, 14].notna() & (m.iloc[:, 14] > 0)].copy()
n = len(m)

tc = [f'y_true_t+{i+1}_mw' for i in range(HORIZON)]
pc = [f'y_pred_t+{i+1}_mw' for i in range(HORIZON)]
T = m[tc].values.astype(float)
P = m[pc].values.astype(float)

# ============================================================
# 2. 计算所有目标变量
# ============================================================
targets = {}
# 每个horizon的误差
for h_idx, h_name in [(3, '1h'), (7, '2h'), (11, '3h'), (15, '4h'), (23, '6h'),
                        (31, '8h'), (47, '12h'), (71, '18h'), (95, '24h')]:
    e = P[:, h_idx] - T[:, h_idx]
    targets[f'E_signed_{h_name}'] = e
    targets[f'E_abs_{h_name}'] = np.abs(e)

# 均值误差
targets['E_signed_mean_all'] = np.mean(P - T, axis=1)
targets['E_abs_mean_all'] = np.mean(np.abs(P - T), axis=1)
targets['RMSE_sample'] = np.sqrt(np.mean((P - T)**2, axis=1))

# ============================================================
# 3. 构建所有特征
# ============================================================
D = m.iloc[:, 14].values.astype(float)
rmax = m['rmax_i'].fillna(0).values.astype(float)
b = m['b_i'].fillna(0).values.astype(float)
atten = m['atten_w'].fillna(0).values.astype(float)
ym = m.iloc[:, 34].fillna(0).values.astype(float)
holland = m['hollandvg_az'].fillna(0).values.astype(float)
rel_az = m.iloc[:, 17].fillna(0).values.astype(float)
az = m.iloc[:, 16].fillna(0).values.astype(float)
mvspd = m.iloc[:, 18].fillna(0).values.astype(float)
mvdir = m.iloc[:, 19].fillna(0).values.astype(float)
ws = m.iloc[:, 10].fillna(0).values.astype(float)
pres = m.iloc[:, 11].fillna(1010).values.astype(float)
level = m.iloc[:, 9].fillna(0).values.astype(float)
lat = m.iloc[:, 7].values.astype(float)
lon = m.iloc[:, 6].values.astype(float)

F = {}

# -- 几何 --
F['dist_D_km'] = D
F['D_over_Rmax'] = np.where(rmax > 0.5, D / rmax, 0.0)
F['exp_neg_D_over_Rmax'] = np.exp(-D / np.maximum(rmax, 1.0))
F['sin_rel_azimuth'] = np.sin(np.radians(rel_az))
F['cos_rel_azimuth'] = np.cos(np.radians(rel_az))
F['sin_azimuth'] = np.sin(np.radians(az))
F['cos_azimuth'] = np.cos(np.radians(az))

az_mod = rel_az % 360
F['quad_RF_right_front'] = ((az_mod >= 0) & (az_mod < 90)).astype(float)
F['quad_LF_left_front'] = ((az_mod >= 90) & (az_mod < 180)).astype(float)
F['quad_LR_left_rear'] = ((az_mod >= 180) & (az_mod < 270)).astype(float)
F['quad_RR_right_rear'] = ((az_mod >= 270) & (az_mod < 360)).astype(float)

# Delta D
dD = np.zeros(n)
for i in range(1, n):
    dt = (m['t0'].iloc[i] - m['t0'].iloc[i-1]).total_seconds() / 3600.0
    if 0 < dt < 2.0:
        dD[i] = (D[i] - D[i-1]) / dt
F['delta_D_km_per_h'] = dD

# -- 强度 --
F['Vmax_ms'] = ws
F['P_deficit_hPa'] = PN - pres

# -- 尺度/结构 --
F['Rmax_km'] = rmax
F['B_param'] = b
F['atten_w'] = atten
F['D_over_Rmax_minus_1'] = np.where(rmax > 0.5, D / rmax - 1.0, 0.0)

# -- 运动 --
F['Vmove_ms'] = mvspd
F['sin_move_dir'] = np.sin(np.radians(mvdir))
F['cos_move_dir'] = np.cos(np.radians(mvdir))

# -- 参数模型 --
F['V_holland_ms'] = holland
F['V_yanmeng_ms'] = ym
F['Vtan_tangential'] = ym * F['sin_rel_azimuth']
F['Vrad_radial'] = ym * F['cos_rel_azimuth']

dV = np.zeros(n)
for i in range(1, n):
    dt = (m['t0'].iloc[i] - m['t0'].iloc[i-1]).total_seconds() / 3600.0
    if 0 < dt < 2.0:
        dV[i] = (ym[i] - ym[i-1]) / dt
F['delta_Vpm_ms_per_h'] = dV

# -- 差异 --
F['NWP_param_diff_12h'] = P[:, 47] - T[:, 47]

# -- 杂项 --
F['typhoon_level'] = level
F['latitude'] = lat
F['longitude'] = lon

# 清理
for k in F:
    F[k] = np.nan_to_num(F[k], nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================
# 4. 特征元数据
# ============================================================
feature_meta = {
    'dist_D_km':              {'category': 'Geometry', 'description': 'Distance from wind farm to typhoon center (km)', 'formula': 'Haversine(lat_farm, lon_farm, lat_ty, lon_ty)'},
    'D_over_Rmax':            {'category': 'Geometry', 'description': 'Normalized distance: D / Rmax', 'formula': 'dist / rmax_i'},
    'exp_neg_D_over_Rmax':    {'category': 'Geometry', 'description': 'Influence intensity index', 'formula': 'exp(-dist / rmax_i)'},
    'sin_rel_azimuth':        {'category': 'Geometry', 'description': 'Sine of relative azimuth', 'formula': 'sin(radians(rel_az))'},
    'cos_rel_azimuth':        {'category': 'Geometry', 'description': 'Cosine of relative azimuth', 'formula': 'cos(radians(rel_az))'},
    'sin_azimuth':            {'category': 'Geometry', 'description': 'Sine of absolute azimuth', 'formula': 'sin(radians(az))'},
    'cos_azimuth':            {'category': 'Geometry', 'description': 'Cosine of absolute azimuth', 'formula': 'cos(radians(az))'},
    'quad_RF_right_front':    {'category': 'Geometry', 'description': 'Right-front quadrant (0-90 deg)', 'formula': 'rel_az in [0,90)'},
    'quad_LF_left_front':     {'category': 'Geometry', 'description': 'Left-front quadrant (90-180 deg)', 'formula': 'rel_az in [90,180)'},
    'quad_LR_left_rear':      {'category': 'Geometry', 'description': 'Left-rear quadrant (180-270 deg)', 'formula': 'rel_az in [180,270)'},
    'quad_RR_right_rear':     {'category': 'Geometry', 'description': 'Right-rear quadrant (270-360 deg)', 'formula': 'rel_az in [270,360)'},
    'delta_D_km_per_h':       {'category': 'Geometry', 'description': 'Distance change rate (km/h)', 'formula': 'd(dist)/dt'},
    'Vmax_ms':                {'category': 'Intensity', 'description': 'Maximum wind speed from CMA best track (m/s)', 'formula': 'CMA best track'},
    'P_deficit_hPa':          {'category': 'Intensity', 'description': 'Pressure deficit P_ambient - P_center (hPa)', 'formula': '1013.3 - Pc'},
    'Rmax_km':                {'category': 'Structure', 'description': 'Radius of maximum wind - Willoughby formula (km)', 'formula': 'Willoughby(1998)'},
    'B_param':                {'category': 'Structure', 'description': 'Holland B parameter - Powell formula', 'formula': 'Powell(2005)'},
    'atten_w':                {'category': 'Structure', 'description': 'Speed-dependent attenuation weight', 'formula': 'exp(-dist / D_eff), D_eff=500*v/5'},
    'D_over_Rmax_minus_1':    {'category': 'Structure', 'description': 'Proximity to Rmax circle: D/Rmax - 1', 'formula': 'D/Rmax - 1'},
    'Vmove_ms':               {'category': 'Motion', 'description': 'Typhoon translation speed (m/s)', 'formula': 'd(center_pos)/dt'},
    'sin_move_dir':           {'category': 'Motion', 'description': 'Sine of movement direction', 'formula': 'sin(radians(move_dir))'},
    'cos_move_dir':           {'category': 'Motion', 'description': 'Cosine of movement direction', 'formula': 'cos(radians(move_dir))'},
    'V_holland_ms':           {'category': 'Parametric', 'description': 'Holland gradient wind speed (m/s)', 'formula': 'Holland(1980)+Batts gradient wind'},
    'V_yanmeng_ms':           {'category': 'Parametric', 'description': 'Yan Meng surface wind speed (m/s)', 'formula': 'Yan Meng(1985) parametric model'},
    'Vtan_tangential':        {'category': 'Parametric', 'description': 'Tangential wind component', 'formula': 'V_ym * sin(rel_az)'},
    'Vrad_radial':            {'category': 'Parametric', 'description': 'Radial wind component', 'formula': 'V_ym * cos(rel_az)'},
    'delta_Vpm_ms_per_h':     {'category': 'Parametric', 'description': 'Parametric wind change rate (m/s/h)', 'formula': 'd(V_ym)/dt'},
    'NWP_param_diff_12h':     {'category': 'Difference', 'description': 'NWP prediction error at 12h', 'formula': 'P_pred_12h - P_true_12h'},
    'typhoon_level':          {'category': 'Misc', 'description': 'CMA typhoon intensity level (0-10)', 'formula': 'CMA best track'},
    'latitude':               {'category': 'Misc', 'description': 'Typhoon center latitude (deg)', 'formula': 'CMA best track'},
    'longitude':              {'category': 'Misc', 'description': 'Typhoon center longitude (deg)', 'formula': 'CMA best track'},
}

# ============================================================
# 5. 计算与 3 个核心目标的全相关性
# ============================================================
core_targets = {
    'E_signed_12h': P[:, 47] - T[:, 47],
    'E_abs_12h': np.abs(P[:, 47] - T[:, 47]),
    'E_abs_mean_all': targets['E_abs_mean_all'],
}

results = []
for fname, fvals in F.items():
    if fvals.std() < 1e-8:
        continue
    row = {'feature': fname, 'category': feature_meta.get(fname, {}).get('category', 'Unknown'),
           'description': feature_meta.get(fname, {}).get('description', ''),
           'formula': feature_meta.get(fname, {}).get('formula', ''),
           'mean': float(fvals.mean()), 'std': float(fvals.std()),
           'min': float(fvals.min()), 'max': float(fvals.max())}
    for tname, tvals in core_targets.items():
        r, p = spearmanr(fvals, tvals)
        row[f'{tname}_spearman_r'] = r
        row[f'{tname}_p_value'] = p
    results.append(row)

df_core = pd.DataFrame(results)
df_core['max_abs_r'] = df_core[[f'{t}_spearman_r' for t in core_targets]].abs().max(axis=1)

# 分类: SIGN / MAGNITUDE / BOTH
def classify_type(row):
    r_sign = abs(row['E_signed_12h_spearman_r'])
    r_abs  = abs(row['E_abs_12h_spearman_r'])
    if r_sign < 0.03 and r_abs < 0.03:
        return 'NEGLIGIBLE'
    if r_sign > r_abs * 1.5:
        return 'SIGN (bias direction)'
    elif r_abs > r_sign * 1.5:
        return 'MAGNITUDE (error size)'
    else:
        return 'BOTH'

df_core['sensitivity_type'] = df_core.apply(classify_type, axis=1)
df_core = df_core.sort_values('max_abs_r', ascending=False)
df_core.insert(0, 'rank', range(1, len(df_core) + 1))

# ============================================================
# 6. 按 horizon 计算相关性 (热力图数据)
# ============================================================
horizons_for_heat = [(3,'1h'),(7,'2h'),(11,'3h'),(15,'4h'),(23,'6h'),(31,'8h'),(47,'12h'),(71,'18h'),(95,'24h')]
horizon_corr_rows = []
for fname in df_core['feature'].values[:15]:  # top 15 features
    fvals = F[fname]
    row = {'feature': fname}
    for h_idx, h_name in horizons_for_heat:
        r_e, _ = spearmanr(fvals, P[:, h_idx] - T[:, h_idx])
        r_ae, _ = spearmanr(fvals, np.abs(P[:, h_idx] - T[:, h_idx]))
        row[f'E_signed_{h_name}'] = r_e
        row[f'E_abs_{h_name}'] = r_ae
    horizon_corr_rows.append(row)
df_horizon = pd.DataFrame(horizon_corr_rows)

# ============================================================
# 7. 保存所有结果
# ============================================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# (a) 主相关性排名 (含类型)
df_core.to_csv(f'{OUT_DIR}/feature_correlation_master.csv', index=False, encoding='utf-8-sig')
print(f'Saved: feature_correlation_master.csv ({len(df_core)} features)')

# (b) 按horizon的相关性
df_horizon.to_csv(f'{OUT_DIR}/feature_correlation_by_horizon.csv', index=False, encoding='utf-8-sig')
print(f'Saved: feature_correlation_by_horizon.csv ({len(df_horizon)} features x {len(horizons_for_heat)*2} targets)')

# (c) 特征元数据
meta_df = pd.DataFrame([
    {'feature': k, **v} for k, v in feature_meta.items() if k in df_core['feature'].values
])
meta_df.to_csv(f'{OUT_DIR}/feature_metadata.csv', index=False, encoding='utf-8-sig')
print(f'Saved: feature_metadata.csv')

# (d) 数据统计摘要
summary = {
    'analysis_date': timestamp,
    'n_samples': int(n),
    'n_typhoon_events': int(m.iloc[:, 1].nunique()),
    'time_range': f'{m["t0"].min()} ~ {m["t0"].max()}',
    'horizon': HORIZON,
    'capacity_MW': 200.0,
    'target_description': 'BiLSTM prediction error during typhoon periods',
    'target_mean_MAE': float(targets['E_abs_mean_all'].mean()),
    'target_std_MAE': float(targets['E_abs_mean_all'].std()),
    'key_findings': {
        'best_feature': df_core.iloc[0]['feature'],
        'best_spearman_r': float(df_core.iloc[0]['max_abs_r']),
        'our_features_ranking': {
            'Rmax_km': int(df_core[df_core['feature']=='Rmax_km']['rank'].values[0]),
            'B_param': int(df_core[df_core['feature']=='B_param']['rank'].values[0]),
            'atten_w': int(df_core[df_core['feature']=='atten_w']['rank'].values[0]),
        },
        'recommended_top5': df_core[df_core['sensitivity_type']=='BOTH'].head(5)['feature'].tolist(),
    }
}
with open(f'{OUT_DIR}/analysis_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print(f'Saved: analysis_summary.json')

# (e) 人类可读摘要
with open(f'{OUT_DIR}/README.txt', 'w', encoding='utf-8') as f:
    f.write(f"Correlation Analysis Results\n")
    f.write(f"{'='*60}\n")
    f.write(f"Date: {timestamp}\n")
    f.write(f"Samples: {n} typhoon-period prediction windows\n")
    f.write(f"Horizon: {HORIZON} steps (15-min each, 24h total)\n\n")

    f.write(f"TOP 10 Features (by max |Spearman r| across signed+absolute error):\n")
    f.write(f"{'-'*60}\n")
    for _, row in df_core.head(10).iterrows():
        f.write(f"  {row['rank']:2d}. {row['feature']:<28s} |E_signed|={abs(row['E_signed_12h_spearman_r']):.3f}  |E_abs|={abs(row['E_abs_12h_spearman_r']):.3f}  [{row['sensitivity_type']}]\n")

    f.write(f"\nOUR IMPR_STRUCT FEATURES:\n")
    f.write(f"{'-'*60}\n")
    for fn in ['Rmax_km', 'B_param', 'atten_w']:
        r = df_core[df_core['feature']==fn]
        if len(r) > 0:
            row = r.iloc[0]
            f.write(f"  {row['feature']}: rank={row['rank']}, r_signed={row['E_signed_12h_spearman_r']:.3f}, r_abs={row['E_abs_12h_spearman_r']:.3f}\n")

    f.write(f"\nRECOMMENDED FEATURE SET (BOTH type, top 5):\n")
    f.write(f"{'-'*60}\n")
    both_top = df_core[df_core['sensitivity_type']=='BOTH'].head(5)
    for _, row in both_top.iterrows():
        f.write(f"  {row['feature']}: r_signed={row['E_signed_12h_spearman_r']:.3f}, r_abs={row['E_abs_12h_spearman_r']:.3f}\n")

print(f'Saved: README.txt')

# (f) 打印到控制台
print(f"\n{'='*60}")
print(f"TOP 15 with sensitivity type:")
print(f"{'='*60}")
cols_show = ['rank','feature','category','E_signed_12h_spearman_r','E_abs_12h_spearman_r','E_abs_mean_all_spearman_r','sensitivity_type']
print(df_core[cols_show].head(15).to_string(index=False))
print(f"\nAll saved to: {OUT_DIR}/")
