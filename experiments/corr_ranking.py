#!/usr/bin/env python3
"""Full feature correlation ranking with English labels only."""
import numpy as np, pandas as pd, sys, glob
sys.path.insert(0, '.')
from scipy.stats import spearmanr

HORIZON = 96
PRED_GLOB = 'data/predictions/bilstm_pred_*.csv'
TY_CSV = 'data/typhoon/typhoon19-24complete_15min_improved.csv'

# Load predictions
pred = pd.concat([pd.read_csv(fp) for fp in sorted(glob.glob(PRED_GLOB))], ignore_index=True)
pred['target_start_time'] = pd.to_datetime(pred['target_start_time'])

# Load typhoon data, sort by distance to get closest typhoon per time
ty = pd.read_csv(TY_CSV, encoding='utf-8-sig')
ty['target_time'] = pd.to_datetime(ty['target_time'], errors='coerce')
# Column indices: 14=distance, 15=target_time
ty_sorted = ty.sort_values([ty.columns[15], ty.columns[14]]).drop_duplicates(ty.columns[15], keep='first')

merged = pred.merge(ty_sorted, left_on='target_start_time', right_on='target_time', how='inner')
print(f'Typhoon samples: {len(merged)}')

# Target: mean absolute error across all horizons
true_cols = [f'y_true_t+{i+1}_mw' for i in range(HORIZON)]
pred_cols = [f'y_pred_t+{i+1}_mw' for i in range(HORIZON)]
T = merged[true_cols].values.astype(float)
P = merged[pred_cols].values.astype(float)
target = np.mean(np.abs(P - T), axis=1)

PN = 1013.3
F = {}  # feature dict

# Extract raw values using column indices where needed
# Col 14: distance (km)
D = merged.iloc[:, 14].fillna(0).values.astype(float)
# Col 16: azimuth, 17: relative azimuth
az = merged.iloc[:, 16].fillna(0).values.astype(float)
rel_az = merged.iloc[:, 17].fillna(0).values.astype(float)
# Col 18: move speed (m/s)
mvspd = merged.iloc[:, 18].fillna(0).values.astype(float)
# Col 19: move direction (deg)
mvdir = merged.iloc[:, 19].fillna(0).values.astype(float)
# Col 10: wind speed (m/s)
ws = merged.iloc[:, 10].fillna(0).values.astype(float)
# Col 11: pressure (hPa)
pres = merged.iloc[:, 11].fillna(1010).values.astype(float)
# Col 9: typhoon level
level = merged.iloc[:, 9].fillna(0).values.astype(float)
# Col 7: latitude
lat = merged.iloc[:, 7].fillna(0).values.astype(float)

# English-named columns
rmax = merged['rmax_i'].fillna(0).values.astype(float)
b = merged['b_i'].fillna(0).values.astype(float)
holland = merged['hollandvg_az'].fillna(0).values.astype(float)
atten = merged['atten_w'].fillna(0).values.astype(float)
ym = merged.iloc[:, 34].fillna(0).values.astype(float)  # col 34: ym风速_i

# ==== GEOMETRY ====
F['dist_D_km'] = D
F['D_over_Rmax'] = D / (rmax + 0.1)
F['exp_neg_D_over_Rmax'] = np.exp(-D / (rmax + 1.0))
F['sin_rel_azimuth'] = np.sin(np.radians(rel_az))
F['cos_rel_azimuth'] = np.cos(np.radians(rel_az))
F['sin_azimuth'] = np.sin(np.radians(az))
F['cos_azimuth'] = np.cos(np.radians(az))

# Quadrant
az_mod = rel_az % 360
F['quad_RF_right_front'] = ((az_mod>=0)&(az_mod<90)).astype(float)
F['quad_LF_left_front'] = ((az_mod>=90)&(az_mod<180)).astype(float)
F['quad_LR_left_rear'] = ((az_mod>=180)&(az_mod<270)).astype(float)
F['quad_RR_right_rear'] = ((az_mod>=270)&(az_mod<360)).astype(float)

# Delta D
times = merged['target_start_time'].values
dD = np.zeros(len(merged))
for i in range(1, len(merged)):
    dt = (pd.Timestamp(times[i]) - pd.Timestamp(times[i-1])).total_seconds() / 3600
    if 0 < dt < 2:
        dD[i] = (D[i] - D[i-1]) / max(dt, 0.01)
F['delta_D_km_per_h'] = dD

# ==== INTENSITY ====
F['Vmax_ms'] = ws
F['P_deficit_hPa'] = PN - pres

# ==== STRUCTURE ====
F['Rmax_km'] = rmax
F['B_param'] = b
F['atten_w'] = atten
F['D_over_Rmax_minus_1'] = D / (rmax + 0.1) - 1.0

# ==== MOTION ====
F['Vmove_ms'] = mvspd
F['sin_move_dir'] = np.sin(np.radians(mvdir))
F['cos_move_dir'] = np.cos(np.radians(mvdir))

# ==== PARAMETRIC ====
F['V_holland_ms'] = holland
F['V_yanmeng_ms'] = ym
F['Vtan_tangential'] = ym * F['sin_rel_azimuth']
F['Vrad_radial'] = ym * F['cos_rel_azimuth']
dY = np.zeros(len(merged))
for i in range(1, len(merged)):
    dt = (pd.Timestamp(times[i]) - pd.Timestamp(times[i-1])).total_seconds() / 3600
    if 0 < dt < 2:
        dY[i] = (ym[i] - ym[i-1]) / max(dt, 0.01)
F['delta_Vpm_ms_per_h'] = dY

# ==== DIFFERENCE ====
p_b12 = merged[pred_cols[47]].fillna(0).values.astype(float)
p_t12 = merged[true_cols[47]].fillna(0).values.astype(float)
F['NWP_param_power_diff'] = (p_b12 - p_t12) / 200.0
F['residual_12h_MW'] = p_b12 - p_t12

# ==== MISC ====
F['typhoon_level'] = level
F['latitude'] = lat

# Clean
for k in list(F.keys()):
    F[k] = np.nan_to_num(np.where(np.isfinite(F[k]), F[k], 0.0), nan=0.0)

# Spearman rank correlation
results = []
for name, vals in F.items():
    if vals.std() < 1e-8:
        continue
    try:
        r, p = spearmanr(vals, target)
        results.append({'feature': name, 'spearman_r': r, 'abs_r': abs(r), 'p_value': p})
    except:
        pass

df = pd.DataFrame(results).sort_values('abs_r', ascending=False)
df['rank'] = range(1, len(df)+1)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 120)
print(df[['rank','feature','spearman_r','abs_r','p_value']].to_string(index=False))
