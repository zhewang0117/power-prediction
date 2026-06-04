#!/usr/bin/env python3
"""
Typhoon power impact window prediction: physics rough + ML refinement.
"""
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Feature extraction per typhoon
# ============================================================
ty = pd.read_csv('data/typhoon/typhoon19-24complete_15min_within1000km_aug8.csv', encoding='utf-8-sig')
ty['target_time'] = pd.to_datetime(ty['target_time'])

YM_THRESH = 2.0
PWR_THRESH = 100
DECAY = 500
ty['ym_eff'] = ty['ym风速'] * np.exp(-ty['场站距离_km'] / DECAY)

def find_main_window(mask, times):
    wins = []
    in_win = False; start = None
    for i, m in enumerate(mask):
        if m and not in_win: start = times[i]; in_win = True
        elif not m and in_win: wins.append((start, times[i-1])); in_win = False
    if in_win: wins.append((start, times[-1]))
    if not wins: return None
    return max(wins, key=lambda w: (w[1]-w[0]).total_seconds())

rows = []
for tid, sub in ty.groupby('台风编号'):
    sub = sub.sort_values('target_time')
    name = sub['台风中文名称'].iloc[0]
    yr = int(sub['target_time'].dt.year.mode().iloc[0])
    d_min = sub['场站距离_km'].min()

    t0 = sub['target_time'].min() - pd.Timedelta(hours=48)
    t1 = sub['target_time'].max() + pd.Timedelta(hours=48)
    try:
        ghdt = pd.read_csv(f'data/power/ghdt_merged_15min_{yr}.csv')
        ghdt['time'] = pd.to_datetime(ghdt['time'])
        w = ghdt[(ghdt['time']>=t0)&(ghdt['time']<=t1)].set_index('time')['VALUE']
    except: continue

    s = sub.set_index('target_time')
    common = w.index.intersection(s.index)
    if len(common) < 48: continue

    p = w.loc[common].values
    ym_eff = s.loc[common, 'ym_eff'].values
    ym_raw = s.loc[common, 'ym风速'].values
    hvg = s.loc[common, 'hollandvg'].values
    dist = s.loc[common, '场站距离_km'].values
    times_arr = common

    pw = find_main_window(ym_eff >= YM_THRESH, times_arr)
    tw = find_main_window(p >= PWR_THRESH, times_arr)
    if pw is None or tw is None: continue

    phys_start = pw[0]; phys_end = pw[1]
    true_start = tw[0]; true_end = tw[1]
    dT_start = (true_start - phys_start).total_seconds() / 3600
    dT_end = (true_end - phys_end).total_seconds() / 3600

    in_win = (times_arr >= phys_start) & (times_arr <= phys_end)
    pre_win = times_arr < phys_start

    features = {
        'name': name, 'yr': yr,
        'dT_start': dT_start, 'dT_end': dT_end,
        'd_min': d_min,
        'd_mean_in_win': dist[in_win].mean() if in_win.any() else dist.mean(),
        'ym_max': ym_raw.max(),
        'ym_mean_in_win': ym_raw[in_win].mean() if in_win.any() else 0,
        'ym_std': ym_raw.std(),
        'hvg_max': hvg.max(),
        'hvg_mean_in_win': hvg[in_win].mean() if in_win.any() else 0,
        'rmax_mean': sub['rmax'].mean(), 'rmax_std': sub['rmax'].std(),
        'b_mean': sub['b'].mean(), 'b_std': sub['b'].std(),
        'speed_mean': sub['移动速度'].mean(), 'speed_max': sub['移动速度'].max(),
        'p_max': p.max(),
        'p_pre_win_mean': p[pre_win].mean() if pre_win.any() else p[:len(p)//4].mean(),
        'p_std': p.std(),
        'total_span_h': (sub['target_time'].max()-sub['target_time'].min()).total_seconds()/3600,
    }
    rows.append(features)

df = pd.DataFrame(rows)
print(f'Typhoons with valid windows: {len(df)}')

feature_cols = [c for c in df.columns if c not in
    ['name','yr','dT_start','dT_end']]
X = df[feature_cols].values
y_start = df['dT_start'].values
y_end = df['dT_end'].values
names = df['name'].values

# Baseline: physics-only (no ML correction)
phys_mae_start = np.abs(y_start).mean()
phys_mae_end = np.abs(y_end).mean()
print(f'Physics baseline: |dT_start|={phys_mae_start:.1f}h, |dT_end|={phys_mae_end:.1f}h')

# ============================================================
# 2. LOO evaluation
# ============================================================
for model_name, model_cls in [
    ('XGBoost', XGBRegressor),
    ('RandomForest', RandomForestRegressor),
    ('GradientBoost', GradientBoostingRegressor),
]:
    loo = LeaveOneOut()
    ps_all, ts_all = [], []
    pe_all, te_all = [], []

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        ys_tr, ys_te = y_start[train_idx], y_start[test_idx]
        ye_tr, ye_te = y_end[train_idx], y_end[test_idx]

        m1 = model_cls(n_estimators=100, max_depth=3, random_state=42)
        m1.fit(X_tr, ys_tr)
        m2 = model_cls(n_estimators=100, max_depth=3, random_state=42)
        m2.fit(X_tr, ye_tr)

        ps_all.append(m1.predict(X_te)[0])
        pe_all.append(m2.predict(X_te)[0])
        ts_all.append(ys_te[0])
        te_all.append(ye_te[0])

    ps, ts = np.array(ps_all), np.array(ts_all)
    pe, te = np.array(pe_all), np.array(te_all)

    mae_s = np.abs(ps - ts).mean()
    mae_e = np.abs(pe - te).mean()
    gain_s = 100 * (1 - mae_s / phys_mae_start)
    gain_e = 100 * (1 - mae_e / phys_mae_end)
    under6_s = (np.abs(ps-ts) < 6).sum()
    under6_e = (np.abs(pe-te) < 6).sum()

    print(f'\n{model_name}:')
    print(f'  Start: phys={phys_mae_start:.0f}h -> ML={mae_s:.0f}h (gain {gain_s:.0f}%), <6h={under6_s}/{len(ps)}')
    print(f'  End:   phys={phys_mae_end:.0f}h -> ML={mae_e:.0f}h (gain {gain_e:.0f}%), <6h={under6_e}/{len(pe)}')

print('\nDone!')
