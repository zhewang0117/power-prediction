#!/usr/bin/env python3
"""Per-stage comparison: ElasticNet vs Ridge vs KRR for all feature groups"""
import numpy as np, pandas as pd, math, sys
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,'.')
from evaluate_typhoon_gwo_elm_loto import build_dataset

HORIZON=96; CAPACITY=200.0; PN=1013.3; np.random.seed(42)

df=build_dataset('data/predictions/bilstm_pred_*.csv','data/typhoon/typhoon19-24complete_15min_improved.csv')
D=df.iloc[:,20].fillna(0).values.astype(float)
rmax=df['rmax_i'].fillna(0).values.astype(float); b=df['b_i'].fillna(0).values.astype(float)
atten=df['atten_w'].fillna(0).values.astype(float)
ym=df.iloc[:,39].fillna(0).values.astype(float); rel_az=df.iloc[:,22].fillna(0).values.astype(float)
ws=df.iloc[:,16].fillna(0).values.astype(float); pres=df.iloc[:,17].fillna(PN).values.astype(float)
level=df.iloc[:,15].fillna(0).values.astype(float); mvspd=df.iloc[:,23].fillna(0).values.astype(float)

df['dist_D_km']=D; df['D_over_Rmax']=np.where(rmax>0.5,D/rmax,0)
df['Rmax_km']=rmax; df['B_param']=b; df['atten_w']=atten
df['V_yanmeng_ms']=ym
df['Vtan_tangential']=ym*np.sin(np.radians(rel_az)); df['Vrad_radial']=ym*np.cos(np.radians(rel_az))
df['cos_rel_azimuth']=np.cos(np.radians(rel_az)); df['sin_rel_azimuth']=np.sin(np.radians(rel_az))
df['Vmax_ms']=ws; df['P_deficit_hPa']=PN-pres; df['typhoon_level']=level; df['Vmove_ms']=mvspd

all_events=df.groupby(df.columns[7])[df.columns[2]].min().sort_values().index.tolist()
shuffled=np.random.permutation(all_events)
tr_ids=list(shuffled[:23]); te_ids=list(shuffled[23:])
val_id=tr_ids[-1]; tr_no_val=[e for e in tr_ids if e!=val_id]
tr_df=df[df.iloc[:,7].isin(tr_no_val)]; va_df=df[df.iloc[:,7].isin([val_id])]; te_df=df[df.iloc[:,7].isin(te_ids)]
y_true=te_df['y_true_mw'].to_numpy(dtype=np.float32)
y_base=np.clip(te_df['y_pred_mw'].to_numpy(dtype=np.float32),0,CAPACITY)
d_test=te_df['dist_D_km'].values.astype(float)

stages=[('FAR(>350)',350,99999),('CORE(100-350)',100,350),('CLOSE(<100)',0,100)]

G1_F=['y_pred_mw','lead_norm']
G2_F=['y_pred_mw','lead_norm','V_yanmeng_ms','Vtan_tangential','Vrad_radial']
G3_F=['y_pred_mw','lead_norm','Rmax_km','B_param','atten_w']
G5_F=['y_pred_mw','lead_norm',
      'V_yanmeng_ms','Vtan_tangential','Vrad_radial',
      'Rmax_km','B_param','atten_w',
      'dist_D_km','D_over_Rmax','cos_rel_azimuth','sin_rel_azimuth',
      'Vmax_ms','P_deficit_hPa','typhoon_level','Vmove_ms']

tests=[
    ('G1_EN',    G1_F, 'elasticnet'),
    ('G1_Ridge', G1_F, 'ridge'),
    ('G1_KRR',   G1_F, 'krr'),
    ('G2_EN',    G2_F, 'elasticnet'),
    ('G2_Ridge', G2_F, 'ridge'),
    ('G2_KRR',   G2_F, 'krr'),
    ('G3_EN',    G3_F, 'elasticnet'),
    ('G3_Ridge', G3_F, 'ridge'),
    ('G3_KRR',   G3_F, 'krr'),
    ('G5_EN',    G5_F, 'elasticnet'),
    ('G5_KRR',   G5_F, 'krr'),
]

preds={}
for tname,feats,mtype in tests:
    X_tr=tr_df[feats].to_numpy(dtype=np.float32); y_tr=tr_df['delta_mw'].to_numpy(dtype=np.float32)
    X_va=va_df[feats].to_numpy(dtype=np.float32); y_va=va_df['delta_mw'].to_numpy(dtype=np.float32)
    X_te=te_df[feats].to_numpy(dtype=np.float32)
    scl=StandardScaler(); X_tr_s=scl.fit_transform(X_tr); X_va_s=scl.transform(X_va); X_te_s=scl.transform(X_te)

    best_a=1.0; best_l1=0.5; best_v=1e9
    if mtype=='elasticnet':
        for a in [0.01,0.05,0.1,0.5,1.0]:
            for l1 in [0.3,0.5,0.7]:
                m=ElasticNet(alpha=a,l1_ratio=l1,max_iter=5000)
                m.fit(X_tr_s,y_tr); vp=m.predict(X_va_s)
                if np.mean((vp-y_va)**2)<best_v: best_v=np.mean((vp-y_va)**2); best_a=a; best_l1=l1
        model=ElasticNet(alpha=best_a,l1_ratio=best_l1,max_iter=5000); model.fit(X_tr_s,y_tr)
    elif mtype=='ridge':
        for a in [0.1,1.0,10.0]:
            m=Ridge(alpha=a); m.fit(X_tr_s,y_tr)
            if np.mean((m.predict(X_va_s)-y_va)**2)<best_v: best_v=np.mean((m.predict(X_va_s)-y_va)**2); best_a=a
        model=Ridge(alpha=best_a); model.fit(X_tr_s,y_tr)
    else:
        if len(X_tr_s)>5000: idx=np.random.choice(len(X_tr_s),5000,replace=False); X_tr_k=X_tr_s[idx]; y_tr_k=y_tr[idx]
        else: X_tr_k=X_tr_s; y_tr_k=y_tr
        for a in [0.1,1.0,10.0]:
            m=KernelRidge(kernel='rbf',alpha=a,gamma=1.0/len(feats)); m.fit(X_tr_k,y_tr_k)
            if np.mean((m.predict(X_va_s)-y_va)**2)<best_v: best_v=np.mean((m.predict(X_va_s)-y_va)**2); best_a=a
        model=KernelRidge(kernel='rbf',alpha=best_a,gamma=1.0/len(feats)); model.fit(X_tr_k,y_tr_k)

    delta=model.predict(X_te_s)
    y_full=np.clip(y_base+delta,0,CAPACITY)
    best_lam=1.0; best_rmse=1e9; best_y=None
    for lam in [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        y_bl=np.clip(y_base+lam*(y_full-y_base),0,CAPACITY)
        r=math.sqrt(np.mean((y_true-y_bl)**2))
        if r<best_rmse: best_rmse=r; best_lam=lam; best_y=y_bl
    preds[tname+'_blend']=best_y
    preds[tname+'_full']=y_full
    print('.',end='',flush=True)
print()

g0_rmse=math.sqrt(np.mean((y_true-y_base)**2))
print('Test: %d samples, G0=%.2f'%(len(y_true),g0_rmse))

hdr='%-14s %8s %10s %10s %10s'%('Method','GLOBAL','FAR(>350)','CORE(100-350)','CLOSE(<100)')
# Show all blends first
print(hdr); print('-'*56)
for tname,_,_ in tests:
    yp=preds[tname+'_blend']
    row='%-14s'%tname
    row+='%8.2f'%math.sqrt(np.mean((y_true-yp)**2))
    for _,lo,hi in stages:
        m=(d_test>=lo)&(d_test<hi)
        row+='%10.2f'%(math.sqrt(np.mean((y_true[m]-yp[m])**2)) if m.sum()>10 else 0)
    print(row)

# Show full vs blend for key methods
print('\n=== Full(1.0) vs Blend(opt) for key methods ===')
key=['G1_KRR','G2_KRR','G3_KRR','G3_Ridge','G5_KRR']
hdr2='%-20s %8s %10s %10s %10s'%('Method','GLOBAL','FAR(>350)','CORE(100-350)','CLOSE(<100)')
print(hdr2); print('-'*62)
for tn in key:
    for sfx,lbl in [('_full','FULL'),('_blend','BLEND')]:
        yp=preds[tn+sfx]
        row='%-20s'%(tn+'_'+lbl)
        row+='%8.2f'%math.sqrt(np.mean((y_true-yp)**2))
        for _,lo,hi in stages:
            m=(d_test>=lo)&(d_test<hi)
            row+='%10.2f'%(math.sqrt(np.mean((y_true[m]-yp[m])**2)) if m.sum()>10 else 0)
        print(row)
    print()
