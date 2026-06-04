#!/usr/bin/env python3
"""
================================================================================
台风功率冲击窗口预测: 两阶段框架
================================================================================

问题定义
--------
  给定台风路径预测(含Holland/YanMeng参数模型输出), 预测风电场的功率冲击时间窗口:
    - t_rise_start:  功率开始显著爬升的时间
    - t_peak:        功率达到峰值的时间
    - t_fall_end:    功率回落到正常水平的时间

两阶段框架
----------
  阶段1 (物理粗定位):
    利用YanMeng参数模型计算场站等效风速 ym_eff(t) = ym(t) * exp(-dist(t)/500),
    阈值判定 ym_eff(t) > 2.0 m/s 得到物理预测窗口 [t0_phys, t1_phys].
    误差约 24-29h, 零训练成本, 纯物理驱动.

  阶段2 (ML精修, 数据积累后启用):
    用台风级特征 (距离、风速、Holland参数、运动特征等) 训练XGBoost/RandomForest,
    学习物理窗口的残差 Δt_start, Δt_end. 留一法(LOO)评估.
    当前14个台风样本不足以支撑ML精修, 框架已就位待更多数据.

数据来源
--------
  - 真实功率: data/power/ghdt_merged_15min_{year}.csv (15min分辨率)
  - 台风参数: data/typhoon/typhoon19-24complete_15min_within1000km_aug8.csv
    (含Holland/YanMeng模型输出: rmax, b, hollandvg, rmax-ym, b-ym, hollandvg-ym, ym风速)
  - 2019-2024年共31个台风事件

关键发现
--------
  - 28/31台风存在 "缓升→高位→缓降" 的功率冲击驼峰模式
  - ym(t)*exp(-d/500) 阈值能有效区分窗口内外功率 (平均区分度 82MW)
  - 物理窗口 start 误差约29h, end 误差约24h
  - 14个台风样本不足以支撑ML精修 (过拟合)

作者: [Your Name]
日期: 2026-05
================================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. 配置参数
# ============================================================================

# 物理窗口阈值
YM_EFF_THRESHOLD = 2.0    # 距离衰减后的YanMeng等效风速阈值 (m/s)
DIST_DECAY = 500           # 距离衰减系数 (km)
PWR_THRESHOLD = 100        # 功率阈值, 定义"显著冲击" (MW)

# 风电场参数
WIND_FARM_CAPACITY = 200.0  # 装机容量 (MW)

# 数据和输出路径
TYPHOON_PATH = "data/typhoon/typhoon19-24complete_15min_within1000km_aug8.csv"
POWER_GLOB = "data/power/ghdt_merged_15min_{year}.csv"


# ============================================================================
# 1. 数据加载与特征工程
# ============================================================================

def load_typhoon_data(typhoon_path: str) -> pd.DataFrame:
    """加载台风参数模型数据并构造物理衍生特征.

    台风参数数据包含Holland和YanMeng两个成熟参数模型的完整输出:
      - Holland模型: rmax (最大风速半径), b (形状参数), hollandvg (梯度风速)
      - YanMeng模型: rmax-ym, b-ym, hollandvg-ym, ym风速 (地表风速)
      - 基础气象: 风速, 气压, 台风等级, 移动速度, 移动方向
      - 几何关系: 场站距离_km, 方位角, 相对方位角

    物理特征构造:
      ym_eff = ym风速 * exp(-场站距离_km / DIST_DECAY)
      → 距离衰减后的等效地表风速. 物理含义: 台风在远处的风到达场站时已减弱.
      → 衰减系数500km基于台风典型风场尺度(rmax通常30-80km,外延数百km).
    """
    ty = pd.read_csv(typhoon_path, encoding='utf-8-sig')
    ty['target_time'] = pd.to_datetime(ty['target_time'])

    # 构造距离衰减等效风速
    ty['ym_eff'] = ty['ym风速'] * np.exp(-ty['场站距离_km'] / DIST_DECAY)
    ty['hvg_eff'] = ty['hollandvg'] * np.exp(-ty['场站距离_km'] / DIST_DECAY)

    return ty


def load_power_window(typhoon_sub: pd.DataFrame, year: int,
                      margin_hours: int = 48) -> pd.Series:
    """加载单个台风前后扩展窗口内的真实功率数据.

    窗口 = [台风首次进入1000km - margin, 台风最后退出1000km + margin]
    扩展margin确保捕获完整的功率升降过程(驼峰前后都需要基线功率).
    """
    t0 = typhoon_sub['target_time'].min() - pd.Timedelta(hours=margin_hours)
    t1 = typhoon_sub['target_time'].max() + pd.Timedelta(hours=margin_hours)

    power_path = POWER_GLOB.format(year=year)
    ghdt = pd.read_csv(power_path)
    ghdt['time'] = pd.to_datetime(ghdt['time'])
    mask = (ghdt['time'] >= t0) & (ghdt['time'] <= t1)

    return ghdt[mask].set_index('time')['VALUE']


def find_main_window(mask: np.ndarray, times: pd.DatetimeIndex):
    """从布尔掩码中提取最长连续True段作为主窗口.

    台风可能产生多个离散的功率峰值(如双驼峰), 取最长段作为主窗口.
    """
    windows = []
    in_win = False
    start = None

    for i, m in enumerate(mask):
        if m and not in_win:
            start = times[i]
            in_win = True
        elif not m and in_win:
            windows.append((start, times[i - 1]))
            in_win = False
    if in_win:
        windows.append((start, times[-1]))

    if not windows:
        return None
    # 返回持续时间最长的窗口
    return max(windows, key=lambda w: (w[1] - w[0]).total_seconds())


def extract_typhoon_features(typhoon_sub: pd.DataFrame, p: np.ndarray,
                              ym_eff: np.ndarray, ym_raw: np.ndarray,
                              hvg: np.ndarray, dist: np.ndarray,
                              times: pd.DatetimeIndex,
                              phys_window: tuple,
                              true_window: tuple) -> dict:
    """从单个台风事件中提取特征, 用于ML精修阶段.

    特征分为四组:
      1. 距离特征: 最近距离, 窗口内平均距离
      2. 风速特征: 最大ym风速, 窗口内平均ym风速, ym风速标准差
      3. 台风参数特征: rmax均值/标准差, b均值/标准差 (Holland模型)
      4. 运动特征: 平均移动速度, 最大移动速度
      5. 功率特征: 最大功率, 冲击前基线功率
      6. 时间特征: 台风总影响时长

    目标变量:
      dT_start = 真实窗口开始 - 物理窗口开始 (小时)
      dT_end   = 真实窗口结束 - 物理窗口结束 (小时)
    """
    phys_start, phys_end = phys_window
    true_start, true_end = true_window

    # 物理窗口内外的数据索引
    in_phys_win = (times >= phys_start) & (times <= phys_end)
    pre_phys_win = times < phys_start

    # 功率特征
    p_max_val = p.max()
    p_std_val = p.std()
    pre_base = p[pre_phys_win].mean() if pre_phys_win.any() else p[:max(1, len(p)//4)].mean()

    return {
        # --- 目标变量 (ML要预测的) ---
        'dT_start_h': (true_start - phys_start).total_seconds() / 3600,
        'dT_end_h': (true_end - phys_end).total_seconds() / 3600,
        'phys_dur_h': (phys_end - phys_start).total_seconds() / 3600,
        'true_dur_h': (true_end - true_start).total_seconds() / 3600,

        # --- 距离特征 ---
        'dist_min_km': typhoon_sub['场站距离_km'].min(),
        'dist_mean_in_win_km': dist[in_phys_win].mean() if in_phys_win.any() else dist.mean(),

        # --- 风速特征 (YanMeng地表风) ---
        'ym_max_ms': ym_raw.max(),
        'ym_mean_in_win_ms': ym_raw[in_phys_win].mean() if in_phys_win.any() else 0.0,
        'ym_std_ms': ym_raw.std(),

        # --- 风速特征 (Holland梯度风) ---
        'hvg_max_ms': hvg.max(),
        'hvg_mean_in_win_ms': hvg[in_phys_win].mean() if in_phys_win.any() else 0.0,

        # --- Holland参数模型特征 ---
        # rmax: 最大风速半径, 描述台风核心大小
        # b: 形状参数, 描述风场宽度 (b小→宽风场, b大→窄风场)
        'rmax_mean_km': typhoon_sub['rmax'].mean(),
        'rmax_std_km': typhoon_sub['rmax'].std(),
        'b_mean': typhoon_sub['b'].mean(),
        'b_std': typhoon_sub['b'].std(),

        # --- 运动特征 ---
        'speed_mean_kmh': typhoon_sub['移动速度'].mean(),
        'speed_max_kmh': typhoon_sub['移动速度'].max(),

        # --- 功率特征 ---
        'p_max_mw': p_max_val,
        'p_pre_win_baseline_mw': pre_base,
        'p_std_mw': p_std_val,

        # --- 时间特征 ---
        'total_span_h': (typhoon_sub['target_time'].max() - typhoon_sub['target_time'].min()).total_seconds() / 3600,
    }


# ============================================================================
# 2. 阶段1: 物理窗口预测
# ============================================================================

def stage1_physics_window(ty: pd.DataFrame) -> pd.DataFrame:
    """阶段1: 基于YanMeng参数模型的物理窗口预测.

    方法:
      对每个台风, 用距离衰减后的YanMeng地表风速 ym_eff(t) 判断窗口:
        ym_eff(t) = ym风速(t) * exp(-场站距离(t) / 500km)
        预测窗口 = {t | ym_eff(t) >= 2.0 m/s}

    物理原理:
      - YanMeng地表风速(ym风速)是参数模型计算的场站位置理论风速
      - 距离衰减因子 exp(-d/500) 修正了"远距离台风风到达场站已减弱"的物理事实
      - 阈值2.0 m/s通过实验确定: 此阈值下功率区分度(窗内vs窗外)最大

    输出:
      DataFrame, 每行一个台风, 含预测窗口、真实窗口、误差指标
    """
    results = []

    for tid, sub in ty.groupby('台风编号'):
        sub = sub.sort_values('target_time')
        name = sub['台风中文名称'].iloc[0]
        year = int(sub['target_time'].dt.year.mode().iloc[0])

        # 加载真实功率数据
        try:
            power = load_power_window(sub, year)
        except FileNotFoundError:
            continue

        # 对齐功率和台风参数的时间索引
        params = sub.set_index('target_time')
        common_times = power.index.intersection(params.index)
        if len(common_times) < 48:  # 至少12小时数据
            continue

        p = power.loc[common_times].values
        ym_eff = params.loc[common_times, 'ym_eff'].values
        ym_raw = params.loc[common_times, 'ym风速'].values
        hvg = params.loc[common_times, 'hollandvg'].values
        dist = params.loc[common_times, '场站距离_km'].values
        times_arr = common_times

        # 物理窗口 = ym_eff超过阈值的连续时间段
        pred_mask = ym_eff >= YM_EFF_THRESHOLD
        phys_win = find_main_window(pred_mask, times_arr)

        # 真实窗口 = 功率超过阈值的连续时间段
        true_mask = p >= PWR_THRESHOLD
        true_win = find_main_window(true_mask, times_arr)

        if phys_win is None or true_win is None:
            continue

        # 计算误差
        dt_start = (true_win[0] - phys_win[0]).total_seconds() / 3600
        dt_end = (true_win[1] - phys_win[1]).total_seconds() / 3600
        phys_dur = (phys_win[1] - phys_win[0]).total_seconds() / 3600
        true_dur = (true_win[1] - true_win[0]).total_seconds() / 3600

        # IoU (Intersection over Union): 窗口重叠率
        overlap_sec = max(0, (min(phys_win[1], true_win[1]) - max(phys_win[0], true_win[0])).total_seconds())
        union_sec = (max(phys_win[1], true_win[1]) - min(phys_win[0], true_win[0])).total_seconds()
        iou = overlap_sec / union_sec if union_sec > 0 else 0.0

        results.append({
            'typhoon_name': name,
            'year': year,
            'd_min_km': sub['场站距离_km'].min(),
            'phys_start': phys_win[0],
            'phys_end': phys_win[1],
            'true_start': true_win[0],
            'true_end': true_win[1],
            'phys_dur_h': phys_dur,
            'true_dur_h': true_dur,
            'dt_start_h': dt_start,
            'dt_end_h': dt_end,
            'iou': iou,
            'ym_max_in_win': ym_raw[pred_mask].max() if pred_mask.any() else 0,
            'p_max_mw': p.max(),
        })

        # 提取ML特征 (为阶段2准备)
        feat = extract_typhoon_features(sub, p, ym_eff, ym_raw,
                                         hvg, dist, times_arr,
                                         phys_win, true_win)
        results[-1].update({f'feat_{k}': v for k, v in feat.items()})

    return pd.DataFrame(results)


# ============================================================================
# 3. 阶段2: ML精修 (数据积累后启用)
# ============================================================================

def stage2_ml_refinement(df: pd.DataFrame) -> dict:
    """阶段2: 用ML模型学习物理窗口的残差修正.

    动机:
      物理窗口虽然有效(28/31台风有驼峰, 区分度82MW), 但仍有24-29h误差.
      这些残差可能由台风个体差异(风场结构、移动速度等)导致,
      可以用台风级特征训练ML模型来预测残差方向.

    当前局限:
      仅有14个有效台风, LOO训练时每折只有13个训练样本.
      在15+个特征的回归任务中严重过拟合, ML不优于物理基线.
      此函数保留完整框架待更多台风数据积累.

    输入:
      df: stage1输出, 每行一个台风, 含所有特征和目标

    返回:
      dict: 各模型在LOO下的MAE对比
    """
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from xgboost import XGBRegressor
        from sklearn.model_selection import LeaveOneOut
    except ImportError:
        print("[Stage2] sklearn/xgboost not installed. Skipping ML refinement.")
        return {}

    # 筛选特征列
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    feat_names = [c.replace('feat_', '') for c in feat_cols]

    # 排除目标列
    exclude = ['dT_start_h', 'dT_end_h', 'phys_dur_h', 'true_dur_h']
    feat_cols = [c for c in feat_cols
                 if not any(c.endswith('_' + e) for e in ['dT_start_h', 'dT_end_h', 'phys_dur_h', 'true_dur_h'])]

    X = df[feat_cols].values
    y_start = df['dt_start_h'].values
    y_end = df['dt_end_h'].values

    n_samples = len(df)
    phys_mae_start = np.abs(y_start).mean()
    phys_mae_end = np.abs(y_end).mean()
    print(f"\n[Stage2] Samples: {n_samples}, Features: {len(feat_cols)}")
    print(f"[Stage2] Physics baseline MAE: start={phys_mae_start:.1f}h, end={phys_mae_end:.1f}h")

    results = {}
    loo = LeaveOneOut()

    for model_name, model_cls in [
        ('XGBoost', XGBRegressor),
        ('RandomForest', RandomForestRegressor),
        ('GradientBoost', GradientBoostingRegressor),
    ]:
        preds_start, trues_start = [], []
        preds_end, trues_end = [], []

        for train_idx, test_idx in loo.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            ys_tr, ys_te = y_start[train_idx], y_start[test_idx]
            ye_tr, ye_te = y_end[train_idx], y_end[test_idx]

            # 为start和end分别训练独立的回归器
            m_start = model_cls(n_estimators=100, max_depth=3, random_state=42)
            m_start.fit(X_tr, ys_tr)

            m_end = model_cls(n_estimators=100, max_depth=3, random_state=42)
            m_end.fit(X_tr, ye_tr)

            preds_start.append(m_start.predict(X_te)[0])
            preds_end.append(m_end.predict(X_te)[0])
            trues_start.append(ys_te[0])
            trues_end.append(ye_te[0])

        ps, ts = np.array(preds_start), np.array(trues_start)
        pe, te = np.array(preds_end), np.array(trues_end)

        mae_s = np.abs(ps - ts).mean()
        mae_e = np.abs(pe - te).mean()

        print(f"  {model_name:15s}: start MAE={mae_s:.1f}h, end MAE={mae_e:.1f}h")
        results[model_name] = {'mae_start': mae_s, 'mae_end': mae_e}

    return results


# ============================================================================
# 4. 主流程
# ============================================================================

def main():
    print("=" * 70)
    print("Typhoon Power Impact Window Prediction: Two-Stage Framework")
    print("=" * 70)

    # ---- 加载数据 ----
    print("\n[1/3] Loading typhoon parametric data...")
    ty = load_typhoon_data(TYPHOON_PATH)
    print(f"  Loaded {len(ty)} records, {ty['台风编号'].nunique()} typhoon events")

    # ---- 阶段1: 物理窗口 ----
    print("\n[2/3] Stage 1: Physics-based window prediction...")
    print(f"  Method: ym_eff = ym_wind * exp(-dist/{DIST_DECAY}km)")
    print(f"  Threshold: ym_eff >= {YM_EFF_THRESHOLD} m/s")
    print(f"  Power threshold (ground truth): {PWR_THRESHOLD} MW")
    df = stage1_physics_window(ty)
    print(f"\n  Valid windows found: {len(df)} typhoons")

    # 汇总指标
    mae_start = df['dt_start_h'].abs().mean()
    mae_end = df['dt_end_h'].abs().mean()
    iou_mean = df['iou'].mean()
    n_iou_good = (df['iou'] > 0.5).sum()
    n_start_12h = (df['dt_start_h'].abs() < 12).sum()
    n_start_24h = (df['dt_start_h'].abs() < 24).sum()

    print(f"\n  Physics Window Accuracy Summary:")
    print(f"    Start time MAE:  {mae_start:.1f}h")
    print(f"    End time MAE:    {mae_end:.1f}h")
    print(f"    Mean IoU:        {iou_mean:.0%}")
    print(f"    IoU > 0.5:       {n_iou_good}/{len(df)}")
    print(f"    Start err < 12h: {n_start_12h}/{len(df)}")
    print(f"    Start err < 24h: {n_start_24h}/{len(df)}")

    # 展示几个典型案例
    print(f"\n  Example predictions (top 5 by IoU):")
    top5 = df.nlargest(5, 'iou')
    for _, row in top5.iterrows():
        print(f"    {row['typhoon_name']:6s} (d_min={row['d_min_km']:.0f}km): "
              f"start_err={row['dt_start_h']:+.0f}h, end_err={row['dt_end_h']:+.0f}h, "
              f"IoU={row['iou']:.0%}")

    # ---- 阶段2: ML精修 (可选) ----
    print("\n[3/3] Stage 2: ML refinement (experimental)...")
    ml_results = stage2_ml_refinement(df)
    if ml_results:
        best = min(ml_results.items(), key=lambda x: x[1]['mae_start'] + x[1]['mae_end'])
        print(f"\n  Best ML model: {best[0]}")
        print(f"  Note: With only {len(df)} samples, ML overfits. "
              "More typhoon data needed for this stage.")

    # ---- 保存 ----
    out_path = "results/typhoon_window_predictions.csv"
    os.makedirs("results", exist_ok=True)
    save_cols = ['typhoon_name', 'year', 'd_min_km',
                 'phys_start', 'phys_end', 'true_start', 'true_end',
                 'phys_dur_h', 'true_dur_h', 'dt_start_h', 'dt_end_h',
                 'iou', 'ym_max_in_win', 'p_max_mw']
    df[save_cols].to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n  Saved: {out_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
