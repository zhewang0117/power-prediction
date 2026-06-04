
#!/usr/bin/env python3
"""
=============================================================================
Improved Parametric Typhoon Wind Field Model
=============================================================================

基于最新台风参数模型论文的6项关键改进，用于生成比原有模型更准确的风速估计。

改进项:
  1. Willoughby et al. (2006) Rmax 参数化 — 替代 Vickery (2005)
  2. Powell (2005) B 参数化 + Hubbert (1991) B — 替代经验 B 公式
  3. Zhong et al. (2023) 方位角依赖 Rmax(θ) & B(θ) — 引入风场非对称
  4. Powell et al. (2003) 风速依赖拖曳系数 Cd — 替代固定 z0
  5. Lin & Chavas (2012) 平移缩减因子 α=0.55 — 替代 α=1.0
  6. Ning et al. (2024) 移速依赖衰减 — 替代固定 500km

用法:
  python methods/improved_parametric_model.py

输入: data/typhoon/typhoon19-24complete_15min_within1000km_aug8.csv
输出: data/typhoon/typhoon19-24complete_15min_improved.csv
      (在原数据基础上追加改进后的参数列)

参考文献:
  - Willoughby, Darling & Rahn (2006), MWR, 134(4), 1102-1120
  - Powell, Vickery & Reinhold (2003), Nature, 422, 279-283
  - Lin & Chavas (2012), JGR, 117, D00V14
  - Zhong, Wei & Shang (2023), Natural Hazards, 117, 2325-2346
  - Chen, Zhao, Wang, Zhang & Yang (2025), Marine Sciences, 49(3), 1-14
  - Ning, Wang et al. (2024), Geomatics, Natural Hazards & Risk, 15(1)
=============================================================================
"""

import math, os, re, io, contextlib
import pandas as pd
import numpy as np
from typing import Optional, Tuple

# ============================================================================
# 0. 基本工具函数
# ============================================================================

def coriolis_parameter(lat: float) -> float:
    omega = 7.2921150e-5
    return 2.0 * omega * math.sin(math.radians(float(lat)))

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def calculate_bearing(lon1, lat1, lon2, lat2):
    """方位角: 从点1到点2, 0=北, 顺时针"""
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    dlon = lon2r - lon1r
    y = math.sin(dlon) * math.cos(lat2r)
    x = (math.cos(lat1r) * math.sin(lat2r) -
         math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


# ============================================================================
# 1. Rmax 参数化 — 改进项 #1: Willoughby (2006)
# ============================================================================

def estimate_vmax_gradient(row) -> float:
    """从最佳路径数据估计梯度层最大风速.

    CMA 最佳路径的'风速'是 2-min 平均近中心最大风速 (m/s, ~10m高度).
    梯度层风速 ≈ 地表风速 / 0.78 (PBL 风廓线缩减, Vickery et al. 2009).
    """
    try:
        vmax_sfc = float(row['风速'])
        if not math.isfinite(vmax_sfc) or vmax_sfc <= 0:
            # 回退: Atkinson-Holliday 风压关系
            pc = float(row.get('气压', 1010))
            vmax_sfc = 3.4 * (max(1010.0 - pc, 1.0)) ** 0.644
    except Exception:
        pc = float(row.get('气压', 1010))
        vmax_sfc = 3.4 * (max(1010.0 - pc, 1.0)) ** 0.644
    # 转换为梯度风 (除以缩减因子)
    return vmax_sfc / 0.78


def willoughby_rmax(vmax_gl: float, lat: float) -> float:
    """Willoughby et al. (2006) Eq.7a.

    Rmax = 46.4 * exp(-0.0155 * Vmax_gl) + 0.0169 * |latitude|

    Vmax_gl: 梯度层最大切向风速 (m/s)
    Chen et al. (2025) 验证此公式 + Powell B 为最优组合 (r≈0.90).
    """
    return 46.4 * math.exp(-0.0155 * max(vmax_gl, 5.0)) + 0.0169 * abs(lat)


def vickery_rmax(pc_hpa: float, lat: float) -> float:
    """Vickery (2005) — 原有公式, 保留用于对比."""
    dp = max(1010.0 - pc_hpa, 1.0)
    return math.exp(2.636 - 0.00005086 * dp**2 + 0.0394899 * abs(lat))


# ============================================================================
# 2. Holland B 参数化 — 改进项 #2: Powell (2005) & Hubbert (1991)
# ============================================================================

def B_powell(vmax_ms: float, pc_hpa: float) -> float:
    """Powell (2005): B 由最大风速和中心气压导出.

    B = Vmax² * ρ * e / Δp (气压单位: Pa)
    约束在 [1.0, 2.5] — Chen et al. (2025) 推荐.
    """
    dp_pa = max(1010.0 - pc_hpa, 1.0) * 100.0
    rho = 1.15
    vmax = max(vmax_ms, 5.0)
    B = (vmax**2 * rho * math.e) / dp_pa
    return max(1.0, min(2.5, B))


def B_hubbert(pc_hpa: float) -> float:
    """Hubbert et al. (1991): B = 1.5 + (980 - Pc) / 120."""
    return max(1.0, min(2.5, 1.5 + (980.0 - pc_hpa) / 120.0))


def B_holland_empirical(pc_hpa: float, lat: float) -> float:
    """原有经验 B 公式 (Vickery, 保留用于对比)."""
    dp = 1010.0 - pc_hpa
    return 1.2858 + 8.6396e-3 * dp - 8.7745e-3 * lat


# ============================================================================
# 3. 方位角依赖参数 — 改进项 #3: Zhong et al. (2023)
# ============================================================================

def azimuth_dependent_rmax(rmax_base: float, azimuth_deg: float,
                           heading_deg: float, move_speed_ms: float) -> float:
    """Zhong et al. (2023): Rmax 随方位角变化.

    Rmax(θ) = Rmax_base * [1 + A * cos(θ - θ_heading)]

    右前象限(北半球) Rmax 通常较大, 左后象限较小.
    不对称幅度 ~15-25% (基于 WRF 模拟 6 个浙江沿岸台风).
    """
    A_amp = 0.15 + 0.005 * max(move_speed_ms, 0.0)  # 移速越大不对称越强
    A_amp = min(A_amp, 0.30)
    theta_rel = math.radians(azimuth_deg - heading_deg)
    return rmax_base * (1.0 + A_amp * math.cos(theta_rel))


def azimuth_dependent_B(B_base: float, azimuth_deg: float,
                        heading_deg: float, move_speed_ms: float) -> float:
    """Zhong et al. (2023): Holland B 随方位角变化.

    B(θ) = B_base * [1 + A_B * cos(θ - θ_heading)]

    B 在台风前方较大 (梯度更强), 后方较小 (梯度更弱).
    """
    A_amp = 0.06 + 0.002 * max(move_speed_ms, 0.0)
    A_amp = min(A_amp, 0.15)
    theta_rel = math.radians(azimuth_deg - heading_deg)
    return B_base * (1.0 + A_amp * math.cos(theta_rel))


# ============================================================================
# 4. Holland 梯度风 (含改进平移修正) — 改进项 #5: Lin & Chavas (2012)
# ============================================================================

def holland_gradient_wind(dist_km: float, Rmax_km: float, B: float,
                          Pc_hpa: float, ambient_hpa: float, lat: float,
                          azimuth_deg: float, move_speed_ms: float,
                          move_dir_deg: float, rho: float = 1.15) -> float:
    """Holland (1980) 梯度风 + Lin & Chavas (2012) 平移非对称修正.

    关键改进: α = 0.55 (原代码 α = 1.0, 会高估平移效应).
    LC12 发现地表背景风仅为平移速度的 ~55%, 且逆时针旋转 ~20°.
    """
    r_m = max(dist_km * 1000.0, 1.0)
    rm_m = max(Rmax_km * 1000.0, 1.0)
    dp_pa = max((ambient_hpa - Pc_hpa) * 100.0, 0.0)
    f = coriolis_parameter(lat)

    # Lin & Chavas (2012): α = 0.55, β = 0° (旋转在合成时处理)
    alpha_lc = 0.55
    theta_rad = math.radians(azimuth_deg)
    beta_rad = math.radians(move_dir_deg)
    c0 = alpha_lc * move_speed_ms * math.sin(theta_rad - beta_rad)

    # Holland 压场梯度
    rm_r_b = (rm_m / r_m) ** B
    dp_dr = (dp_pa * B * (rm_m ** B) / (r_m ** (B + 1.0)) *
             math.exp(-rm_r_b))

    term1 = 0.5 * (c0 - f * r_m)
    inner = max(0.0, term1 * term1 + (r_m / rho) * dp_dr)
    Vg = term1 + math.sqrt(inner)

    if not math.isfinite(Vg) or Vg < 0:
        return 0.0
    return float(min(Vg, 150.0))


# ============================================================================
# 5. Yan Meng 地表风 (含改进 Cd) — 改进项 #4: Powell et al. (2003)
# ============================================================================

def drag_coefficient_powell(u10: float) -> float:
    """Powell et al. (2003): 风速依赖的拖曳系数.

    关键发现: Cd 在 >30 m/s 后不再单调增加, 而是趋于平稳甚至下降.
    传统固定 z0 的模型在台风内核会严重高估海面阻力.
    """
    if u10 <= 0:
        return 0.0010
    elif u10 <= 25:
        return (0.49 + 0.065 * u10) * 1e-3
    elif u10 <= 40:
        # 过渡区: Cd 增速放缓
        Cd25 = (0.49 + 0.065 * 25) * 1e-3
        return Cd25 + 0.00003 * (u10 - 25)
    else:
        # >40 m/s: Cd 趋于平稳 (Powell et al. 2003 Fig.3)
        Cd25 = (0.49 + 0.065 * 25) * 1e-3
        return min(Cd25 + 0.00003 * 15, 0.0028)


def surface_roughness_from_wind(u10: float) -> float:
    """Charnock 关系: z0 = α_c * u*² / g, 其中 u* = u10 * sqrt(Cd)."""
    Cd = drag_coefficient_powell(u10)
    u_star = u10 * math.sqrt(Cd)
    return max(0.0185 * u_star**2 / 9.81, 1e-5)


def yanmeng_gradient_solution(c_theta: float, f: float, r_m: float, rho: float,
                               dp_pa: float, rm_m: float, B: float
                               ) -> Tuple[float, float]:
    """Yan Meng 梯度风解析解.

    Returns: (vtheta_g, dvtheta_g/dr)
    """
    if r_m <= 0 or rm_m <= 0:
        return 0.0, 0.0

    # dp/dr (解析)
    rm_r = rm_m / r_m
    rm_r_b = rm_r ** B
    exp_term = math.exp(-rm_r_b)
    dp_dr = dp_pa * B * (rm_m ** B) / (r_m ** (B + 1.0)) * exp_term

    # d²p/dr² (解析)
    d2p_dr2 = dp_pa * B * (rm_m ** B) * (
        -(B + 1.0) / (r_m ** (B + 2.0)) * exp_term +
        (1.0 / (r_m ** (B + 1.0))) * exp_term *
        (-B * (rm_m ** B) / (r_m ** (B + 1.0)))
    )

    a_term = c_theta - f * r_m
    b_term = (r_m / rho) * dp_dr
    sqrt_term = max(0.25 * a_term**2 + b_term, 0.0)
    vtheta_g = 0.5 * a_term + math.sqrt(sqrt_term)

    dA_dr = -f
    dB_dr = dp_dr / rho + (r_m / rho) * d2p_dr2
    denom = math.sqrt(max(0.25 * a_term**2 + b_term, 0.0))

    dvtheta_g_dr = (-0.5 * f + (0.5 * (0.5 * a_term * dA_dr + dB_dr)) / denom
                    if denom > 1e-10 else -0.5 * f)

    if not math.isfinite(vtheta_g):
        vtheta_g = 0.0
    if not math.isfinite(dvtheta_g_dr) or abs(dvtheta_g_dr) > 1e5:
        dvtheta_g_dr = -0.5 * f

    return float(vtheta_g), float(dvtheta_g_dr)


def yanmeng_surface_iteration(vg: float, dvtheta_g_dr: float, r_m: float,
                               lat: float) -> Tuple[float, float, float]:
    """Yan Meng (1995) 地表风迭代求解, 使用风速依赖 Cd.

    关键改进:
      - z0(Cd) 每轮迭代根据当前地表风速更新 (原来固定 z0=0.001)
      - 更好地反映了高风速下海面阻力的物理特性

    Returns: (V_surface, V_theta_sfc, V_r_sfc)
    """
    f = coriolis_parameter(lat)
    k = 0.4
    A_bl = 11.4
    km = 100.0

    max_iter = 60
    tol = 1e-3
    relax = 0.5

    v_theta_p = 0.0
    v_r_p = 0.0

    for iteration in range(max_iter):
        v_theta_sfc = vg + v_theta_p
        v_r_sfc = v_r_p
        v_sfc = math.sqrt(v_theta_sfc**2 + v_r_sfc**2)

        # 每轮更新 z0 和 Cd (基于当前 v_sfc)
        z0 = surface_roughness_from_wind(max(v_sfc, 0.5))
        h = A_bl * (z0 ** 0.86)
        d = 0.75 * h
        z10 = 10.0
        gh = h + z10  # gradient height above surface

        log_arg = max((z10 + h - d) / max(z0, 1e-6), 1.0)
        Cd = (k ** 2) / (math.log(log_arg) ** 2)

        eps = 1e-10
        term1 = max(2.0 * vg / max(r_m, 1.0) + f, eps)
        term2 = max(dvtheta_g_dr + vg / max(r_m, 1.0) + f, eps)
        xi = math.sqrt(term2 / term1)
        lam = math.sqrt(math.sqrt(term1 * term2)) / math.sqrt(2.0 * km)

        chi = min(Cd * v_sfc / (km * max(lam, 1e-8)), 10.0)
        denom = 1.0 + (chi + 1.0)**2
        D1 = -(chi * (chi + 1.0) * vg) / denom
        D2 = (chi * vg) / denom

        lam_d = lam * gh
        exp_term = math.exp(lam_d)
        new_v_theta_p = exp_term * (D1 * math.cos(lam_d) + D2 * (-math.sin(lam_d)))
        new_v_r_p = -xi * exp_term * (D2 * math.cos(lam_d) - D1 * (-math.sin(lam_d)))

        v_theta_p = v_theta_p * (1.0 - relax) + new_v_theta_p * relax
        v_r_p = v_r_p * (1.0 - relax) + new_v_r_p * relax

        new_vs = math.sqrt((vg + v_theta_p)**2 + v_r_p**2)
        if iteration > 3 and abs(new_vs - v_sfc) < tol:
            return (float(new_vs),
                    float(vg + v_theta_p),
                    float(v_r_p))

    fallback = max(0.0, 0.7 * vg)
    return float(fallback), float(fallback), 0.0


# ============================================================================
# 6. 移速依赖衰减 — 改进项 #6: Ning et al. (2024)
# ============================================================================

def speed_dependent_distance_weight(dist_km: float, move_speed_ms: float) -> float:
    """Ning et al. (2024): 慢速台风需要更强的距离衰减.

    衰减因子 = exp(-dist / D_eff)
    D_eff = 500 * (move_speed / 5.0), clamped to [250, 1000]

    R²=0.54 线性关系: 移速 ↔ 衰减因子.
    """
    s = max(move_speed_ms, 1.0)
    D_eff = 500.0 * s / 5.0
    D_eff = max(250.0, min(1000.0, D_eff))
    return math.exp(-dist_km / D_eff)


# ============================================================================
# 7. 核心计算: 单行数据处理
# ============================================================================

def compute_improved_winds(row, site_lat: float, site_lon: float,
                           ambient: float = 1010.0,
                           move_speed_ms: float = 0.0,
                           move_dir_deg: float = 0.0,
                           theta_deg: Optional[float] = None
                           ) -> Tuple[float, ...]:
    """对单行台风数据计算所有改进后的参数风场量.

    Returns (12 values):
      rmax_w        — Willoughby Rmax (km)
      B_p           — Powell B
      Vg_holland    — Holland 梯度风 (含 LC12 平移修正)
      rmax_ym       — YM Rmax (Willoughby)
      B_ym          — YM B (Hubbert)
      Vg_ym         — YM 梯度风
      V_ym_sfc      — YM 地表风 (全距离)
      V_ym_250      — YM 地表风 (仅 ≤250km)
      Vg_h_az       — Holland 梯度风 (方位角依赖 Rmax+B)
      rmax_az       — 方位角依赖 Rmax
      B_az          — 方位角依赖 B
      atten_w       — 移速依赖衰减权重
    """
    nan = float('nan')

    try:
        clat = float(row['纬度'])
        clon = float(row['经度'])
        Pc  = float(row['气压'])
    except Exception:
        return (nan,) * 12

    if not (math.isfinite(clat) and math.isfinite(clon) and math.isfinite(Pc)):
        return (nan,) * 12

    # 距离
    try:
        if '场站距离_km' in row.index and pd.notna(row.get('场站距离_km')):
            dist_km = float(row['场站距离_km'])
        else:
            dist_km = haversine_km(clon, clat, site_lon, site_lat)
    except Exception:
        dist_km = haversine_km(clon, clat, site_lon, site_lat)
    dist_km = max(dist_km, 1e-6)

    # 方位角: 台风中心 → 站点
    try:
        azimuth = float(theta_deg) if theta_deg is not None else calculate_bearing(
            clon, clat, site_lon, site_lat)
    except Exception:
        azimuth = 0.0

    # 估计 Vmax_gl (梯度风)
    vmax_gl = estimate_vmax_gradient(row)

    # ---- Rmax (Willoughby) ----
    rmax_w = willoughby_rmax(vmax_gl, clat)

    # ---- B (Powell + Hubbert) ----
    try:
        vmax_sfc = float(row['风速']) if math.isfinite(float(row['风速'])) else vmax_gl * 0.78
    except Exception:
        vmax_sfc = vmax_gl * 0.78
    B_p = B_powell(vmax_sfc, Pc)
    B_h = B_hubbert(Pc)

    # ---- 方位角依赖 Rmax & B ----
    rmax_az = azimuth_dependent_rmax(rmax_w, azimuth, move_dir_deg, move_speed_ms)
    B_az = azimuth_dependent_B(B_h, azimuth, move_dir_deg, move_speed_ms)

    # ---- Holland 梯度风 ----
    Vg_h = holland_gradient_wind(dist_km, rmax_w, B_p,
                                  Pc, ambient, clat, azimuth,
                                  move_speed_ms, move_dir_deg)

    Vg_h_az = holland_gradient_wind(dist_km, rmax_az, B_az,
                                     Pc, ambient, clat, azimuth,
                                     move_speed_ms, move_dir_deg)

    # ---- Yan Meng 风场 ----
    r_m = dist_km * 1000.0
    rm_m = rmax_w * 1000.0
    dp_pa = max((ambient - Pc) * 100.0, 0.0)
    f = coriolis_parameter(clat)
    c_theta = move_speed_ms * math.sin(math.radians(azimuth - move_dir_deg))
    # Lin & Chavas 缩减
    c_theta *= 0.55

    vg_ym, dvg_dr = yanmeng_gradient_solution(
        c_theta, f, max(r_m, 1.0), 1.15, dp_pa, max(rm_m, 1.0), B_h)

    with contextlib.redirect_stdout(io.StringIO()):
        V_ym_sfc, _, _ = yanmeng_surface_iteration(
            vg_ym, dvg_dr, max(r_m, 1.0), clat)

    V_ym_250 = V_ym_sfc if dist_km <= 250.0 else nan

    # ---- YM 梯度风 ----
    Vg_ym = holland_gradient_wind(dist_km, rmax_w, B_h,
                                   Pc, ambient, clat, azimuth,
                                   move_speed_ms, move_dir_deg)

    # ---- 移速依赖衰减 ----
    atten_w = speed_dependent_distance_weight(dist_km, move_speed_ms)

    return (
        rmax_w,     # rmax (Willoughby)
        B_p,        # b   (Powell)
        Vg_h,       # hollandvg
        rmax_w,     # rmax-ym (用 Willoughby)
        B_h,        # b-ym    (用 Hubbert)
        Vg_ym,      # hollandvg-ym
        V_ym_sfc,   # ym风速
        V_ym_250,   # ym风速(<=250km)
        Vg_h_az,    # hollandvg_az
        rmax_az,    # rmax_az
        B_az,       # b_az
        atten_w,    # attenuation_weight
    )


# ============================================================================
# 8. 文件级处理 (兼容原 add_station_wind_columns_single.py 接口)
# ============================================================================

def compute_row_metrics(row, lat_col, lon_col, p_col, dist_col,
                         site_lat, site_lon, ambient=1010.0,
                         move_speed_ms=0.0, move_dir_deg=0.0,
                         theta_deg=None):
    """与原接口兼容的计算函数.

    返回 12 个值 (原8个 + 新增4个), 可直接替代原有 compute_row_metrics.
    """
    # 构造与 compute_improved_winds 兼容的 row
    row_dict = {
        '纬度': float(row[lat_col]),
        '经度': float(row[lon_col]),
        '气压': float(row[p_col]),
        '场站距离_km': (float(row[dist_col]) if dist_col and pd.notna(row.get(dist_col))
                       else haversine_km(float(row[lon_col]), float(row[lat_col]),
                                         site_lon, site_lat)),
    }
    # 尝试读取风速列
    for c in ('风速', 'wind', 'Wind'):
        if c in row.index:
            row_dict['风速'] = float(row[c])
            break
    if '风速' not in row_dict:
        row_dict['风速'] = float('nan')

    s = pd.Series(row_dict)
    return compute_improved_winds(
        s, site_lat, site_lon, ambient,
        move_speed_ms, move_dir_deg, theta_deg)


# ============================================================================
# 9. 主流程: 处理台风 CSV 并输出对比统计
# ============================================================================

def process_typhoon_csv(input_path: str, output_path: str):
    """读取包含原始参数列的台风数据, 计算改进后的风场, 输出增强 CSV."""
    print(f"读取: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    df['target_time'] = pd.to_datetime(df['target_time'], errors='coerce')

    # 检测站坐标 (假设所有行同一站点)
    if '场站距离_km' in df.columns and '经度' in df.columns and '纬度' in df.columns:
        # 根据最近距离反推站点经纬度
        closest = df.loc[df['场站距离_km'].idxmin()]
        # 站点经纬度需要用反向方位角推算, 简化处理: 直接使用台风中心坐标作为参考
        site_lat = float(closest['纬度'])
        site_lon = float(closest['经度'])
        print(f"参考站点位置: ({site_lat:.2f}N, {site_lon:.2f}E) "
              f"(最近距离={closest['场站距离_km']:.0f}km)")
    else:
        raise ValueError("缺少必要的经纬度和距离列")

    n = len(df)
    results = {
        'rmax_i':       [0.0]*n, 'b_i':         [0.0]*n,
        'hollandvg_i':  [0.0]*n, 'rmax_ym_i':   [0.0]*n,
        'b_ym_i':       [0.0]*n, 'hollandvg_ym_i': [0.0]*n,
        'ym风速_i':     [0.0]*n, 'ym风速250_i': [0.0]*n,
        'hollandvg_az': [0.0]*n, 'rmax_az':     [0.0]*n,
        'b_az':         [0.0]*n, 'atten_w':     [0.0]*n,
    }

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 5000 == 0:
            print(f"  处理进度: {i}/{n} ({100*i/n:.0f}%)")

        spd = float(row.get('移动速度(m/s)', 0)) if pd.notna(row.get('移动速度(m/s)')) else 0.0
        ddir = float(row.get('移动方向(°)', 0)) if pd.notna(row.get('移动方向(°)')) else 0.0
        theta = float(row.get('方位角', float('nan'))) if pd.notna(row.get('方位角')) else None

        vals = compute_improved_winds(
            row, site_lat, site_lon, 1010.0, spd, ddir, theta)

        for j, key in enumerate(results.keys()):
            results[key][i] = vals[j]

    # 写入 DataFrame
    for key, arr in results.items():
        df[key] = arr

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"保存: {output_path} ({len(df)} rows)")

    return df


def print_comparison(df: pd.DataFrame):
    """打印改进前后风速统计对比."""
    print("\n" + "=" * 70)
    print("改进前后参数风速对比")
    print("=" * 70)

    # 找出有原始参数列的 rows
    has_orig = all(c in df.columns for c in ['rmax', 'b', 'hollandvg', 'ym风速'])

    if has_orig:
        cols_orig = ['rmax', 'b', 'hollandvg', 'hollandvg-ym', 'ym风速']
        cols_impr = ['rmax_i', 'b_i', 'hollandvg_i', 'hollandvg_ym_i', 'ym风速_i']
        labels     = ['Rmax (km)', 'B', 'Vg_holland (m/s)',
                      'Vg_ym (m/s)', 'V_ym_sfc (m/s)']

        print(f"\n{'变量':20s} {'原始均值':>10s} {'改进均值':>10s} "
              f"{'差异':>8s} {'原始std':>8s} {'改进std':>8s}")
        print("-" * 70)
        for co, ci, lb in zip(cols_orig, cols_impr, labels):
            orig = df[co].dropna()
            impr = df[ci].dropna()
            if len(orig) > 0 and len(impr) > 0:
                print(f"{lb:20s} {orig.mean():10.3f} {impr.mean():10.3f} "
                      f"{impr.mean()-orig.mean():+8.3f} {orig.std():8.3f} {impr.std():8.3f}")

    # 风速差异分布
    if has_orig and 'ym风速_i' in df.columns:
        d_ym = df['ym风速_i'] - df['ym风速']
        valid = d_ym.dropna()
        if len(valid) > 0:
            print(f"\nYan Meng 地表风速差异 (改进-原始):")
            print(f"  Mean={valid.mean():+.3f} m/s, Std={valid.std():.3f} m/s")
            print(f"  P10={valid.quantile(0.10):+.3f}, "
                  f"P50={valid.quantile(0.50):+.3f}, "
                  f"P90={valid.quantile(0.90):+.3f}")

    if 'hollandvg_az' in df.columns and 'hollandvg_i' in df.columns:
        d_az = df['hollandvg_az'] - df['hollandvg_i']
        valid_az = d_az.dropna()
        if len(valid_az) > 0:
            print(f"\n方位角依赖修正幅度 (hollandvg_az - hollandvg):")
            print(f"  Mean={valid_az.mean():+.3f} m/s, Std={valid_az.std():.3f} m/s")
            print(f"  |Δ|>1 m/s: {(valid_az.abs()>1).mean()*100:.1f}%")

    # 衰减权重统计
    if 'atten_w' in df.columns:
        aw = df['atten_w'].dropna()
        if len(aw) > 0:
            print(f"\n移速依赖衰减权重:")
            print(f"  Mean={aw.mean():.4f}, Std={aw.std():.4f}")
            print(f"  Range: [{aw.min():.4f}, {aw.max():.4f}]")

    print()


def _estimate_motion_for_group(g, time_col, lat_col, lon_col):
    """中心差分估算平移速度 (m/s) 与方向 (°)."""
    g = g.sort_values(by=time_col).copy()
    times = pd.to_datetime(g[time_col])
    lats = g[lat_col].astype(float).values
    lons = g[lon_col].astype(float).values
    n = len(g)
    speeds = [0.0] * n
    dirs = [0.0] * n

    for i in range(n):
        if 0 < i < n - 1:
            dt_h = (times.iloc[i+1] - times.iloc[i-1]).total_seconds() / 3600.0
            if dt_h <= 0:
                dt_h = (times.iloc[min(i+1,n-1)] - times.iloc[i]).total_seconds() / 3600.0
                ref_i0, ref_i1 = i, min(i+1, n-1)
            else:
                ref_i0, ref_i1 = i-1, i+1
        elif i == 0 and n > 1:
            dt_h = (times.iloc[1] - times.iloc[0]).total_seconds() / 3600.0
            ref_i0, ref_i1 = 0, 1
        elif i == n - 1 and n > 1:
            dt_h = (times.iloc[n-1] - times.iloc[n-2]).total_seconds() / 3600.0
            ref_i0, ref_i1 = n-2, n-1
        else:
            dt_h = 1.0
            ref_i0, ref_i1 = i, i

        d_km = haversine_km(lons[ref_i0], lats[ref_i0],
                            lons[ref_i1], lats[ref_i1])
        speed_ms = (d_km * 1000.0) / max(dt_h * 3600.0, 1.0)
        direction = calculate_bearing(lons[ref_i0], lats[ref_i0],
                                      lons[ref_i1], lats[ref_i1])
        speeds[i] = speed_ms
        dirs[i] = direction

    return pd.Series(speeds, index=g.index), pd.Series(dirs, index=g.index)


def process_station_files(data_folder: str):
    """批量处理站点台风 Excel 文件 (兼容原 add_station_wind_columns_single.py 接口).

    输出带 _improved.xlsx 后缀的文件.
    """
    if not os.path.isdir(data_folder):
        raise SystemExit(f"目录不存在: {data_folder}")

    files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
             if f.lower().endswith('.xlsx')
             and '_aug8' not in f.lower()
             and '_improved' not in f.lower()
             and '_with_wind' not in f.lower()]
    files.sort()

    if not files:
        print(f"在 {data_folder} 下没有找到可处理的 xlsx 文件")
        return

    for fp in files:
        print(f"\n处理: {os.path.basename(fp)}")
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            print(f"  读取失败: {e}")
            continue

        # 检测列
        lat_col = lon_col = p_col = dist_col = None
        for c in ('纬度', 'latTC', 'Lat', 'lat'):
            if c in df.columns: lat_col = c; break
        for c in ('经度', 'lonTC', 'Lon', 'lon'):
            if c in df.columns: lon_col = c; break
        for c in ('气压', 'mslp', 'Pc', '中心气压'):
            if c in df.columns: p_col = c; break
        for c in ('距离km', 'distance_km', 'distance', '场站距离_km'):
            if c in df.columns: dist_col = c; break
        time_col = id_col = None
        for c in ('时间', '当前台风时间', '当前时间', 'time', 'Time',
                  'date', 'datetime', 'DateTime', 'dateUTC'):
            if c in df.columns: time_col = c; break
        for c in ('编号', 'id', '台风编号', 'TyID', 'tfid', 'storm_id'):
            if c in df.columns: id_col = c; break

        if not all([lat_col, lon_col, p_col]):
            print(f"  跳过 (缺少关键列: 纬度/经度/气压)")
            continue

        if id_col is None:
            id_col = '__NO_ID__'
            df[id_col] = 0

        # 解析站点
        name = os.path.basename(fp)
        m = re.match(r"([A-Za-z0-9]+)_N([0-9.]+)E([0-9.]+)_\d+km", name)
        if m:
            site_lat = float(m.group(2))
            site_lon = float(m.group(3))
        else:
            site_lat = 32.86
            site_lon = 121.62

        # 按台风分组计算
        result_list = []
        for gid, g in df.groupby(id_col):
            g = g.copy()
            if time_col is not None:
                try:
                    g[time_col] = pd.to_datetime(g[time_col])
                except Exception:
                    pass

            if time_col is not None:
                spd_s, dir_s = _estimate_motion_for_group(g, time_col, lat_col, lon_col)
            else:
                spd_s = pd.Series([0.0]*len(g), index=g.index)
                dir_s = pd.Series([0.0]*len(g), index=g.index)

            try:
                theta_s = g.apply(
                    lambda r: calculate_bearing(
                        float(r[lon_col]), float(r[lat_col]), site_lon, site_lat), axis=1)
            except Exception:
                theta_s = pd.Series([float('nan')]*len(g), index=g.index)

            sub_vals = g.apply(
                lambda r: compute_row_metrics(
                    r, lat_col, lon_col, p_col, dist_col,
                    site_lat, site_lon, 1010.0,
                    float(spd_s.loc[r.name]),
                    float(dir_s.loc[r.name]),
                    float(theta_s.loc[r.name]) if pd.notna(theta_s.loc[r.name]) else None
                ), axis=1
            )
            result_list.append(sub_vals)

        vals = pd.concat(result_list).sort_index()

        # 12 列输出
        out_cols = ['rmax', 'b', 'hollandvg', 'rmax-ym', 'b-ym',
                    'hollandvg-ym', 'ym风速', 'ym风速(<=250km)',
                    'hollandvg_az', 'rmax_az', 'b_az', 'atten_w']
        for ci, cn in enumerate(out_cols):
            df[cn] = [t[ci] if isinstance(t, tuple) else float('nan') for t in vals]

        out_path = os.path.splitext(fp)[0] + '_improved.xlsx'
        try:
            df.to_excel(out_path, index=False)
            print(f"  -> {os.path.basename(out_path)} ({len(df)} rows)")
        except PermissionError:
            out_path2 = os.path.splitext(fp)[0] + '_improved_v2.xlsx'
            df.to_excel(out_path2, index=False)
            print(f"  -> {os.path.basename(out_path2)}")


# ============================================================================
# 10. 命令行入口
# ============================================================================

if __name__ == '__main__':
    import sys

    # 默认: 处理合并后的台风 CSV
    INPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         'data', 'typhoon',
                         'typhoon19-24complete_15min_within1000km_aug8.csv')
    OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          'data', 'typhoon',
                          'typhoon19-24complete_15min_improved.csv')

    if os.path.exists(INPUT):
        print("=" * 70)
        print("Improved Parametric Typhoon Wind Field Model")
        print("=" * 70)
        print()
        print("改进项:")
        print("  1. Willoughby (2006) Rmax — 替代 Vickery (2005)")
        print("  2. Powell (2005) B + Hubbert (1991) B — 替代经验B")
        print("  3. Zhong et al. (2023) 方位角依赖 Rmax(θ) & B(θ)")
        print("  4. Powell et al. (2003) 风速依赖 Cd — 替代固定 z0")
        print("  5. Lin & Chavas (2012) 平移缩减 α=0.55")
        print("  6. Ning et al. (2024) 移速依赖衰减")
        print()

        df_out = process_typhoon_csv(INPUT, OUTPUT)
        print_comparison(df_out)

        print("新增列:")
        print("  hollandvg_az — 方位角依赖 Holland 梯度风 (含非对称 Rmax+B)")
        print("  rmax_az     — 方位角依赖 Rmax")
        print("  b_az        — 方位角依赖 B")
        print("  atten_w     — 移速依赖距离衰减权重")
        print()
        print("完成! 下游脚本可使用 hollandvg_az / ym风速_i 等替代原参数列.")
    else:
        print(f"未找到输入文件: {INPUT}")
        print("尝试将脚本作为模块导入, 或直接调用 process_station_files()")

    # 可选: 也支持批量处理站点文件
    if len(sys.argv) > 1 and sys.argv[1] == '--stations':
        folder = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'Tydata', 'by_station')
        process_station_files(folder)
