import os
import re
import math
import pandas as pd
import io
import contextlib
from typing import Optional, Tuple

BASE = os.path.abspath(os.path.dirname(__file__))


def coriolis_parameter(lat: float) -> float:
    """计算科里奥利参数 f = 2 * omega * sin(lat)"""
    omega = 7.2921150e-5
    return 2.0 * omega * math.sin(math.radians(float(lat)))


def calculate_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """计算从点1到点2的地理方位角（0=北，顺时针）"""
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    bearing_rad = math.atan2(y, x)
    return (math.degrees(bearing_rad) + 360.0) % 360.0


def vickery_rmax(pc: float, lat: float) -> float:
    """Vickery 参数化：Rmax(km)"""
    delta_p = 1010.0 - float(pc)
    exponent = 2.636 - 0.00005086 * (delta_p ** 2) + 0.0394899 * abs(float(lat))
    return math.exp(exponent)


def calculate_B(pc: float, phi: float) -> float:
    """经验 Holland B 参数"""
    delta_p = 1010.0 - float(pc)
    return 1.2858 + 8.6396e-3 * delta_p - 8.7745e-3 * float(phi)


def ym_model_rmax(pc: float, lat: float) -> float:
    """YanMeng 方案配套 Rmax 参数化(km)"""
    delta_p = max(1010.0 - float(pc), 0.0)
    ln_rmax = 3.5 - 3e-5 * (delta_p ** 2) + 0.02 * abs(float(lat))
    return math.exp(ln_rmax)


def ym_model_B(pc: float, lat: float) -> float:
    """YanMeng 方案配套 Holland B 参数化"""
    delta_p = max(1010.0 - float(pc), 0.0)
    rmax = ym_model_rmax(pc, lat)
    return 1.4 + 0.001 * delta_p - 0.002 * rmax


def calculate_dp_dr(delta_p: float, rm: float, b: float, r: float) -> float:
    """Holland 压场一阶导数 dp/dr（Pa/m）"""
    if r <= 0:
        return 0.0
    rm_over_r = rm / r
    exp_term = math.exp(-(rm_over_r) ** b)
    return delta_p * b * (rm ** b) / (r ** (b + 1.0)) * exp_term


def calculate_d2p_dr2(delta_p: float, rm: float, b: float, r: float, dr: float = 1.0) -> float:
    """中心差分近似二阶导数 d2p/dr2（Pa/m^2）"""
    if r <= dr:
        return 0.0
    dp_dr_plus = calculate_dp_dr(delta_p, rm, b, r + dr)
    dp_dr_minus = calculate_dp_dr(delta_p, rm, b, r - dr)
    return (dp_dr_plus - dp_dr_minus) / (2.0 * dr)


def calculate_vtheta_g(c_theta: float, f: float, r: float, rho: float,
                       delta_p: float, rm: float, b: float) -> Tuple[float, float]:
    """计算 vtheta_g 与其径向导数 dvtheta_g/dr"""
    dp_dr = calculate_dp_dr(delta_p, rm, b, r)
    d2p_dr2 = calculate_d2p_dr2(delta_p, rm, b, r)

    a_term = c_theta - f * r
    b_term = (r / max(rho, 1e-8)) * dp_dr
    sqrt_term = max(0.25 * (a_term ** 2) + b_term, 0.0)
    vtheta_g = 0.5 * a_term + math.sqrt(sqrt_term)

    dA_dr = -f
    dB_dr = (1.0 / max(rho, 1e-8)) * dp_dr + (r / max(rho, 1e-8)) * d2p_dr2
    denom = math.sqrt(max(0.25 * (a_term ** 2) + b_term, 0.0))
    if denom < 1e-10:
        dvtheta_g_dr = -0.5 * f
    else:
        dvtheta_g_dr = -0.5 * f + (0.5 * (0.5 * a_term * dA_dr + dB_dr)) / denom

    if (not math.isfinite(vtheta_g)):
        vtheta_g = 0.0
    if (not math.isfinite(dvtheta_g_dr)) or abs(dvtheta_g_dr) > 1e5:
        dvtheta_g_dr = -0.5 * f
    return float(vtheta_g), float(dvtheta_g_dr)


def _yanmeng_surface_wind(vg: float, dvtheta_g_dr: float, r_m: float, params: dict) -> Tuple[float, float, float]:
    """YanMeng 迭代地表风速（返回 V_surface, v_theta_surface, v_r_surface）"""
    z0 = float(params.get('z0', 0.001))
    km = float(params.get('km', 100.0))
    A = float(params.get('A', 11.4))
    k = float(params.get('k', 0.4))
    f = float(params.get('f', 5e-5))

    h = A * (z0 ** 0.86)
    d = 0.75 * h
    z10 = 10.0
    gradient_height = h + z10

    log_arg = (z10 + h - d) / max(z0, 1e-6)
    Cd = (k ** 2) / (math.log(log_arg) ** 2)

    eps = 1e-10
    term1 = max(2.0 * vg / max(r_m, 1.0) + f, eps)
    term2 = max(dvtheta_g_dr + vg / max(r_m, 1.0) + f, eps)
    xi = math.sqrt(term2 / term1)
    lam = (term1 * term2) ** 0.25 / math.sqrt(2.0 * max(km, 1e-8))

    max_iter = 50
    tol = 1e-3
    relax = 0.5

    v_theta_prime = 0.0
    v_r_prime = 0.0

    for _ in range(max_iter):
        v_theta_surface = vg + v_theta_prime
        v_r_surface = v_r_prime
        v_surface = math.sqrt(v_theta_surface ** 2 + v_r_surface ** 2)

        chi = (Cd / (max(km, 1e-8) * max(lam, 1e-8))) * v_surface
        chi = min(chi, 10.0)
        denominator = 1.0 + (chi + 1.0) ** 2
        D1 = -(chi * (chi + 1.0) * vg) / denominator
        D2 = (chi * vg) / denominator

        lam_d = lam * gradient_height
        exp_term = math.exp(lam_d)
        cos_term = math.cos(lam_d)
        sin_term = -math.sin(lam_d)

        new_v_theta_prime = exp_term * (D1 * cos_term + D2 * sin_term)
        new_v_r_prime = -xi * exp_term * (D2 * cos_term - D1 * sin_term)

        v_theta_prime = v_theta_prime * (1.0 - relax) + new_v_theta_prime * relax
        v_r_prime = v_r_prime * (1.0 - relax) + new_v_r_prime * relax

        new_v_theta_surface = vg + v_theta_prime
        new_v_r_surface = v_r_prime
        if abs(new_v_theta_surface - v_theta_surface) < tol and abs(new_v_r_surface - v_r_surface) < tol:
            v_surface = math.sqrt(new_v_theta_surface ** 2 + new_v_r_surface ** 2)
            return float(v_surface), float(new_v_theta_surface), float(new_v_r_surface)

    # 兜底：未收敛时给保守估计
    fallback = max(0.0, 0.7 * vg)
    return float(fallback), float(fallback), 0.0


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def parse_station_from_filename(path: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    name = os.path.basename(path)
    # Expect like: station_N{lat}E{lon}_{radius}km.xlsx
    m = re.match(r"([A-Za-z0-9]+)_N([0-9.]+)E([0-9.]+)_\d+km", name)
    if not m:
        return None, None, None
    st = m.group(1)
    lat = float(m.group(2))
    lon = float(m.group(3))
    return st, lat, lon


def detect_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    lat_col = None
    lon_col = None
    p_col = None
    dist_col = None
    for c in ('纬度', 'latTC', 'Lat', 'lat'):
        if c in df.columns:
            lat_col = c
            break
    for c in ('经度', 'lonTC', 'Lon', 'lon'):
        if c in df.columns:
            lon_col = c
            break
    for c in ('气压', 'mslp', 'Pc', '中心气压'):
        if c in df.columns:
            p_col = c
            break
    for c in ('distance_km', '距离km', 'distance'):
        if c in df.columns:
            dist_col = c
            break
    return lat_col, lon_col, p_col, dist_col


def detect_time_id_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    time_col = None
    id_col = None
    for c in ('时间', '当前台风时间', '当前时间', 'time', 'Time', 'date', 'datetime', 'DateTime', 'dateUTC', '时间10min', 'time10min'):
        if c in df.columns:
            time_col = c
            break
    for c in ('编号', 'id', '台风编号', 'TyID', 'tfid', 'storm_id'):
        if c in df.columns:
            id_col = c
            break
    return time_col, id_col


def _movement_adjusted_gradient_wind(dist_km: float, Rmax_km: float, B: float, Pc_hpa: float,
                                     ambient_hpa: float, lat: float,
                                     theta_deg: float, move_speed_ms: float, move_dir_deg: float,
                                     rho: float = 1.2) -> float:
    """按 Holland 压场 + 平移非对称叠加，返回含移动效应的梯度风模长（m/s）。
    公式参考 yanmeng_wind 注释：V = 0.5*(c0 - f r) + sqrt( 0.25*(c0 - f r)^2 + (r/ρ) dp/dr )，其中 c0=-c*sin(theta-beta)
    """
    try:
        r_m = max(dist_km * 1000.0, 1.0)
        rm_m = max(Rmax_km * 1000.0, 1.0)
        delta_p_pa = max((ambient_hpa - Pc_hpa) * 100.0, 0.0)
        f = float(coriolis_parameter(lat))
        # 使用地理方位角直接计算，与 yanmeng_wind 中 -sin(theta_math - beta_math) 等价
        theta = math.radians(theta_deg)
        beta = math.radians(move_dir_deg)
        c0 = float(move_speed_ms) * math.sin(theta - beta)
        dp_dr = float(calculate_dp_dr(delta_p_pa, rm_m, float(B), r_m))  # Pa/m
        term1 = 0.5 * (c0 - f * r_m)
        inner = max(0.0, term1 * term1 + (r_m / max(rho, 1e-6)) * dp_dr)
        Vg = term1 + math.sqrt(inner)
        if not math.isfinite(Vg) or Vg < 0:
            Vg = 0.0
        return float(min(Vg, 150.0))
    except Exception:
        return math.nan


def compute_row_metrics(row, lat_col, lon_col, p_col, dist_col, site_lat, site_lon,
                        ambient=1010.0,
                        move_speed_ms: float = 0.0,
                        move_dir_deg: float = 0.0,
                        theta_deg: Optional[float] = None):
    try:
        clat = float(row[lat_col])
        clon = float(row[lon_col])
        Pc = float(row[p_col])
    except Exception:
        return (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    if not (math.isfinite(clat) and math.isfinite(clon) and math.isfinite(Pc)):
        return (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    # Distance and latitude for coriolis
    try:
        if dist_col and pd.notna(row[dist_col]):
            dist_km = float(row[dist_col])
        else:
            dist_km = haversine_km(clon, clat, site_lon, site_lat)
    except Exception:
        dist_km = haversine_km(clon, clat, site_lon, site_lat)
    dist_km = max(float(dist_km), 1e-6)

    # Default paramization (Vickery + empirical B)
    Rmax_km = float(vickery_rmax(Pc, clat))
    B_emp = float(calculate_B(Pc, clat))
    # 站点相对台风中心的方位角（从中心指向站点）
    try:
        theta_to_site = float(theta_deg if theta_deg is not None else calculate_bearing(clon, clat, site_lon, site_lat))
    except Exception:
        theta_to_site = 0.0
    # 叠加移动效应的 Holland 梯度风
    Vg_holland = _movement_adjusted_gradient_wind(dist_km, Rmax_km, B_emp, Pc, ambient, clat,
                                                  theta_to_site, move_speed_ms, move_dir_deg)

    # YanMeng parameterization
    Rmax_ym = float(ym_model_rmax(Pc, clat))
    B_ym = float(ym_model_B(Pc, clat))
    # Holland gradient using YM parameters
    Vg_holland_ym = _movement_adjusted_gradient_wind(dist_km, Rmax_ym, B_ym, Pc, ambient, clat,
                                                     theta_to_site, move_speed_ms, move_dir_deg)

    # YM surface wind (10m) via iteration based on YanMeng
    try:
        # Holland pressure field params
        r_m = dist_km * 1000.0
        rm_m = max(Rmax_ym * 1000.0, 1.0)
        delta_p_pa = max((ambient - Pc) * 100.0, 0.0)
        rho = 1.2
        f = float(coriolis_parameter(clat))
        # cθ：平移切向分量（与 yanmeng_wind 等价形式：-ms*sin(theta_math - beta_math) = ms*sin(theta_geo - beta_geo)）
        c_theta = float(move_speed_ms) * math.sin(math.radians(theta_to_site - move_dir_deg))
        vg, dvtheta_g_dr = calculate_vtheta_g(c_theta, f, max(r_m, 1.0), rho, delta_p_pa, rm_m, float(B_ym))
        # 抑制迭代内部的调试打印
        with contextlib.redirect_stdout(io.StringIO()):
            V_ym, _, _ = _yanmeng_surface_wind(vg, dvtheta_g_dr, max(r_m, 1.0), {'f': f, 'ρ': rho})
    except Exception:
        V_ym = math.nan
    # Conditional YM wind within 250km
    V_ym_250 = V_ym if dist_km <= 250.0 else math.nan

    # Return exactly 8 values per requirements
    return (
        Rmax_km,          # rmax
        B_emp,            # b
        Vg_holland,       # hollandvg (包含移动效应)
        Rmax_ym,          # rmax-ym
        B_ym,             # b-ym
        Vg_holland_ym,    # hollandvg using ym params (包含移动效应)
        V_ym,             # ym风速（迭代法）
        V_ym_250          # ym风速（<=250km）
    )


def _estimate_motion_for_group(g: pd.DataFrame, time_col: str, lat_col: str, lon_col: str) -> Tuple[pd.Series, pd.Series]:
    """中心差分估算平移速度(m/s)与方向(度，0=北？此处采用地理方位：0=北, 90=东)。
    我们使用 calculate_bearing（以正北为0度，顺时针）来得到方向。
    """
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
                dt_h = (times.iloc[min(i+1, n-1)] - times.iloc[i]).total_seconds() / 3600.0
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
        d_km = haversine_km(lons[ref_i0], lats[ref_i0], lons[ref_i1], lats[ref_i1])
        speed_ms = (d_km * 1000.0) / max(dt_h * 3600.0, 1.0)
        direction = calculate_bearing(lons[ref_i0], lats[ref_i0], lons[ref_i1], lats[ref_i1])
        speeds[i] = float(speed_ms)
        dirs[i] = float(direction)
    return pd.Series(speeds, index=g.index), pd.Series(dirs, index=g.index)


def process_file(path: str):
    station, site_lat, site_lon = parse_station_from_filename(path)
    if station is None:
        print(f'Skip (无法解析站点/坐标): {path}')
        return
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f'读取失败 {path}: {e}')
        return
    lat_col, lon_col, p_col, dist_col = detect_cols(df)
    if not all([lat_col, lon_col, p_col]):
        print(f'Skip (缺少关键列 经度/纬度/气压): {path}')
        return
    time_col, id_col = detect_time_id_cols(df)
    if time_col is None:
        # 若无时间列，也尽量继续（移动效应=0）
        time_col = None
    if id_col is None:
        # 没有台风编号时，按整体一组（移动效应用相邻行估算）
        id_col = '__NO_ID__'
        df[id_col] = 0

    # 按台风编号分组，组内按时间估算移速与方向，并计算方位角
    vals_list = []
    bearing_list = []
    rel_bearing_list = []
    speed_list = []
    movedir_list = []
    for gid, g in df.groupby(id_col):
        g = g.copy()
        if time_col is not None:
            try:
                g[time_col] = pd.to_datetime(g[time_col])
            except Exception:
                pass
        # 估算移动
        if time_col is not None:
            spd_s, dir_s = _estimate_motion_for_group(g, time_col, lat_col, lon_col)
        else:
            spd_s = pd.Series([0.0] * len(g), index=g.index)
            dir_s = pd.Series([0.0] * len(g), index=g.index)
        # 计算中心->站点方位角（0-360°, 正北为0，顺时针）
        try:
            theta_deg_series = g.apply(lambda r: calculate_bearing(float(r[lon_col]), float(r[lat_col]), site_lon, site_lat), axis=1)
        except Exception:
            theta_deg_series = pd.Series([float('nan')] * len(g), index=g.index)
        # 相对方位角 = 方位角 - 移动方向，归一化到(-180, 180]
        rel = (theta_deg_series - dir_s + 540.0) % 360.0 - 180.0
        bearing_list.append(theta_deg_series)
        rel_bearing_list.append(rel)
        speed_list.append(spd_s)
        movedir_list.append(dir_s)

        sub_vals = g.apply(
            lambda r: compute_row_metrics(
                r, lat_col, lon_col, p_col, dist_col, site_lat, site_lon,
                ambient=1010.0,
                move_speed_ms=float(spd_s.loc[r.name]),
                move_dir_deg=float(dir_s.loc[r.name]),
                theta_deg=float(theta_deg_series.loc[r.name])
            ), axis=1
        )
        vals_list.append(sub_vals)
    vals = pd.concat(vals_list).sort_index()
    # 聚合方位角与移动信息列
    try:
        bearing_all = pd.concat(bearing_list).sort_index()
        rel_bearing_all = pd.concat(rel_bearing_list).sort_index()
        speed_all = pd.concat(speed_list).sort_index()
        movedir_all = pd.concat(movedir_list).sort_index()
        df['方位角'] = bearing_all.reindex(df.index)
        df['相对方位角'] = rel_bearing_all.reindex(df.index)
        df['移动速度(m/s)'] = speed_all.reindex(df.index)
        df['移动方向(°)'] = movedir_all.reindex(df.index)
    except Exception:
        df['方位角'] = math.nan
        df['相对方位角'] = math.nan
        df['移动速度(m/s)'] = math.nan
        df['移动方向(°)'] = math.nan
    # Columns as specified by user: 8 columns（追加在方位角两列之后）
    cols = ['rmax', 'b', 'hollandvg', 'rmax-ym', 'b-ym', 'hollandvg-ym', 'ym风速', 'ym风速(<=250km)']
    for i, cname in enumerate(cols):
        df[cname] = [t[i] if isinstance(t, tuple) else math.nan for t in vals]

    # Save to a sibling file with suffix _aug8
    out_path = os.path.splitext(path)[0] + '_aug8.xlsx'
    try:
        df.to_excel(out_path, index=False)
        print(f'Saved: {out_path} (rows={len(df)})')
    except PermissionError:
        # fallback name
        out_path = os.path.splitext(path)[0] + '_aug8_v1.xlsx'
        df.to_excel(out_path, index=False)
        print(f'Saved: {out_path} (rows={len(df)})')


def main():
    folder = os.path.join(BASE, 'Tydata', 'by_station')
    if not os.path.isdir(folder):
        raise SystemExit('未找到输出目录: ' + folder)
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith('.xlsx') and ('_aug8' not in f.lower()) and ('_with_wind' not in f.lower())]
    files.sort()
    if not files:
        print('by_station 目录下没有可处理的原始 xlsx 文件（已过滤 *_aug8.xlsx/_with_wind.xlsx）')
        return
    for f in files:
        process_file(f)


if __name__ == '__main__':
    main()
