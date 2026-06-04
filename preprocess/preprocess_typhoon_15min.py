import numpy as np
import pandas as pd


INPUT_FILE = "typhoon19-24complete.csv"
OUTPUT_ALL = "typhoon19-24complete_15min.csv"
OUTPUT_2020 = "typhoon19-24complete_15min_2020.csv"

# ghdt32.86N121.62E0m
STATION_LAT = 32.86
STATION_LON = 121.62


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def resample_one_event(g):
    g = g.sort_values("当前台风时间").copy()
    start = g["当前台风时间"].min()
    end = g["当前台风时间"].max()
    idx = pd.date_range(start=start, end=end, freq="15min")

    g = g.set_index("当前台风时间").reindex(idx)
    g.index.name = "当前台风时间"

    # Keep event-level identity/categorical fields continuous after reindex.
    cat_cols = ["台风编号", "台风中文名称", "台风英文名称", "台风起始时间", "台风结束时间", "台风强度", "移动方向"]
    for c in cat_cols:
        if c in g.columns:
            g[c] = g[c].ffill().bfill()

    num_cols = ["经度", "纬度", "台风等级", "风速", "气压", "移动速度"]
    for c in num_cols:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")

    g[num_cols] = g[num_cols].interpolate(method="time", limit_direction="both")

    # Keep integer-like cols clean after interpolation.
    if "台风等级" in g.columns:
        g["台风等级"] = np.rint(g["台风等级"]).astype("Int64")
    if "风速" in g.columns:
        g["风速"] = np.rint(g["风速"]).astype("Int64")
    if "气压" in g.columns:
        g["气压"] = np.rint(g["气压"]).astype("Int64")

    out = g.reset_index()
    return out


def main():
    df = pd.read_csv(INPUT_FILE)

    df["当前台风时间"] = pd.to_datetime(df["当前台风时间"], errors="coerce")
    df["台风起始时间"] = pd.to_datetime(df["台风起始时间"], errors="coerce")
    df["台风结束时间"] = pd.to_datetime(df["台风结束时间"], errors="coerce")

    df = df.dropna(subset=["当前台风时间", "台风编号", "经度", "纬度"]).copy()
    df = df.sort_values(["台风编号", "当前台风时间"]).drop_duplicates(subset=["台风编号", "当前台风时间"])

    parts = []
    for _, g in df.groupby("台风编号", sort=True):
        parts.append(resample_one_event(g))

    out = pd.concat(parts, ignore_index=True)

    out["场站距离_km"] = haversine_km(out["纬度"].astype(float), out["经度"].astype(float), STATION_LAT, STATION_LON)

    # For corrector script compatibility.
    out["target_time"] = out["当前台风时间"]

    out = out.sort_values(["台风编号", "当前台风时间"]).reset_index(drop=True)
    out.to_csv(OUTPUT_ALL, index=False, encoding="utf-8-sig")

    out2020 = out[(out["当前台风时间"] >= pd.Timestamp("2020-01-01")) & (out["当前台风时间"] < pd.Timestamp("2021-01-01"))].copy()
    out2020.to_csv(OUTPUT_2020, index=False, encoding="utf-8-sig")

    print("saved_all", OUTPUT_ALL, "rows", len(out))
    print("saved_2020", OUTPUT_2020, "rows", len(out2020))
    print("range_all", out["当前台风时间"].min(), out["当前台风时间"].max())
    if len(out2020) > 0:
        print("range_2020", out2020["当前台风时间"].min(), out2020["当前台风时间"].max())


if __name__ == "__main__":
    main()
