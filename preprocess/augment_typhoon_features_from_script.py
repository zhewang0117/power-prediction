import argparse

import pandas as pd

import add_station_wind_columns_single as fext


def augment_file(input_csv: str, output_csv: str, site_lat: float, site_lon: float) -> None:
    df = pd.read_csv(input_csv)

    lat_col, lon_col, p_col, dist_col = fext.detect_cols(df)
    time_col, id_col = fext.detect_time_id_cols(df)

    if not all([lat_col, lon_col, p_col]):
        raise ValueError("Input file missing required columns for feature generation")
    if time_col is None:
        raise ValueError("Input file missing recognizable time column")

    if id_col is None:
        id_col = "__NO_ID__"
        df[id_col] = 0

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).copy()

    vals_list = []
    bearing_list = []
    rel_bearing_list = []
    speed_list = []
    movedir_list = []

    for _, g in df.groupby(id_col):
        g = g.copy().sort_values(time_col)

        spd_s, dir_s = fext._estimate_motion_for_group(g, time_col, lat_col, lon_col)
        theta_deg = g.apply(
            lambda r: fext.calculate_bearing(float(r[lon_col]), float(r[lat_col]), site_lon, site_lat), axis=1
        )
        rel_deg = (theta_deg - dir_s + 540.0) % 360.0 - 180.0

        vals = g.apply(
            lambda r: fext.compute_row_metrics(
                r,
                lat_col,
                lon_col,
                p_col,
                dist_col,
                site_lat,
                site_lon,
                ambient=1010.0,
                move_speed_ms=float(spd_s.loc[r.name]),
                move_dir_deg=float(dir_s.loc[r.name]),
                theta_deg=float(theta_deg.loc[r.name]),
            ),
            axis=1,
        )

        vals_list.append(vals)
        bearing_list.append(theta_deg)
        rel_bearing_list.append(rel_deg)
        speed_list.append(spd_s)
        movedir_list.append(dir_s)

    vals_all = pd.concat(vals_list).sort_index()
    df["方位角"] = pd.concat(bearing_list).sort_index().reindex(df.index)
    df["相对方位角"] = pd.concat(rel_bearing_list).sort_index().reindex(df.index)
    df["移动速度(m/s)"] = pd.concat(speed_list).sort_index().reindex(df.index)
    df["移动方向(°)"] = pd.concat(movedir_list).sort_index().reindex(df.index)

    cols = ["rmax", "b", "hollandvg", "rmax-ym", "b-ym", "hollandvg-ym", "ym风速", "ym风速(<=250km)"]
    for i, c in enumerate(cols):
        df[c] = [t[i] if isinstance(t, tuple) else float("nan") for t in vals_all]

    # Keep compatibility with downstream scripts.
    if "target_time" not in df.columns:
        df["target_time"] = df[time_col]

    df = df.sort_values([id_col, time_col]).reset_index(drop=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("saved", output_csv, "rows", len(df))
    print("time_range", df[time_col].min(), df[time_col].max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="typhoon19-24complete_15min_within1000km.csv")
    p.add_argument("--output", default="typhoon19-24complete_15min_within1000km_aug8.csv")
    p.add_argument("--site-lat", type=float, default=32.86)
    p.add_argument("--site-lon", type=float, default=121.62)
    args = p.parse_args()

    augment_file(args.input, args.output, args.site_lat, args.site_lon)


if __name__ == "__main__":
    main()
