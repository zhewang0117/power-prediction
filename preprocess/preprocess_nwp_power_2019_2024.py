import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


NWP_FILE = "ghdt32.86N121.62E0m.xlsx"
POWER_GLOB = "power/ghdt_*.xlsx"
OUT_DIR = Path("processed_15min")


def load_nwp_15min() -> pd.DataFrame:
    df = pd.read_excel(NWP_FILE, header=3)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")

    if "time" not in df.columns:
        raise ValueError("NWP file does not contain 'time' column")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"])

    value_cols = [c for c in df.columns if c != "time"]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.set_index("time").sort_index()
    nwp_15min = df.resample("15min").asfreq().interpolate(method="time", limit_direction="both")
    nwp_15min = nwp_15min.reset_index()
    return nwp_15min


def load_power_15min(file_path: str, year: int) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    if "CREATE_TIME" not in df.columns or "VALUE" not in df.columns:
        raise ValueError(f"Power file columns mismatch: {file_path}")

    df["time"] = pd.to_datetime(df["CREATE_TIME"], errors="coerce")
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"])

    # Reindex to complete minutely timeline for each year, then interpolate missing values.
    full_min = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:59:00", freq="1min")
    s = df.set_index("time")["VALUE"].reindex(full_min)
    s = s.interpolate(method="time", limit_direction="both")

    power_15min = s.resample("15min").mean().to_frame(name="VALUE").reset_index().rename(columns={"index": "time"})
    return power_15min


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nwp_15min = load_nwp_15min()
    nwp_path = OUT_DIR / "nwp_15min_2019_2024.csv"
    nwp_15min.to_csv(nwp_path, index=False, encoding="utf-8-sig")

    summary_rows = []

    for fp in sorted(glob.glob(POWER_GLOB)):
        name = os.path.basename(fp)
        m = re.search(r"ghdt_(\d{4})\.xlsx", name)
        if not m:
            continue
        year = int(m.group(1))

        power_15min = load_power_15min(fp, year)
        power_path = OUT_DIR / f"power_15min_{year}.csv"
        power_15min.to_csv(power_path, index=False, encoding="utf-8-sig")

        nwp_year = nwp_15min[
            (nwp_15min["time"] >= pd.Timestamp(f"{year}-01-01 00:00:00"))
            & (nwp_15min["time"] <= pd.Timestamp(f"{year}-12-31 23:45:00"))
        ].copy()

        merged = pd.merge(power_15min, nwp_year, on="time", how="inner")
        merged_path = OUT_DIR / f"ghdt_merged_15min_{year}.csv"
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

        summary_rows.append(
            {
                "year": year,
                "power_rows_15min": len(power_15min),
                "nwp_rows_15min": len(nwp_year),
                "merged_rows_15min": len(merged),
                "merged_start": merged["time"].min() if len(merged) else pd.NaT,
                "merged_end": merged["time"].max() if len(merged) else pd.NaT,
                "power_value_na": int(power_15min["VALUE"].isna().sum()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("year")
    summary_path = OUT_DIR / "preprocess_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("saved:", nwp_path)
    print("saved:", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
