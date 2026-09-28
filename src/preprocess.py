"""Create leakage-safe chronological train/test files and Feast source data."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml

FEATURES = [
    "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1",
    "Sub_metering_2", "Sub_metering_3", "hour", "day_of_week", "month",
    "active_power_lag_1",
]
TARGET = "Global_active_power"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    p = yaml.safe_load(open(args.params))
    raw_path = p["data"]["raw_path"]
    # '?' marks missing values in the original UCI household-power data.
    df = pd.read_csv(raw_path, na_values="?", low_memory=False, nrows=p["data"].get("max_rows"))
    df["event_timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    numeric = [TARGET, "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("event_timestamp").dropna(subset=["event_timestamp", TARGET]).copy()
    df["hour"] = df.event_timestamp.dt.hour
    df["day_of_week"] = df.event_timestamp.dt.dayofweek
    df["month"] = df.event_timestamp.dt.month
    # A past-only lag is legitimate at prediction time; it never uses a future value.
    df["active_power_lag_1"] = df[TARGET].shift(1)
    df["household_id"] = "household_1"
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    # Chronological split happens before the model's imputer/scaler are fitted.
    split = int(len(df) * (1 - p["data"]["test_fraction"]))
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
    keep = ["event_timestamp", "household_id", TARGET] + FEATURES
    train[keep].to_csv(out / "train.csv", index=False)
    test[keep].to_csv(out / "test.csv", index=False)
    # Keep one row per hour for this local Feast demo. It makes historical joins
    # fast on a laptop; batch ML continues to use every minute-level row.
    # Feast point-in-time joins in fetch_features.py avoid future leakage.
    df[keep].iloc[::60].to_parquet(out / "feast_source.parquet", index=False)
    print(f"Saved {len(train):,} train rows and {len(test):,} test rows.")

if __name__ == "__main__": main()
