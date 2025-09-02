#!/usr/bin/env python3
# Pandas ETL to create a cleaned daily dataset with derived metrics (TotalActiveMinutes, SleepEfficiency).
# Steps: load CSVs, normalize schemas, dedupe, parse dates, merge daily + sleep.

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main(data_dir: Path, out_dir: Path):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load candidate daily activity files (support multiple and concat)
    daily_files = list(data_dir.glob('dailyActivity*merged*.csv'))
    if not daily_files:
        daily_files = list(data_dir.glob('dailyActivity_*.csv'))
    daily_frames = []
    for f in daily_files:
        df = pd.read_csv(f)
        daily_frames.append(df)
    if not daily_frames:
        raise FileNotFoundError('No daily activity CSVs found in data/. Expected e.g. dailyActivity_merged.csv')
    daily = pd.concat(daily_frames, ignore_index=True)

    # Normalize column names (common Kaggle names)
    daily.columns = [c.strip().replace(' ', '').replace('/', '_') for c in daily.columns]
    daily = daily.rename(columns={'ActivityDate': 'Date', 'Id': 'Id'})

    # Parse Date
    if 'Date' in daily.columns:
        daily['Date'] = pd.to_datetime(daily['Date'], errors='coerce')

    # Remove duplicates
    daily = daily.drop_duplicates(subset=['Id', 'Date'])

    # 2) Load sleep
    sleep_path = None
    for candidate in ['sleepDay_merged.csv', 'sleepDay.csv']:
        fp = data_dir / candidate
        if fp.exists():
            sleep_path = fp
            break

    if sleep_path is None:
        sleep = pd.DataFrame(columns=['Id', 'Date', 'TotalMinutesAsleep', 'TotalTimeInBed'])
    else:
        sleep = pd.read_csv(sleep_path)
        sleep.columns = [c.strip().replace(' ', '').replace('/', '_') for c in sleep.columns]
        sleep = sleep.rename(columns={'SleepDay': 'Date'})
        if 'Date' in sleep.columns:
            sleep['Date'] = pd.to_datetime(sleep['Date'], errors='coerce')
        sleep = sleep.drop_duplicates(subset=['Id', 'Date'])

    # 3) Merge daily + sleep
    merged = pd.merge(
        daily,
        sleep[['Id', 'Date', 'TotalMinutesAsleep', 'TotalTimeInBed']],
        on=['Id', 'Date'],
        how='left',
    )

    # 4) Derived metrics
    for col in ['VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes']:
        if col not in merged.columns:
            merged[col] = 0
    merged['TotalActiveMinutes'] = merged['VeryActiveMinutes'].fillna(0) + merged['FairlyActiveMinutes'].fillna(0) + merged['LightlyActiveMinutes'].fillna(0)
    merged['SleepEfficiency'] = np.where(
        merged['TotalTimeInBed'].fillna(0) > 0,
        100 * merged['TotalMinutesAsleep'].fillna(0) / merged['TotalTimeInBed'].replace(0, np.nan),
        np.nan,
    )

    # 5) Save
    out_path = out_dir / 'daily_merged.csv'
    merged.to_csv(out_path, index=False)
    print(f'Saved: {out_path.resolve()} (rows={len(merged)})')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--out_dir', type=Path, required=True)
    args = ap.parse_args()
    main(args.data_dir, args.out_dir)