# Data Directory

This folder should contain the raw CSV files from the **FitBit Fitness Tracker Data** available on Kaggle.  The project does not include the data itself because of size and licensing restrictions.

## Downloading the Data

1. Create a Kaggle account (if you don't already have one).
2. Visit the FitBit Fitness Tracker Data page on Kaggle: <https://www.kaggle.com/datasets/arinjbarnes/fitbit-fitness-tracker-data>.
3. Agree to the terms and download the dataset.
4. Extract the contents of the zip file and copy the CSV files (e.g., `dailyActivity_merged.csv`, `sleepDay_merged.csv`, `heartrate_seconds_merged.csv`, etc.) into this `data/` directory.

Once the data is in place, you can run the analysis script from the root of the repository:

```bash
python scripts/analysis.py
```

The script expects specific file names, so please do not rename the CSV files.  If your downloaded files have different names, update the `DATA_FILES` dictionary in `scripts/analysis.py` accordingly.