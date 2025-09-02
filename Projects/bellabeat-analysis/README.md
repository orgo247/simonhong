# Bellabeat Smart Device Market Analysis (Showcase)

This repository is the **GitHub translation of my Bellabeat case study**. It highlights **what I did**, the **findings**, and the **exact code** to reproduce them. For a quick visual walkthrough, open the slide deck in `docs/SlideDeck.pdf` and the narrative in `docs/FINDINGS.md`.

---

## What I Did (overview)
- **Framed the business task**: use tracker data to identify behavior patterns (steps, intensity, sleep, hourly/weekly rhythms) and convert them into **product and marketing actions**.
- **Prepared the data** (Kaggle Fitbit): standardized schemas, parsed dates, deduplicated, and merged daily activity with sleep.
- **Engineered features**: `TotalActiveMinutes`, `SleepEfficiency`, plus clean hourly and weekly views.
- **Analyzed & segmented**:
  - Activity levels from steps (low/moderate/high)
  - Sleep adequacy buckets (>= 7 hours vs. < 7 hours)
  - User engagement (days logged)
  - Hour-of-day and day-of-week trends
- **Visualized** key relationships and rhythms with R/ggplot2.
- **Packaged for reproducibility**: scripts + notebook, instructions, findings docs, and deck.

---

## Key Findings (as presented in the deck)
- **Moderate activity** corresponds to the **best sleep balance** (≈ **7.75 hours**).
- **High activity** burns the most calories but is associated with **less sleep**.
- Users average roughly **~16 hours/day sedentary**; more movement does **not automatically** yield better sleep.
- **Engagement vs. Activity**: engagement does **not strongly correlate** with activity; many users are **passive**.
- **Engagement vs. Sleep**: no strong link; indicates **different user priorities**.
- **Temporal patterns**: activity peaks around **12–2 PM** and **5–7 PM**; **Tue–Thu dips**; **weekend sleep** improves.
- **Weight tracking** is **underused** (~13 users logged); likely due to manual friction.

**Slide deck (PDF):** `docs/SlideDeck.pdf`
**Narrative summary:** `docs/FINDINGS.md`

---

## Skills Demonstrated
- **SQL**: joins, CASE bucketing (activity/sleep/engagement), hourly/weekly aggregations, validation queries.
- **Python (Pandas)**: multi-CSV ingestion, schema standardization, deduping, date parsing, **feature engineering**, reproducible CLI.
- **R (tidyverse/ggplot2)**: clean, publication-style visuals saved to disk.

---

## Repo Map
```
bellabeat-analysis/
├─ scripts/
│  ├─ analysis.py              # Pandas ETL → data/daily_merged.csv (with new features)
│  ├─ queries.sql              # SQL segments (activity/sleep/engagement), hourly/weekly checks
│  └─ visualizations.R         # ggplot2 charts → outputs/figures/
├─ notebooks/
│  └─ bellabeat_analysis.ipynb # Text-based EDA, sanity checks, segmentations
├─ data/
│  └─ README.md                # Kaggle download & file placement
├─ outputs/
│  └─ figures/                 # Charts written by visualizations.R
├─ docs/
│  ├─ SlideDeck.pdf            # My slide presentation
│  └─ FINDINGS.md              # Plain-language summary
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## How to Run (reproduce my analysis)

### 1) Get the data
Download the **Fitbit Fitness Tracker** dataset from Kaggle and put the CSVs in `data/`. Keep the usual Kaggle filenames (e.g., `hourlySteps_merged.csv`, `hourlyIntensities_merged.csv`, `sleepDay_merged.csv`, etc.).

### 2) Create the merged daily dataset (Python)
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/analysis.py --data_dir data --out_dir data
```
This writes `data/daily_merged.csv` with `TotalActiveMinutes` and `SleepEfficiency`.

### 3) (Optional) SQL checks
Load CSVs into your DB and run pieces of `scripts/queries.sql` to recreate the segmentations and QA checks.

### 4) Visuals (R)
```r
# from repo root
source('scripts/visualizations.R')
```
Charts are saved into `outputs/figures/`.

### 5) Notebook (optional)
Open `notebooks/bellabeat_analysis.ipynb` to see text-based EDA, null checks, and the same segmentations.

---

## Method: Ask → Prepare → Process → Analyze → Share → Act
- **Ask**: What behaviors matter for engagement and wellness? What can we ship to move them?
- **Prepare**: Import CSVs; standardize schemas; ensure clean keys (Id, Date).
- **Process**: ETL in Python; feature engineering; consistent derived metrics.
- **Analyze**: Segment activity/sleep/engagement; hourly/weekly rhythms; sanity-check with SQL and notebook.
- **Share**: Charts, deck, and short narrative in this repo.
- **Act**: Time-windowed reminders; weekly cadence tuned to peaks; streamline weight logging; A/B test impact on steps, intensity, and sleep.

---

## Limitations & Next Steps
- Small, volunteer Fitbit sample; not representative of all users.
- Limited time frame; potential seasonality effects not captured.
- Next: extend to longer periods, add cohorts, test personalized policies with controlled experiments.

---

## Attribution
Dataset: **Fitbit Fitness Tracker Data (Kaggle)**. This repo is for educational/portfolio purposes.