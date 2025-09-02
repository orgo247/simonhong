# K‑Beauty Home Device – Jion Meditech Analysis (Portfolio Project)

Diagnose **why Jion Meditech’s performance deteriorated despite sales growth** and benchmark its KPIs versus industry leaders. This repo mimics a consulting analyst task executed with a **data‑science** workflow: data ingestion → cleaning → KPI construction → peer benchmarking → brief.

---

## 1) What this project shows 
- **Business framing → KPIs:** Translate business questions into measurable KPIs (Op margin, Ad/Sales, CFO margin, Current ratio, D/E, inventory & valuation losses).
- **Data engineering with pandas/numpy:** Normalize heterogeneous statements (IS/BS/CF), long↔wide reshaping, canonical account mapping, robust ratio math.
- **External data retrieval:** Pull 5‑year peer financials via **OpenDART** and standardize.
- **Analytic storytelling:** Compare Jion vs peer averages (leaders), show trend lines, and turn signals into decisions (inventory, ad ROI, liquidity).
- **Reproducible pipeline:** Clear `src/` scripts, deterministic outputs in `data/processed/`, deliverables `reports/`.

---

## 2) Repository structure

```
kbeauty-device-brief/
├── data/
│   ├── raw/
│   │   ├── jion/                # Jion files (given): Jion_IS.xlsx, Jion_BS.xlsx, Jion_CF.xlsx
│   │   └── peers/               # OpenDART pulls cached here
│   └── processed/               # All derived CSVs live here
├── src/
│   ├── 00_fetch_raw_data.py     # Fetch peers from OpenDART (2019/20–2024)
│   ├── 01_process_raw.py        # Clean Jion → metrics_jion.csv
│   └── 02_kpi_diagnostics.py    # KPIs, YoY, peer means, charts
├── .env                         # DART_API_KEY=xxxxxxxx...(40 hex chars)
├── requirements.txt
└── reports/report.md            # Narrative brief with findings
```

> **Note on periods**
> Jion’s current/prior columns are **normalized to 2023–2024**. Peers are processed for **2020–2024**; if a peer year is missing for 2024, the 2023 mean is carried forward (“2024≈2023”) and explicitly noted in the brief.

---

## 3) Quickstart

### A) Environment
```bash
# from repo root
python -m venv .venv && source .venv/bin/activate  # or conda env
pip install -r requirements.txt
```

### B) Secrets
Create `.env` at repo root:
```
DART_API_KEY=YOUR_40_CHAR_HEX_KEY #retrieved from OPEN DART
```
> Must be **exactly 40 hex chars** (0–9, a–f).

```

### C) Data (Jion given)
Put the three files here:
```
data/raw/jion/Jion_IS.xlsx
data/raw/jion/Jion_BS.xlsx
data/raw/jion/Jion_CF.xlsx
```

### D) Run the pipeline
```bash
# fetch peers from OpenDART
python src/00_fetch_raw_data.py

# process Jion → metrics & wide table
python src/01_process_raw.py

# compute KPIs + peer means + charts
python src/02_kpi_diagnostics.py
```

Outputs:
- `data/processed/metrics_jion.csv` – Jion KPIs (2023–2024)
- `data/processed/jion_summary.csv` – Jion YoY deltas
- `data/processed/metrics_peers.csv` – Peer KPIs panel
- `data/processed/peer_avg_by_year.csv` – Peer **yearly means** (leaders, 2020–2024)
- `data/processed/jion_vs_peer.csv` / `jion_vs_peer_avg.csv` – Comparison tables
- Figures in `reports/figures/` and `reports/figures_peer_trend/`
- Narrative: `reports/report.md`

---

## 4) Methodology

1) **Normalize statements** (IS/BS/CF): map synonymous Korean/English account names to canonical labels.
2) **Construct KPIs** (levels & ratios):
   - Levels: Revenue, COGS, Operating income, SG&A, Advertising, Inventories, CFO, Short‑term borrowings, Cash.
   - Ratios: `OpMargin = OperatingIncome / Revenue`, `SGA_ratio`, `Ad_ratio`, `CFO_margin`,
     `CurrentRatio = CurrentAssets / CurrentLiabilities`, `DebtToEquity = TotalLiabilities / TotalEquity`.
3) **Peer mean by year**: for APR/Amorepacific/LG H&H, compute **mean** 2020–2024.
4) **Compare Jion 2023–2024 vs peer means**: deltas and trend lines to show **direction and magnitude** of gap.
5) **Interpretation**: tie financial signals to operations: demand forecasting → inventory → cash; promotion ROI; liquidity/leverage discipline.

---

## 5) Findings

- **Sales ↑, COGS ↓** → gross margin improved, **but** **Operating income ↓** because **SG&A/Advertising ↑**.
- **Inventories ↑ + valuation loss** → demand miss; **CFO margin ↓** (cash conversion weak).
- **Current ratio thin**, **Debt‑to‑Equity high** → reliance on short‑term debt.
- **Peer comparison (means)**: Jion’s **Ad/Sales higher** and **Op/CFO margins lower** than leaders’ trend → **promotion ROI & overhead absorption** are the pain points.

See **`reports/report.md`** for details and charts.

---

## 6) Limitations & next steps

- Peer set limited to three leaders; add more comparables if available.
- 2024 peer gaps are explicitly documented if carried forward from 2023.
- For causality, run **incrementality** (MMM‑lite/geo split) and **inventory liquidation experiments**; fold results into the forecast loop.

---

## 7) Credits

Built as a learning/portfolio exercise to mimic a consulting analyst + data‑science workflow for K‑beauty home devices.
