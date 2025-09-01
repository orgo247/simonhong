# Fetching raw financials (OpenDART)

This script pulls 5-year **BS/IS/CF** for the listed peers (APR, Amorepacific, LG H&H) and writes **unaltered raw CSVs** into `Projects/kbeauty-device-brief/data/raw/peers/`.

## 0) Requirements
- Python 3.10+
- `pip install pandas requests`
- An OpenDART API key (40 chars).

## 1) Set your key (one time)
- Create `.env` at repo root (optional) and export the key in your shell:
  ```bash
  export DART_API_KEY=YOUR_40_CHAR_KEY
  ```

## 2) Run from repo root
```bash
python 00_fetch_raw_dart.py
```

## 3) What you get
- `Projects/kbeauty-device-brief/data/raw/peers/all_peers_fnltt_raw_long.csv` (entire payload, long format)
- Per-company/statement splits, e.g.:
  - `Projects/kbeauty-device-brief/data/raw/peers/278470_IS_raw.csv`
  - `Projects/kbeauty-device-brief/data/raw/peers/090430_BS_raw.csv`
  - `Projects/kbeauty-device-brief/data/raw/peers/051900_CF_raw.csv`

These files contain everything returned by OpenDART’s **Single company’s full financial statements** endpoint, including `account_id`, `account_nm`, `thstrm_amount`(당기), `frmtrm_amount`(전기), and metadata.
