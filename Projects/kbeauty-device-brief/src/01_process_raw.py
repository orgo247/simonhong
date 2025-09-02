"""
Reads Jion Meditech’s income statement, balance sheet and cash‑flow statement
from `data/raw/jion/` (English‑labelled Excel files), extracts key metrics
for 2024 and 2023, computes financial ratios, and writes:
  - data/processed/jion_long.csv (tidy long form)
  - data/processed/metrics_jion.csv (wide KPIs + ratios)
  - data/processed/jion_summary.csv (year‑over‑year change)

Units are converted to hundreds of millions of won (억원) for readability.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Map canonical KPIs to labels in the Jion files
ACCOUNT_MAP = {
    "Revenue": ["Revenue"],
    "CostOfSales": ["Cost of sales"],
    "OperatingIncome": ["Operating profit (loss)", "Operating income"],
    "SGA": ["Selling, general and administrative expenses"],
    "Advertising": ["Advertising and promotion expenses"],
    "Inventory": ["Inventories"],
    "CurrentAssets": ["Current assets"],
    "CurrentLiabilities": ["Current liabilities"],
    "TotalLiabilities": ["Total liabilities", "부 채"],
    "TotalEquity": ["Total equity"],
    "CashAndCashEquivalents": ["Cash and cash equivalents"],
    "ShortTermBorrowings": ["Short‑term borrowings", "Short-term borrowings"],
    "CFO": ["Cash flows from operating activities"],
}

def tidy_jion_files(raw_dir: Path) -> pd.DataFrame:
    # Combine Jion IS/BS/CF into a tidy long DataFrame.
    def melt(file_name, year_current, year_prior, statement):
        df = pd.read_excel(raw_dir / file_name)
        cur = df[["account_en","current_amount"]].rename(columns={"current_amount":"amount"}).assign(year=year_current)
        pri = df[["account_en","prior_amount"]].rename(columns={"prior_amount":"amount"}).assign(year=year_prior)
        def to_num(x):
            s = str(x).replace(",", "").replace("(", "-").replace(")", "").strip()
            try: return float(s)
            except: return np.nan
        cur["amount"] = cur["amount"].map(to_num)
        pri["amount"] = pri["amount"].map(to_num)
        res = pd.concat([cur, pri], ignore_index=True).dropna(subset=["amount","account_en"])
        res["statement"] = statement
        return res
    is_long = melt("Jion_IS.xlsx", 2024, 2023, "IS")
    bs_long = melt("Jion_BS.xlsx", 2024, 2023, "BS")
    cf_long = melt("Jion_CF.xlsx", 2024, 2023, "CF")
    return pd.concat([is_long, bs_long, cf_long], ignore_index=True)[["year","statement","account_en","amount"]]

def compute_kpis(jion_long: pd.DataFrame) -> pd.DataFrame:
    # Pivot the tidy long form into KPIs and derive financial ratios.
    rows = []
    for year in sorted(jion_long["year"].unique()):
        row = {"company":"Jion Meditech", "ticker":"JION", "year":int(year)}
        df_year = jion_long[jion_long["year"] == year]
        for metric, names in ACCOUNT_MAP.items():
            mask = pd.Series(False, index=df_year.index)
            for name in names:
                mask |= df_year["account_en"].str.strip().str.lower() == name.strip().lower()
            val = df_year.loc[mask,"amount"].sum()
            if pd.notna(val):
                val = val / 1e8  # scale to hundreds of millions of won
            row[metric] = val
        rows.append(row)
    wide = pd.DataFrame(rows)
    eps = 1e-9
    wide["OpMargin"]     = wide["OperatingIncome"] / (wide["Revenue"] + eps)
    wide["SGA_ratio"]    = wide["SGA"]             / (wide["Revenue"] + eps)
    wide["Ad_ratio"]     = wide["Advertising"]     / (wide["Revenue"] + eps)
    wide["CurrentRatio"] = wide["CurrentAssets"]   / (wide["CurrentLiabilities"] + eps)
    wide["DebtToEquity"] = wide["TotalLiabilities"]/ (wide["TotalEquity"] + eps)
    wide["CFO_margin"]   = wide["CFO"]             / (wide["Revenue"] + eps)
    return wide

def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw" / "jion"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    jion_long = tidy_jion_files(raw_dir)
    jion_long.to_csv(processed_dir / "jion_long.csv", index=False)

    kpis = compute_kpis(jion_long)
    kpis.to_csv(processed_dir / "metrics_jion.csv", index=False)

    # Year-over-year summary
    metrics = ["Revenue","OperatingIncome","SGA","Advertising","Inventory","CFO",
               "OpMargin","SGA_ratio","Ad_ratio","CurrentRatio","DebtToEquity","CFO_margin"]
    summary = kpis[["year"] + metrics].sort_values("year").copy()
    for m in metrics:
        summary[f"{m}_YoY"] = summary[m].pct_change()
    summary.to_csv(processed_dir / "jion_summary.csv", index=False)
    print(f"Wrote Jion metrics and summary to {processed_dir}")

if __name__ == "__main__":
    main()
