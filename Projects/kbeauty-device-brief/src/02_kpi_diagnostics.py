"""
Compares Jion 2023–2024 performance to peers. It reads the processed Jion
metrics from metrics_jion.csv and, if available, processes peer raw files
from data/raw/peers/ (or from any *_raw.csv in the project root) into KPI
tables, computes per‑year means across the peers, and writes:

  - data/processed/jion_summary.csv         (year-over-year % changes)
  - data/processed/metrics_peers.csv        (peer-level KPIs)
  - data/processed/peer_avg_by_year.csv     (peer means per year, 2019–2023; 2023 repeated for 2024)
  - data/processed/jion_vs_peer_avg.csv     (Jion vs. peer differences)

Run:
    python src/02_kpi_diagnostics.py
from the project root.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import glob

def compute_jion_yoy(df):
    years = sorted(df.index)
    prev_year, curr_year = years[0], years[-1]
    rows = []
    for col in df.columns:
        v0 = df.loc[prev_year, col]
        v1 = df.loc[curr_year, col]
        change = np.nan if v0 == 0 else (v1 - v0) / abs(v0) * 100.0
        rows.append({"Metric": col, str(prev_year): v0, str(curr_year): v1, "Change_pct": change})
    return pd.DataFrame(rows)

def load_raw_peers(project_root):
    """
    Look for raw peer files in data/raw/peers/ or the project root.
    Returns a concatenated DataFrame or None if nothing found.
    """
    # Candidate directories: project root and data/raw/peers
    dirs = [project_root, project_root / "data" / "raw" / "peers"]
    for d in dirs:
        if not d.exists():
            continue
        # find split files *_IS_raw.csv, *_BS_raw.csv, *_CF_raw.csv
        files = []
        for stmt in ["*_IS_raw.csv","*_BS_raw.csv","*_CF_raw.csv"]:
            files.extend(glob.glob(str(d / stmt)))
        if files:
            frames = [pd.read_csv(f) for f in files]
            return pd.concat(frames, ignore_index=True)
    return None

def map_account(account_nm):
    # Map a Korean account name to a KPI label. 
    synonyms = {
        'Sales': ['매출액','매출','영업수익'],
        'CostOfSales': ['매출원가'],
        'OperatingIncome': ['영업이익'],
        'SGA': ['판매비와관리비','판매비및관리비'],
        'Advertising': ['광고선전비','판매촉진비','마케팅비'],
        'Inventory': ['재고자산'],
        'CurrentAssets': ['유동자산'],
        'CurrentLiabilities': ['유동부채'],
        'TotalLiabilities': ['부채총계','부 채','총부채'],
        'TotalEquity': ['자본총계','총자본'],
        'CashAndCashEquivalents': ['현금및현금성자산','현금및현금성'],
        'ShortTermBorrowings': ['단기차입금'],
        'CFO': ['영업활동으로인한현금흐름','영업활동현금흐름','영업활동으로 인한 현금흐름'],
    }
    for k, names in synonyms.items():
        for nm in names:
            if nm in account_nm:
                return k
    return None

def process_peers_raw(raw_df):
    """
    Convert raw peer data into KPI and ratio columns per company/year.
    Returns a DataFrame with columns:
       peer_corp_name, peer_stock_code, year,
       sales, op_margin, sga_ratio, ad_ratio,
       current_ratio, debt_to_equity, cfo_margin, inventory
    """
    # Use only annual reports (reprt_code == 11011)
    raw_df = raw_df[raw_df["reprt_code"] == 11011].copy()
    raw_df["metric"] = raw_df["account_nm"].map(map_account)
    raw_df = raw_df.dropna(subset=["metric"])

    # Convert amounts to numeric (hundreds of millions)
    def to_num(x):
        s = str(x).replace(",", "").replace("(", "-").replace(")", "").strip()
        try: return float(s) / 1e8
        except: return np.nan
    raw_df["amount"] = raw_df["thstrm_amount"].apply(to_num)

    # Pivot to wide: one row per company/year
    pivot = raw_df.pivot_table(index=["peer_corp_name","peer_stock_code","bsns_year"],
                               columns="metric", values="amount", aggfunc="sum").reset_index()
    pivot = pivot.rename(columns={"bsns_year":"year"})
    eps = 1e-9
    pivot["sales"]         = pivot.get("Sales")
    pivot["op_margin"]     = pivot.get("OperatingIncome") / (pivot["sales"] + eps)
    pivot["sga_ratio"]     = pivot.get("SGA")             / (pivot["sales"] + eps)
    pivot["ad_ratio"]      = pivot.get("Advertising")     / (pivot["sales"] + eps)
    pivot["current_ratio"] = pivot.get("CurrentAssets")   / (pivot.get("CurrentLiabilities") + eps)
    pivot["debt_to_equity"]= pivot.get("TotalLiabilities")/ (pivot.get("TotalEquity") + eps)
    pivot["cfo_margin"]    = pivot.get("CFO")             / (pivot["sales"] + eps)
    pivot["inventory"]     = pivot.get("Inventory")
    return pivot[["peer_corp_name","peer_stock_code","year",
                  "sales","op_margin","sga_ratio","ad_ratio",
                  "current_ratio","debt_to_equity","cfo_margin","inventory"]]

def compute_peer_mean_by_year(kpis):
    """Compute average (mean) of each KPI per year and replicate the last year for the next year."""
    metrics = ["sales","op_margin","sga_ratio","ad_ratio",
               "current_ratio","debt_to_equity","cfo_margin","inventory"]
    out = []
    for year, group in kpis.groupby("year"):
        row = {"year": int(year)}
        for m in metrics:
            row[f"{m}_mean"] = pd.to_numeric(group[m], errors="coerce").mean(skipna=True)
        out.append(row)
    summary = pd.DataFrame(out).sort_values("year")
    # replicate last year forward if needed
    if not summary.empty:
        last_year = summary["year"].max()
        next_year = last_year + 1
        if next_year not in summary["year"].values:
            last_row = summary[summary["year"] == last_year].copy()
            last_row["year"] = next_year
            summary = pd.concat([summary, last_row], ignore_index=True).sort_values("year")
    return summary

def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load Jion metrics
    jion = pd.read_csv(processed_dir / "metrics_jion.csv")

    # Compute Jion YoY changes (unchanged from earlier)
    yo_cols = ["Revenue","OperatingIncome","SGA","Advertising","Inventory","CFO",
               "OpMargin","SGA_ratio","Ad_ratio","CurrentRatio","DebtToEquity","CFO_margin"]
    sorted_j = jion.set_index("year").sort_index()[yo_cols]
    jion_yoy = compute_jion_yoy(sorted_j)
    jion_yoy.to_csv(processed_dir / "jion_summary.csv", index=False)

    # Load and process peers
    raw_peers = load_raw_peers(project_root)
    if raw_peers is not None and not raw_peers.empty:
        peer_kpis = process_peers_raw(raw_peers)
        peer_kpis.to_csv(processed_dir / "metrics_peers.csv", index=False)

        peer_avg = compute_peer_mean_by_year(peer_kpis)
        peer_avg.to_csv(processed_dir / "peer_avg_by_year.csv", index=False)

        # Align Jion columns to peer metrics
        aligned = jion.copy()
        aligned["sales"]         = aligned["Revenue"]
        aligned["op_margin"]     = aligned["OpMargin"]
        aligned["sga_ratio"]     = aligned["SGA_ratio"]
        aligned["ad_ratio"]      = aligned["Ad_ratio"]
        aligned["current_ratio"] = aligned["CurrentRatio"]
        aligned["debt_to_equity"]= aligned["DebtToEquity"]
        aligned["cfo_margin"]    = aligned["CFO_margin"]
        aligned["inventory"]     = aligned["Inventory"]

        merged = aligned.merge(peer_avg, on="year", how="left")
        # compute differences vs. peer mean
        for m in ["sales","op_margin","sga_ratio","ad_ratio",
                  "current_ratio","debt_to_equity","cfo_margin","inventory"]:
            mean_col = f"{m}_mean"
            if mean_col in merged.columns:
                merged[f"{m}_vs_peer_mean"] = merged[m] - merged[mean_col]
        merged.to_csv(processed_dir / "jion_vs_peer_avg.csv", index=False)
        print("Peer averages and Jion vs peer mean comparison saved.")
    else:
        print("No peer data found. Run 00_fetch_raw_data.py first or check raw files in data/raw/peers.")

if __name__ == "__main__":
    main()
