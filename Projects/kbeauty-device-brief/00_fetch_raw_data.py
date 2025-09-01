#!/usr/bin/env python3
# Fetch 5-year BS/IS/CF raw financials for listed peers from OpenDART and save under data/raw/peers/.
# Requires: DART_API_KEY in environment (40 chars).

import os, io, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import requests, pandas as pd

API = "https://engopendart.fss.or.kr/engapi"
KEY = os.getenv("DART_API_KEY")
assert KEY and len(KEY) == 40, "Set DART_API_KEY (40 chars) in your environment."

# ----- config -----
PEER_TICKERS = ["278470", "090430", "051900"]  # APR, Amorepacific, LG H&H
SJ_DIVS = ["BS","IS","CF"]                     # Statements
REPRT_CODE = "11011"                           # Annual
FS_DIV = "CFS"                                 # Consolidated
# last 5 completed fiscal years (adjust if needed)
y = datetime.now().year
YEARS = list(range(y-5, y))

# paths (run from repo root)
RAW_DIR = Path("Projects/kbeauty-device-brief/data/raw/peers")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def get_corp_table():
    r = requests.get(f"{API}/corpCode.xml", params={"crtfc_key": KEY}, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xml_name = zf.namelist()[0]
        xml_bytes = zf.read(xml_name)
    root = ET.fromstring(xml_bytes)
    rows = []
    for el in root.findall(".//list"):
        rows.append({
            "corp_code": el.findtext("corp_code"),
            "corp_name": el.findtext("corp_name"),
            "stock_code": el.findtext("stock_code"),
        })
    return pd.DataFrame(rows)

def resolve_by_stock(df, stock_codes):
    stock_codes = {str(s).zfill(6) for s in stock_codes}
    return df[df["stock_code"].isin(stock_codes)][["stock_code","corp_name","corp_code"]].reset_index(drop=True)

def fetch_fnltt(corp_code: str, year: int, sj_div: str, fs_div: str = FS_DIV, reprt_code: str = REPRT_CODE) -> pd.DataFrame:
    params = {
        "crtfc_key": KEY,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
        "fs_div": fs_div,
        "sj_div": sj_div,
    }
    r = requests.get(f"{API}/fnlttSinglAcntAll.json", params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "000":
        return pd.DataFrame()
    df = pd.DataFrame(data.get("list", []))
    if df.empty:
        return df
    df["source"] = "opendart_fnlttSinglAcntAll"
    df["requested_fs_div"] = fs_div
    df["requested_reprt_code"] = reprt_code
    df["requested_sj_div"] = sj_div
    return df

def main():
    corp_df = get_corp_table()
    peer_map = resolve_by_stock(corp_df, PEER_TICKERS)
    if peer_map.empty:
        raise SystemExit("No peers resolved. Check tickers.")

    all_frames = []
    for _, row in peer_map.iterrows():
        code, name, ticker = row.corp_code, row.corp_name, row.stock_code
        print(f"[{ticker}] {name} ({code})")
        for yr in YEARS:
            for sj in SJ_DIVS:
                df = fetch_fnltt(code, yr, sj)
                if not df.empty:
                    df["peer_stock_code"] = ticker
                    df["peer_corp_name"] = name
                    all_frames.append(df)
                else:
                    print(f"  - No data: year={yr} sj={sj} (skipped)")

    if not all_frames:
        raise SystemExit("No data returned from API. Check key/years.")
    raw_long = pd.concat(all_frames, ignore_index=True)

    out_all = RAW_DIR / "all_peers_fnltt_raw_long.csv"
    raw_long.to_csv(out_all, index=False, encoding="utf-8-sig")
    print("Wrote raw long:", out_all)

    # per-company/statement splits
    for (ticker, sj), g in raw_long.groupby(["peer_stock_code","requested_sj_div"]):
        out_path = RAW_DIR / f"{ticker}_{sj}_raw.csv"
        g.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("Wrote:", out_path)

if __name__ == "__main__":
    main()
