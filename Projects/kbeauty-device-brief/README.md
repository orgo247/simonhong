# K-Beauty Home-Device: Why Is Performance Declining at Jion Meditech?

## What this project does
A consulting-style, **data-science** analysis that explains *why* performance is declining at Jion Meditech (brand: DualSonic), by reading the story **across financial statements** (P/L, B/S, C/F) and benchmarking to listed leaders (**APR 278470, Amorepacific 090430, LG H&H 051900**).

- **Data collection (OpenDART peers):** corporation code (corpCode) + standardized XBRL statements (single-company full statements API).  
- **Target firm (지온메디텍):** public web pages and credit/company info pages for qualitative/limited quantitative context if DART/XBRL data are unavailable.  
- **Why inventory matters:** under **IAS 2**, inventories are measured at the **lower of cost and NRV**; rising DIO and “inventory write-downs” indicate demand/obsolescence risk and hit margins/cash.  
- **Deliverables:** Excel peer table, diagnostic plots, a 2-page executive brief.

Sources: OpenDART corpCode & financials, DART search portal, DualSonic brand pages, IAS 2. 
(Refs: OpenDART corpCode & single-firm statements; DART search; DualSonic site; IAS 2 inventory.) 

## Business question (framed)
**Why is performance deteriorating at Jion Meditech?**  
We test hypotheses that typically show up together:
1) Revenue up, but **operating-margin improvement decelerates**.  
2) **Advertising & promotion ratio** spikes (push marketing to clear stock).  
3) **Inventories and DIO** trend up; **inventory write-downs**/allowances appear.  
4) **Cash conversion** weak (CFO margin falls vs. EBIT).  
5) **Debt/Equity up**, **interest coverage down** (financing the working-capital gap).

## Data & scope
- **Peers (OpenDART APIs):** APR (278470), Amorepacific (090430), LG H&H (051900).  
- **Target firm:** Jion Meditech — brand: DualSonic (home-beauty devices). Public sources confirm brand/operator identity and company profile pages.  
- **Horizon:** last 5 fiscal years, quarterly & LTM views.

## Methods (DS workflow)
1) **ETL:** requests-based client for OpenDART (corpCode zip → parse → statements API).  
2) **Cleaning:** IFRS account normalization; KR/EN labels; currency units; period alignment.  
3) **KPIs:** Revenue growth/CAGR, Gross/Operating margins, SG&A%, Ad ratio (if present), Inventories & DIO, CFO margin, Debt/Equity, Interest coverage.  
4) **Diagnostics:**  
   - ΔOp% on ΔDIO / ΔAdRatio / ΔRevYoY with company fixed effects (interpretable panel).  
   - Event view: quarters with top-quintile Ad ratio and next-quarter Op% change.  
5) **Visualization:** margin trends, DIO vs. CFO margin, Ad ratio vs. Revenue growth, D/E & coverage.  
6) **Brief:** 2 pages with findings + actions (inventory discipline, promo ROI gating, channel/price mix).

## Repro (quick)
1) Add `.env` at repo root with `DART_API_KEY=xxxxx`.  
2) Run notebooks in order (once code stubs are added):  
   01_dart_pull → 02_kpi_clean → 03_decline_diagnostics → 04_brief_plots.  
3) Open `excel/peer_summary.xlsx` and `reports/figures/` for outputs.

*Note:* If 지온메디텍 lacks XBRL statements in OpenDART, treat its section as a qualitative deep-dive plus any public numeric snippets; keep all quantitative modeling to listed peers.
