# K-Beauty Device Brief 

## What this is
A mini, consulting-style data science project: I pull standardized financial statements from **OpenDART** (Korea FSS), clean them with **pandas/NumPy**, compute KPIs, visualize trends, and write an executive-style brief. 

Target: 지온메디텍 (DualSonic)
Task: research & analyze why performance is declining (narrative across financials: revenue vs. margin decel, ad spend, inventory build/write-downs, cash squeeze, leverage).
Peers (for quantitative DART pulls): APR 278470, Amorepacific 090430, LG H&H 051900.
We’ll fetch standardized statements from OpenDART for the listed peers (corpCode + single-company full statements API) and use public pages for the target firm. 
Inventory write-downs are recognized under IAS 2 (lower of cost/NRV) — we’ll reference this in the brief.
