# demo.py
# A script to demonstrate the capabilities of the Custom DataFrame Engine
# Usage: python demo.py

try:
    from custom_engine import DataFrame
except ImportError:
    from custom_engine import DataFrame

print("=======================================================")
print("   CUSTOM DATAFRAME ENGINE - FUNCTIONALITY DEMO")
print("=======================================================\n")

# 1. Loading Data
print("--- 1. LOADING DATA ---")
orders = DataFrame("data/List of Orders.csv")
details = DataFrame("data/Order Details.csv")
print(f"✓ Orders Loaded: {len(orders)} rows")
print(f"✓ Details Loaded: {len(details)} rows")


# 2. Feature Engineering
print("\n--- 2. FEATURE ENGINEERING (Vectorized Math) ---")
print("Action: details['Tax'] = details['Amount'] * 0.1")
# Adds a new column in-place using operator overloading
details['Tax'] = details['Amount'] * 0.1
print("✓ Column 'Tax' added successfully.")
print(details.select(['Order ID', 'Amount', 'Tax']).head(3))


# 3. Filtering
print("\n--- 3. FILTERING (Boolean Masking) ---")
print("Action: details[details['Amount'] > 5000]")
# Selects high-value transactions
high_value = details[details['Amount'] > 5000]
print(f"✓ Found {len(high_value)} high-value transactions.")
print(high_value.select(['Order ID', 'Amount']).head(3))


# 4. Relational Join
print("\n--- 4. RELATIONAL JOIN (Inner Hash Join) ---")
print("Action: orders.join(high_value, left_on='Order ID', right_on='Order ID')")
# Merges Customers with their Transactions
merged = orders.join(high_value, left_on="Order ID", right_on="Order ID")
print(f"✓ Join Complete. Resulting rows: {len(merged)}")
print(merged.select(['Order ID', 'CustomerName', 'Amount']).head(3))


# 5. Aggregation
print("\n--- 5. AGGREGATION (Split-Apply-Combine) ---")
print("Action: merged.groupby(['State']).agg({'Profit': 'sum', 'Order ID': 'count'})")
# Calculates total profit per State
stats = merged.groupby(["State"]).agg({
    "Profit": "sum",
    "Order ID": "count"
})
print("✓ Aggregation Results (Profit per State):")
print(stats)

print("\n=======================================================")
print("   DEMO COMPLETE")
print("=======================================================")
