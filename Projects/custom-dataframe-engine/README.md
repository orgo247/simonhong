# Custom DataFrame Engine & Analytics Application

**Author:** Simon Hong

## File Structure
```text
custom-dataframe-engine/
├── data/
│   ├── List of Orders.csv      # Sample dataset 1
│   ├── Order Details.csv       # Sample dataset 2
│   └── Sales target.csv        # Sample dataset 3
├── docs/
│   └── demo.png                # Screenshot(s)
├── scripts/
│   ├── app.py                  # Streamlit interactive dashboard
│   ├── custom_engine.py   # Core DataFrame engine (backend)
│   └── demo.py                 # Usage examples
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```


## Project Overview
This project is a Data Science engine built from scratch in Python. It mimics the core functionality of **Pandas** and **SQL** (Select, Join, GroupBy, Aggregate) without relying on external data manipulation libraries like NumPy or Pandas.

The goal was to deconstruct the "black box" of data engineering by manually implementing low-level algorithms for **Parsing**, **Indexing**, **Relational Algebra**, and **Vectorization**. The engine is wrapped in a Streamlit dashboard (`app.py`) for real-time script execution and visualization.

## Technical Architecture & Algorithms
To ensure high performance and intuitive usage, I implemented several key architectural patterns:

### 1. Optimized Hash Join ($O(N+M)$)
Instead of a naive Nested-Loop Join ($O(N^2)$), I implemented an **Inner Hash Join**:
* **Build Phase:** The engine indexes the right table into a Hash Map (Dictionary) based on the join key.
* **Probe Phase:** It scans the left table once, performing $O(1)$ lookups against the hash map.
* **Impact:** Drastic performance improvement when joining large tables compared to standard iteration.

### 2. Columnar Storage Architecture
* Data is stored in a column-oriented dictionary structure (`{column_name: [list_of_values]}`).
* **Why:** This makes **Projection** (selecting specific columns) an $O(1)$ operation, which is significantly faster than row-oriented storage for analytical queries.

### 3. Vectorized Arithmetic & Logic
I implemented a custom `Series` class with **Operator Overloading** to allow expressive, "Pythonic" syntax similar to Pandas:
* **Vectorization:** `df['Tax'] = df['Amount'] * 0.1` (Element-wise operations without manual loops).
* **Boolean Masking:** `df[df['Profit'] > 500]` (Filters data using boolean arrays).
* **Polymorphism:** The comparison operators automatically handle type conversion (Strings vs Floats) and robustly handle dirty data.

## The Data
The system was tested using a real-world E-Commerce dataset consisting of three relational tables:
1.  **List of Orders.csv:** Customer and shipping information (Primary Key: Order ID).
2.  **Order Details.csv:** Transaction specifics, profit, and quantity (Foreign Key: Order ID).
3.  **Sales Target.csv:** Monthly targets per category.

##  Usage & API Design
The API supports both **Functional** (Chaining) and **Imperative** (In-Place) paradigms, giving developers flexibility:

**Run the included demo script to see the engine in action:**

## How to Run
1. Clone repository
git clone https://github.com/orgo247/simonhong.git
cd simonhong/Projects/custom-dataframe-engine
2. Install dependencies
python3 -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
3. Lanch dashboard
streamlit run scripts/app.py
4. (Optional) Run demo script
python scripts/demo.py
