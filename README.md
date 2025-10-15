# Anomaly Detection & Reporting

This repo implements a causal, adaptive outlier detector for personal-finance transactions and produces a JSON payload with:
- the metric values that triggered each anomaly, and
- a “transaction” string containing date, description, amount.

It mirrors the logic used in the notebook (`data_exploration.ipynb`): adaptive z-threshold, modified z, IQR fence, refund pairing, “quick reversals”, and last-month focus.

---

## 📂 Project Structure

```
Case_credit_card/
├─ analysis/
│  ├─ data_exploration.ipynb         # Full exploratory analysis & metric development
│  ├─ report.md                      # Report about the metrics used here
│  ├─ cv_category_tx.png                # Example plot used in the report (optional)
├─ code/
│  ├─ main.py                        # Entry-point script: reads CSV and writes Section 4 JSON
│  ├─ utils.py                       # Helper functions
│  ├─ output.json                    # Example output (produced by main.py)
├─ transactions.csv                  # Raw input data (date, category, amount, description/merchant…)
├─ Test tech - Data Scientist.pdf    # Assignment/instructions
├─ requirements.txt                  # Python package requirements for your venv
└─ README.md                         # This file
```
---

## 🛠️ Environment Setup

You can use conda (recommended) or pip.

### Option A — Conda

```bash
# 1) Create and activate the environment
conda create -n envname python=3.11 -y
conda activate envname

# 2) Install Python deps from requirements.txt (pip format)
pip install -r requirements.txt
```

### Option B — Pip / venv (no conda)

```bash
python -m venv envname
# Windows:
envname\Scripts\activate
# macOS/Linux:
source envname/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ How to Run

From the repo root:

```bash
# Windows PowerShell example
python code\main.py --csv transactions.csv --out code\output.json
```

- `--csv`: path to the raw CSV (with `date`, `category`, `amount`; `description`/`merchant` optional).
- `--out`: path to the JSON output.

### What `main.py` does

1. read_df: loads and normalizes input (parses `date`, builds `inflow`/`outflow`, `year_month`).
2. df_tx: computes anomaly flags (causal z/modz/IQR), matches refunds, sets `effective_outflow`, and marks `reversed_quickly`.
3. latest_month_anomalies: selects the latest month, filters `outlier_status == "Valid"`, `outlier_gross == True`, `reversed_quickly == False`, and `outflow > 0`. If `description` is missing, it tries to merge from the raw file using robust keys.
4. build_payload: for each anomaly, records the triggering metric (|z|, |modz|, or IQR ratio) and a human string `YYYY-MM-DD — description — amount`. Writes JSON like:

```json
{
  "metric_name": "adaptive_outlier_rule",
  "value": [3.92, 4.31, 1.27],
  "transaction": [
    "2025-08-14 — Electronics Store — -1450.00",
    "2025-08-22 — Travel — -980.00",
    "2025-08-05 — Restaurants — -320.00"
  ]
}
```

---

## ✅ Data Requirements

- Required columns: `date`, `category`, `amount`
  - `amount` < 0 → spending (outflow)
  - `amount` > 0 → income (inflow)
- Optional: `description` (or equivalents like `merchant`, `memo`, `details`)

---

## 📚 References

- Notebook: `data_exploration.ipynb` (full metric derivation and validations).
- PDF: `Test tech - Data Scientist.pdf` (assignment specification).
