# 📈 Sales Forecasting Tool for Small E-Commerce Stores

A Streamlit web app that turns a Shopify/WooCommerce order export into a
demand forecast, top-product insights, and inventory reorder recommendations
— no data science background required from the end user.

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload `data/ecommerce_orders.csv` to try it
on a 2-year synthetic dataset with realistic seasonality (weekday patterns,
Black Friday/Cyber Monday spikes, Christmas surge, New Year fitness boost,
Valentine's Day gift spike).

---

## What it does

### 1. Flexible CSV ingestion
Auto-detects column names across common export formats:

| Concept | Recognized column names |
|---|---|
| Date | `Order Date`, `Date`, `Created at`, `Paid at` |
| Sales total | `Total Sales`, `Total`, `Subtotal`, `Lineitem price` |
| Quantity | `Quantity`, `Qty`, `Lineitem quantity` |
| Unit price | `Unit Price`, `Price`, `Item price` |
| Product | `Product`, `Lineitem name`, `Title` |
| Category | `Category`, `Product Type`, `Collection` |

If no sales-total column exists, sales = `quantity × unit_price`.

### 2. Demand forecasting
- Aggregates order lines into a **daily sales time series**
- Engineers ~25 time-series features: lags (1/7/14/28 days), rolling
  mean/std (7/14/30 days), cyclical day-of-week & month encodings,
  and retail-specific flags (Black Friday week, Christmas period, New Year)
- Trains a **Gradient Boosting Regressor** (scikit-learn)
- Produces a **recursive multi-step forecast** (7–90 days, configurable)
- Reports validation MAE, MAPE, and R² on a held-out period

### 3. Sales pattern analysis
- Average sales by day of week (spot weekend vs. weekday patterns)
- Monthly sales trend

### 4. Top-product breakdown
Revenue, units sold, and order count per product — sorted by revenue.

### 5. Inventory reorder recommendations
Enter your **current stock level** and **supplier lead time**, and get:
- Average daily demand
- Projected stockout date
- Reorder point and recommended order quantity
- A clear "reorder now" alert if stock is below the reorder point

---

## Project Structure

```
sales_forecast/
├── app.py                  # Streamlit UI
├── forecasting_engine.py    # Core logic (no Streamlit deps — testable standalone)
├── requirements.txt
├── data/
│   └── ecommerce_orders.csv # 2-year synthetic demo dataset (36K order lines)
└── README.md
```

---

## Use the engine standalone

```python
import pandas as pd
from forecasting_engine import (
    detect_columns, load_and_aggregate, train_model, forecast_future
)

df = pd.read_csv("my_orders.csv")
col_map = detect_columns(df)
daily = load_and_aggregate(df, col_map)

model, metrics, _ = train_model(daily)
forecast = forecast_future(model, daily, horizon=30)

print(forecast)
print("Validation MAPE:", metrics["mape"])
```

---

## Notes & Limitations

- **Minimum data:** ~45 days of daily history recommended (the model uses
  28-day lags and 30-day rolling averages). Less data → less reliable
  seasonality detection.
- **Demand units:** The forecast is in the same units as your sales
  column (typically revenue/$). For unit-based inventory planning, ensure
  your file has a quantity column.
- **New stores:** With <2 months of data, the model falls back to simpler
  patterns (day-of-week, short-term trend) since yearly seasonality can't
  yet be learned.
- **Cold-start products:** Per-product forecasting isn't included by
  default (only aggregate store-level demand) — this can be added as a
  per-SKU loop for stores with consistent SKU-level history.

---

## Customization Ideas (for client work)

- Per-product / per-category forecasts (loop the engine over each SKU)
- Multi-store comparison dashboards
- Email/Slack alerts when reorder points are hit
- Promotional impact analysis (compare forecast vs. actual during sales events)
- Currency/timezone handling for international stores
