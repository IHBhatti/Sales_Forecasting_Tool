# 📈 Sales Forecasting Tool for E-Commerce Stores
**An AI-powered demand forecasting web app for Shopify & WooCommerce stores.**
Upload your order export → get a 30–90 day sales forecast, top-product breakdown,
and inventory reorder recommendations — in under 60 seconds.

---

## 🧩 The Problem It Solves

Most small e-commerce store owners manage inventory with gut feeling and spreadsheets.
The result:

- 📦 **Overstock** — money tied up in slow-moving products
- ❌ **Stockouts** — missed sales during peak demand (Black Friday, Christmas)
- ⏰ **Time waste** — hours spent on manual demand planning every week

This tool replaces all of that with a machine learning forecast built specifically on
**your store's historical data** — not generic industry averages.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Smart Column Detection** | Auto-detects date, sales, quantity, product columns across Shopify, WooCommerce, and generic CSV export formats — no manual mapping needed |
| 📈 **AI Demand Forecast** | Gradient Boosting model trained on your sales history, capturing weekly patterns, yearly seasonality, and retail holiday spikes |
| 🔁 **Recursive Multi-Step Forecasting** | Forecasts 7 to 90 days into the future with a single upload |
| 📊 **Sales Pattern Analysis** | Day-of-week averages and monthly trend charts — spot your best and worst selling days |
| 🏆 **Top Product Breakdown** | Revenue, units sold, and order count ranked per product |
| 📦 **Inventory Reorder Alerts** | Enter current stock + supplier lead time → get a projected stockout date and recommended order quantity |
| ✅ **Model Accuracy Reporting** | MAE, MAPE, and R² computed on a held-out validation period so you know exactly how trustworthy the forecast is |
| ⬇️ **CSV Download** | Export the full forecast as a downloadable CSV |

---

## 🎬 Demo

### Upload → Forecast in seconds

```
Upload order CSV  →  Auto-detect columns  →  Train model  →  Forecast + Insights
```

### Sample Forecast Output

| Date       | Forecasted Sales ($) |
|------------|----------------------|
| 2025-01-01 | 1,050.41             |
| 2025-01-02 | 1,189.73             |
| 2025-01-03 | 1,143.32             |
| ...        | ...                  |
| 2025-01-30 | 1,398.55             |

**30-day total forecast: $38,450 · Validation MAPE: ~17%**

> 💡 Try it yourself with the included `data/ecommerce_orders.csv`
> — 2 years of synthetic order data across 12 products with realistic
> Black Friday, Christmas, and seasonal demand patterns.

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.
Upload `data/ecommerce_orders.csv` to see a live demo instantly.

---


## 🗂️ Supported CSV Formats

The app auto-detects columns — no reformatting required for standard exports.

### Shopify Order Export
```
Name, Email, Created at, Lineitem name, Lineitem quantity, Lineitem price, Total
```

### WooCommerce Order Export
```
Order ID, Order Date, Product, Quantity, Unit Price, Order Total
```

### Generic / Custom Format
Any CSV with at minimum:
- A **date** column (called anything: `date`, `order_date`, `created_at`, etc.)
- A **sales** column (called anything: `total`, `amount`, `revenue`, etc.)
  — OR — a **quantity** + **unit price** pair to compute sales from

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FULL PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. CSV Upload                                                      │
│     └── Auto-detect columns (date, sales, product, qty, price)     │
│                                                                     │
│  2. Data Aggregation                                                │
│     └── Order lines → daily sales time series                      │
│     └── Fill zero-sales days (asfreq + fillna)                     │
│                                                                     │
│  3. Feature Engineering (25 features)                               │
│     ├── Calendar: day-of-week, month, quarter, day-of-year         │
│     ├── Cyclical: sin/cos encoding (no artificial discontinuities) │
│     ├── Retail flags: Black Friday, Christmas, New Year            │
│     ├── Lag features: sales[-1], [-7], [-14], [-28]                │
│     └── Rolling stats: mean & std over 7, 14, 30 days             │
│                                                                     │
│  4. Model Training                                                  │
│     ├── Chronological train/validation split (last 30 days)        │
│     ├── Gradient Boosting Regressor (200 trees, depth=4)           │
│     └── Refit on full data for final forecasting                   │
│                                                                     │
│  5. Recursive Multi-Step Forecast                                  │
│     └── Predict 1 day → feed back as history → predict next day    │
│                                                                     │
│  6. Output                                                          │
│     ├── Forecast chart (history + future)                          │
│     ├── Validation metrics (MAE, MAPE, R²)                        │
│     ├── Top products table + bar chart                             │
│     ├── Sales pattern charts (weekly + monthly)                    │
│     ├── Inventory reorder recommendation                           │
│     └── Downloadable forecast CSV                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 ML Techniques Used

| Technique | Purpose |
|---|---|
| **Gradient Boosting Regression** | Core forecasting model — ensemble of 200 sequential decision trees |
| **Lag Features** (1/7/14/28 days) | Capture autoregressive patterns (recent sales predict near-future sales) |
| **Rolling Mean & Std** (7/14/30 days) | Capture trend momentum and volatility signals |
| **Cyclical Sin/Cos Encoding** | Represent weekly/yearly periodicity without artificial number-line discontinuities |
| **Retail Holiday Flags** | Inject domain knowledge for Black Friday, Christmas, New Year demand spikes |
| **Chronological Train/Val Split** | Prevent data leakage — always train on past, validate on future |
| **Recursive Multi-Step Forecasting** | Extend a 1-day-ahead model to a 30–90 day forecast horizon |
| **Feature Importance** | Explain which signals drive the model's predictions (interpretability) |

---

## 📁 Project Structure

```
sales-forecasting-tool/
│
├── app.py                    # Streamlit UI — all 4 dashboard tabs
├── forecasting_engine.py     # Core ML pipeline (no Streamlit deps — fully testable)
├── requirements.txt          # Python dependencies
├── README.md                 # You are here
│
└── data/
    └── ecommerce_orders.csv  # 2-year synthetic demo dataset
                              # 36,412 order lines · 12 products · 5 categories
                              # Realistic seasonality: Black Friday 2.8x spike,
                              # Christmas surge, New Year fitness boost,
                              # Valentine's Day gift spike
```

## 📊 Model Performance (on included demo dataset)

| Metric | Value |
|---|---|
| **Validation MAE** | ~$317 / day |
| **Validation MAPE** | ~17% |
| **R² Score** | ~0.78 |
| **Top driving features** | 7-day rolling mean, lag-7, day-of-year (seasonality) |

> Performance on your real store data will vary based on how consistent
> and seasonally predictable your sales patterns are. Stores with 1+ years
> of history and clear seasonal peaks typically see MAPE in the 10–20% range.

---

## 📦 Inventory Planning Formula

The reorder recommendation uses classical inventory management math:

```
Reorder Point = Avg Daily Demand × (Lead Time Days + Safety Stock Days)

Recommended Order Qty = Avg Daily Demand × (Lead Time + Safety Stock + 30 days)
```

`Safety Stock` defaults to 7 days as a buffer against demand spikes
or supplier delays. You can adjust this in `forecasting_engine.py`.

---

📩 **Reach out via [LinkedIn][((https://www.linkedin.com/in/irfan-hyder-03885a142/)] or open an issue.**

