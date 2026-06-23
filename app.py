"""
Sales Forecasting Tool for Small E-Commerce Stores
====================================================
Upload a Shopify/WooCommerce order export (CSV), get an automated demand
forecast, top-product breakdown, and inventory reorder recommendations.

Run with:
    streamlit run app.py
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from forecasting_engine import (
    detect_columns,
    forecast_future,
    generate_reorder_recommendations,
    load_and_aggregate,
    per_product_summary,
    train_model,
)

st.set_page_config(
    page_title="Sales Forecasting Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 18px;text-align:center}
.metric-card .num{font-size:1.6rem;font-weight:700;color:#1e293b}
.metric-card .lbl{font-size:0.8rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em}
.alert-box{background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:12px 16px;color:#78350f;font-weight:600}
.ok-box{background:#dcfce7;border:1px solid #22c55e;border-radius:8px;padding:12px 16px;color:#14532d;font-weight:600}
.section-header{font-size:1.05rem;font-weight:600;color:#1e293b;margin:0.5rem 0}
</style>
""", unsafe_allow_html=True)

st.title("📈 Sales Forecasting Tool")
st.caption("Upload your Shopify / WooCommerce order export → get a demand forecast, "
          "top-product insights, and inventory reorder recommendations.")
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30, 7)
    st.divider()

    st.markdown("**📦 Inventory Planning (optional)**")
    enable_inventory = st.checkbox("Enable reorder recommendations")
    current_stock = None
    lead_time = None
    if enable_inventory:
        current_stock = st.number_input("Current stock level (units)", 0, 1_000_000, 1000, 100)
        lead_time = st.number_input("Supplier lead time (days)", 1, 180, 14, 1)

    st.divider()
    st.caption(
        "**Expected columns** (auto-detected, flexible naming):\n\n"
        "- Order date\n"
        "- Total sales / order total\n"
        "- Quantity + unit price (used if no total column)\n"
        "- Product name (optional, for top-product breakdown)"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


def df_to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def plot_forecast(daily_df, forecast_df, history_days=90):
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_color("#f8fafc")
    ax.set_facecolor("#f8fafc")

    hist = daily_df.tail(history_days)
    ax.plot(hist["date"], hist["sales"], color="#3b82f6", lw=1.5, label="Historical Sales")
    ax.plot(forecast_df["date"], forecast_df["forecast"], color="#f97316", lw=2,
            linestyle="--", label="Forecast")

    # Connect last historical point to first forecast point
    ax.plot(
        [hist["date"].iloc[-1], forecast_df["date"].iloc[0]],
        [hist["sales"].iloc[-1], forecast_df["forecast"].iloc[0]],
        color="#f97316", lw=2, linestyle="--"
    )

    ax.axvline(hist["date"].iloc[-1], color="#94a3b8", linestyle=":", lw=1)
    ax.set_ylabel("Sales ($)")
    ax.set_title("Sales History & Forecast", fontsize=12, fontweight="600")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_weekly_pattern(daily_df):
    d = daily_df.copy()
    d["dayofweek"] = d["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avg = d.groupby("dayofweek")["sales"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_color("#f8fafc")
    ax.set_facecolor("#f8fafc")
    colors = ["#3b82f6" if d not in ("Saturday", "Sunday") else "#f97316" for d in order]
    ax.bar(avg.index, avg.values, color=colors, alpha=0.85)
    ax.set_ylabel("Avg Daily Sales ($)")
    ax.set_title("Average Sales by Day of Week", fontsize=11, fontweight="600")
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_monthly_trend(daily_df):
    d = daily_df.copy()
    d["month"] = d["date"].dt.to_period("M")
    monthly = d.groupby("month")["sales"].sum()

    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.patch.set_color("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.bar(monthly.index.astype(str), monthly.values, color="#3b82f6", alpha=0.85)
    ax.set_ylabel("Total Sales ($)")
    ax.set_title("Monthly Sales Trend", fontsize=11, fontweight="600")
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_feature_importance(feat_imp, top_n=10):
    names = [x[0] for x in feat_imp[:top_n]]
    vals = [x[1] for x in feat_imp[:top_n]]
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_color("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.barh(names[::-1], vals[::-1], color="#3b82f6", alpha=0.8)
    ax.set_xlabel("Importance")
    ax.set_title("What Drives the Forecast", fontsize=11, fontweight="600")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your order export (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("👆 Upload a CSV export from Shopify, WooCommerce, or any order system "
            "with a date column and a sales total (or quantity + price).")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📊 Demand Forecasting**")
        st.write("Gradient Boosting model trained on your sales history, with "
                "engineered seasonality features (day-of-week, monthly, holiday spikes).")
    with c2:
        st.markdown("**🏆 Product Insights**")
        st.write("See your best-selling products by revenue and units sold.")
    with c3:
        st.markdown("**📦 Inventory Alerts**")
        st.write("Enter your current stock and supplier lead time to get a "
                "recommended reorder date and quantity.")

    st.divider()
    st.caption("Need a sample file to try? Use the included `data/ecommerce_orders.csv` "
              "(2 years of synthetic order data with realistic seasonality).")

else:
    raw_df = load_csv(uploaded_file)
    col_map = detect_columns(raw_df)

    with st.expander("🔍 Detected columns (click to verify)"):
        st.write(col_map)
        if col_map["date"] is None:
            st.error("Could not detect a date column. Please ensure your CSV has a "
                    "column like 'Order Date', 'Date', or 'Created At'.")
            st.stop()
        if col_map["sales"] is None and (col_map["quantity"] is None or col_map["unit_price"] is None):
            st.error("Could not detect a sales total column, and no quantity+price "
                    "columns to compute it from.")
            st.stop()

    try:
        daily = load_and_aggregate(raw_df, col_map)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    n_days = len(daily)
    date_min, date_max = daily["date"].min(), daily["date"].max()

    st.markdown('<div class="section-header">📋 Data Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="num">{n_days:,}</div>'
                    f'<div class="lbl">Days of History</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="num">${daily["sales"].sum():,.0f}</div>'
                    f'<div class="lbl">Total Sales</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="num">${daily["sales"].mean():,.0f}</div>'
                    f'<div class="lbl">Avg Daily Sales</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="num">{date_min.date()} → {date_max.date()}</div>'
                    f'<div class="lbl">Date Range</div></div>', unsafe_allow_html=True)

    st.markdown("")

    if n_days < 45:
        st.warning(
            f"⚠️ Only {n_days} days of history found. Forecasting works best with "
            "45+ days of data (the model uses up to 30-day rolling averages and "
            "28-day lags). Results may be unreliable."
        )

    # ── Train model & forecast ──
    with st.spinner("Training forecasting model..."):
        try:
            model, metrics, featured = train_model(daily)
            forecast = forecast_future(model, daily, horizon=horizon)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    st.success("✅ Forecast generated!")

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast", "📊 Sales Patterns", "🏆 Top Products", "📦 Inventory Planning"
    ])

    # ── TAB 1: Forecast ──
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="num">${forecast["forecast"].sum():,.0f}</div>'
                        f'<div class="lbl">{horizon}-Day Forecast Total</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="num">${forecast["forecast"].mean():,.0f}</div>'
                        f'<div class="lbl">Avg Daily Forecast</div></div>', unsafe_allow_html=True)
        with c3:
            if metrics.get("mape") is not None:
                st.markdown(f'<div class="metric-card"><div class="num">{metrics["mape"]*100:.1f}%</div>'
                            f'<div class="lbl">Validation MAPE</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card"><div class="num">R²={metrics.get("r2",0):.2f}</div>'
                            f'<div class="lbl">Model Fit</div></div>', unsafe_allow_html=True)

        st.markdown("")
        st.pyplot(plot_forecast(daily, forecast), use_container_width=True)

        with st.expander("📐 Model performance details"):
            cc1, cc2 = st.columns(2)
            with cc1:
                st.write(f"**MAE:** ${metrics.get('mae', 0):,.2f}")
                if metrics.get("mape") is not None:
                    st.write(f"**MAPE:** {metrics['mape']*100:.2f}%")
                st.write(f"**R²:** {metrics.get('r2', 0):.4f}")
                st.write(f"**Validated on:** last {metrics.get('validation_days', 0)} days")
            with cc2:
                st.pyplot(plot_feature_importance(metrics["feature_importance"]))

        with st.expander("📥 Forecast data table"):
            display_fc = forecast.copy()
            display_fc["forecast"] = display_fc["forecast"].round(2)
            display_fc.columns = ["Date", "Forecasted Sales ($)"]
            st.dataframe(display_fc, use_container_width=True, hide_index=True)

        # Download button
        download_df = forecast.copy()
        download_df.columns = ["date", "forecasted_sales"]
        download_df["forecasted_sales"] = download_df["forecasted_sales"].round(2)
        st.download_button(
            "📥 Download Forecast CSV",
            data=df_to_csv_bytes(download_df),
            file_name="sales_forecast.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )

    # ── TAB 2: Sales Patterns ──
    with tab2:
        col_l, col_r = st.columns([1, 1.3])
        with col_l:
            st.pyplot(plot_weekly_pattern(daily), use_container_width=True)
        with col_r:
            st.pyplot(plot_monthly_trend(daily), use_container_width=True)

    # ── TAB 3: Top Products ──
    with tab3:
        if col_map.get("product"):
            top_products = per_product_summary(raw_df, col_map, top_n=15)
            if len(top_products):
                top_products_display = top_products.copy()
                top_products_display.columns = ["Product", "Total Sales ($)", "Units Sold", "# Orders"]
                top_products_display["Total Sales ($)"] = top_products_display["Total Sales ($)"].round(2)
                st.dataframe(top_products_display, use_container_width=True, hide_index=True)

                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_color("#f8fafc")
                ax.set_facecolor("#f8fafc")
                top10 = top_products.head(10)
                ax.barh(top10[col_map["product"]][::-1], top10["total_sales"][::-1],
                       color="#3b82f6", alpha=0.85)
                ax.set_xlabel("Total Sales ($)")
                ax.set_title("Top 10 Products by Revenue", fontsize=11, fontweight="600")
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("No product-level data available.")
        else:
            st.info("No product column detected in your file — upload a CSV with a "
                   "'Product' or 'Lineitem name' column to see top-product insights.")

    # ── TAB 4: Inventory Planning ──
    with tab4:
        if enable_inventory and current_stock is not None:
            rec = generate_reorder_recommendations(forecast, current_stock, lead_time)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="num">{rec["avg_daily_demand"]:.1f}</div>'
                            f'<div class="lbl">Avg Daily Demand (units-equiv.)</div></div>', unsafe_allow_html=True)
                if rec["days_until_stockout"] is not None:
                    st.markdown(f'<div class="metric-card"><div class="num">{rec["days_until_stockout"]} days</div>'
                                f'<div class="lbl">Until Projected Stockout</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><div class="num">No stockout</div>'
                                f'<div class="lbl">Within forecast horizon</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="num">{rec["reorder_point"]:,.0f}</div>'
                            f'<div class="lbl">Reorder Point (units-equiv.)</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="num">{rec["recommended_order_qty"]:,.0f}</div>'
                            f'<div class="lbl">Recommended Order Qty</div></div>', unsafe_allow_html=True)

            st.markdown("")
            if rec["should_reorder_now"]:
                st.markdown(
                    f'<div class="alert-box">⚠️ Stock level ({current_stock:,.0f}) is at or below '
                    f'the reorder point ({rec["reorder_point"]:,.0f}). '
                    f'Place an order now to avoid running out before your '
                    f'{lead_time}-day lead time elapses.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="ok-box">✅ Stock level ({current_stock:,.0f}) is above '
                    f'the reorder point ({rec["reorder_point"]:,.0f}). No action needed yet.</div>',
                    unsafe_allow_html=True
                )

            st.caption(
                "**Note:** Demand here is expressed in the same units as your sales "
                "total (i.e. dollar-equivalent if your file has revenue per order). "
                "For unit-based inventory planning, ensure your file has a quantity "
                "column and that 'current stock' is in matching units."
            )
        else:
            st.info("Enable **'Inventory Planning'** in the sidebar and enter your "
                   "current stock level + supplier lead time to get reorder recommendations.")

st.divider()
st.caption(
    "💡 **Tip:** For best results, export at least 2-3 months of order history. "
    "More history → better seasonality detection → more accurate forecasts."
)
