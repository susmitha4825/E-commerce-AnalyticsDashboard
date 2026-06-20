"""
app/streamlit.py  — E-commerce Analytics & Prediction Dashboard
Run: streamlit run app/streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-commerce Analytics",
    page_icon="🛒",
    layout="wide",
)

# ── Gradient background CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background gradient */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}

/* Sidebar gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a3e 0%, #2d2b55 100%);
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(8px);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #185FA5, #378ADD);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 2rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #378ADD, #185FA5);
    transform: translateY(-1px);
}

/* Card boxes */
.card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(6px);
}

/* Prediction result */
.result-good {
    background: linear-gradient(135deg, #0F6E56, #1D9E75);
    color: white;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 600;
}
.result-bad {
    background: linear-gradient(135deg, #993C1D, #D85A30);
    color: white;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 600;
}

h1, h2, h3 { color: #ffffff !important; }
p, label { color: #d0d0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data & model ─────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_data():
    path = os.path.join(BASE, "ecommerce_orders_cleaned.csv")
    if not os.path.exists(path):
        path = os.path.join(BASE, "ecommerce_orders.csv")
    return pd.read_csv(path, parse_dates=["order_date"])

@st.cache_resource
def load_model():
    try:
        with open(os.path.join(BASE, "models", "loan_model.pkl"), "rb") as f:
            m = pickle.load(f)
        with open(os.path.join(BASE, "models", "scaler.pkl"), "rb") as f:
            s = pickle.load(f)
        imputer_path = os.path.join(BASE, "models", "imputer.pkl")
        imp = pickle.load(open(imputer_path, "rb")) if os.path.exists(imputer_path) else None
        return m, s, imp
    except FileNotFoundError:
        return None, None, None

df = load_data()
model, scaler, imputer = load_model()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🛒 E-commerce App")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Predict Order"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df)} orders")
st.sidebar.markdown(f"**Columns:** {df.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 E-commerce Analytics Dashboard")
    st.markdown("Real-time insights from your order data.")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Revenue", f"₹{df['revenue'].sum():,.0f}")
    with col2:
        st.metric("🛒 Total Orders", len(df))
    with col3:
        delivery_rate = round((df['order_status'] == 'Delivered').mean() * 100, 1)
        st.metric("✅ Delivery Rate", f"{delivery_rate}%")
    with col4:
        avg_order = round(df['revenue'].mean(), 0)
        st.metric("📦 Avg Order Value", f"₹{avg_order:,.0f}")

    st.markdown("---")

    # Charts row 1
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📅 Monthly Revenue Trend")
        if 'month' not in df.columns:
            df['month'] = pd.to_datetime(df['order_date']).dt.month
        monthly = df.groupby('month')['revenue'].sum()
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        ax.plot(months[:len(monthly)], monthly.values, marker='o',
                color='#378ADD', linewidth=2)
        ax.fill_between(months[:len(monthly)], monthly.values,
                        alpha=0.15, color='#378ADD')
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor((1,1,1,0.15))
        st.pyplot(fig)

    with c2:
        st.subheader("🏷️ Revenue by Category")
        cat_rev = df.groupby('category')['revenue'].sum().sort_values()
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor('none')
        colors = ['#185FA5','#0F6E56','#D85A30','#7F77DD','#1D9E75','#BA7517']
        ax2.barh(cat_rev.index, cat_rev.values, color=colors[:len(cat_rev)])
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor((1,1,1,0.15))
        st.pyplot(fig2)

    # Charts row 2
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("📦 Order Status")
        status = df['order_status'].value_counts()
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        fig3.patch.set_alpha(0)
        ax3.set_facecolor('none')
        ax3.pie(status.values, labels=status.index,
                autopct='%1.1f%%',
                colors=['#1D9E75','#378ADD','#E24B4A','#EF9F27'],
                textprops={'color': 'white'})
        st.pyplot(fig3)

    with c4:
        st.subheader("💳 Payment Methods")
        pay = df['payment_method'].value_counts()
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        fig4.patch.set_alpha(0)
        ax4.set_facecolor('none')
        ax4.bar(pay.index, pay.values,
                color=['#185FA5','#0F6E56','#7F77DD','#D85A30','#BA7517'])
        ax4.tick_params(colors='white', axis='both')
        plt.xticks(rotation=20)
        for spine in ax4.spines.values():
            spine.set_edgecolor((1,1,1,0.15))
        st.pyplot(fig4)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🔮 Order Outcome Predictor")
    st.markdown("Fill in order details to predict whether the order will be delivered or returned.")

    if model is None:
        st.warning("⚠️ Model not found. Please run the Training notebook first to generate model files.")
    else:
        # Encoding maps shown to user
        cat_map     = {"Electronics":0,"Furniture":1,"Footwear":2,"Clothing":3,"Accessories":4,"Books":5}
        city_map    = {"Mumbai":0,"Delhi":1,"Hyderabad":2,"Bangalore":3,"Chennai":4,
                       "Pune":5,"Kolkata":6,"Ahmedabad":7,"Jaipur":8,"Lucknow":9}
        gender_map  = {"Female":0,"Male":1}
        payment_map = {"Cash on Delivery":0,"UPI":1,"Credit Card":2,"Debit Card":3,"Net Banking":4}

        with st.form("predict_form"):
            st.subheader("👤 Customer Info")
            c1, c2, c3 = st.columns(3)
            with c1:
                age    = st.number_input("Age", 18, 80, 30)
                gender = st.selectbox("Gender", list(gender_map.keys()))
            with c2:
                city     = st.selectbox("City", list(city_map.keys()))
                category = st.selectbox("Category", list(cat_map.keys()))
            with c3:
                payment = st.selectbox("Payment Method", list(payment_map.keys()))
                month   = st.slider("Order Month", 1, 12, 6)

            st.subheader("🛒 Order Details")
            c4, c5, c6 = st.columns(3)
            with c4:
                quantity = st.number_input("Quantity", 1, 20, 2)
                unit_price = st.number_input("Unit Price (₹)", 100.0, 200000.0, 1500.0)
            with c5:
                discount_pct = st.slider("Discount %", 0.0, 50.0, 10.0)
                discount_amount = round(unit_price * quantity * discount_pct / 100, 2)
                st.metric("Discount Amount", f"₹{discount_amount:,.2f}")
            with c6:
                revenue = round(unit_price * quantity - discount_amount, 2)
                st.metric("Final Revenue", f"₹{revenue:,.2f}")
                product_enc = st.number_input("Product Code (0–20)", 0, 20, 0)

            submitted = st.form_submit_button("🔮 Predict")

        if submitted:
            features = np.array([[
                age, quantity, unit_price, discount_pct,
                discount_amount, revenue, month,
                cat_map[category], city_map[city],
                gender_map[gender], payment_map[payment],
                product_enc
            ]])
            if imputer is not None:
                features = imputer.transform(features)
            features_scaled = scaler.transform(features)
            prediction  = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            confidence  = round(float(max(probability)) * 100, 2)
            label = "✅ Delivered / Shipped" if prediction == 1 else "❌ Returned / Cancelled"
            css_class = "result-good" if prediction == 1 else "result-bad"

            st.markdown(f"""
            <div class="{css_class}">
                {label}<br>
                <small>Confidence: {confidence}%</small>
            </div>
            """, unsafe_allow_html=True)

            # Probability bar
            st.markdown("#### Probability Breakdown")
            prob_df = pd.DataFrame({
                "Outcome": ["Returned / Cancelled", "Delivered / Shipped"],
                "Probability": [round(probability[0]*100,1), round(probability[1]*100,1)]
            })
            st.bar_chart(prob_df.set_index("Outcome"))
