# 🛒 E-commerce Order Analytics & Predictor

A beginner-friendly end-to-end data science project that analyzes e-commerce orders and predicts whether an order will be **Delivered** or **Returned/Cancelled** using Machine Learning.

---

## 📁 Project Structure

```
ecommerce_project/
│
├── Notebooks/
│   ├── Datacleaning.ipynb      # Step 1 — Clean raw data
│   ├── EDA.ipynb               # Step 2 — Explore & visualize
│   ├── Training_data.ipynb     # Step 3 — Train ML model
│   └── Advanced.ipynb          # Step 4 — Tune & evaluate
│
├── models/
│   ├── loan_model.pkl          # Trained Random Forest model
│   ├── scaler.pkl              # StandardScaler
│   └── feature_names.pkl      # Feature column names
│
├── app/
│   ├── api.py                  # FastAPI REST backend
│   └── streamlit.py            # Streamlit dashboard frontend
│
├── ecommerce_orders.csv        # Raw dataset (508 orders)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 How to Run (Step by Step)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebooks in order
Open Jupyter and run notebooks in this order:
```
Datacleaning.ipynb → EDA.ipynb → Training_data.ipynb → Advanced.ipynb
```

### 3. Start the Streamlit app
```bash
streamlit run app/streamlit.py
```

### 4. (Optional) Start the FastAPI backend
```bash
uvicorn app.api:app --reload
```
API docs available at: `http://localhost:8000/docs`

---

## 📊 Dataset Overview

| Column | Description |
|---|---|
| order_id | Unique order identifier |
| order_date | Date the order was placed |
| customer_age | Age of the customer |
| customer_gender | Male / Female |
| city | Customer's city (10 cities) |
| category | Product category (6 types) |
| product | Product name |
| quantity | Units ordered |
| unit_price | Price per unit (₹) |
| discount_pct | Discount percentage |
| discount_amount | Discount in rupees |
| revenue | Final order revenue (₹) |
| payment_method | Payment type used |
| order_status | Delivered / Shipped / Returned / Cancelled |

**Total records:** 508 orders across 10 Indian cities

---

## 🤖 ML Model

| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Target | Delivered/Shipped (1) vs Returned/Cancelled (0) |
| Features | 12 (age, revenue, category, city, payment, etc.) |
| Evaluation | Cross-validation, Confusion Matrix, ROC-AUC |

---

## 🎨 Features

- **Gradient UI** — Deep purple-blue gradient background in Streamlit
- **Interactive Dashboard** — Monthly trends, category revenue, order status, city rankings
- **ML Prediction** — Input order details and get delivery outcome prediction with confidence score
- **REST API** — FastAPI backend with `/predict` endpoint
- **Beginner-friendly notebooks** — Each step clearly explained with comments

---

## 👨‍💻 Tech Stack

| Layer | Technology |
|---|---|
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | Scikit-learn |
| Frontend | Streamlit |
| Backend API | FastAPI |

---

*Built as a beginner data science project submission.*
