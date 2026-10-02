import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model

model = load_model("laptop_price_model.keras")

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

with open("dropdowns.pkl", "rb") as f:
    dropdowns = pickle.load(f)

scaler = joblib.load("laptop_price_scaler.pkl")

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="⌂",
    layout="centered"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Playfair+Display:wght@500;600&display=swap');

    .stApp {
        background: #20262b;
    }

    .block-container {
        max-width: 900px;
        padding-top: 4rem;
        padding-bottom: 4rem;
    }

    .eyebrow {
        color: #8ea7b8;
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 3px;
        text-align: center;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .main-title {
        color: #e8e5de;
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 52px;
        font-weight: 600;
        line-height: 1.15;
        text-align: center;
        margin-bottom: 12px;
    }

    .subtitle {
        color: #aab1b5;
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        line-height: 1.6;
        text-align: center;
        max-width: 560px;
        margin: 0 auto 38px auto;
    }

    .section-title {
        color: #e1e3e1;
        font-family: 'DM Sans', sans-serif;
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        color: #aeb5b9 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="select"] > div {
        background: #252b30 !important;
        border: 1px solid #3d464d !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] span {
        color: #e5e6e4 !important;
    }

    div[data-testid="stNumberInput"] input {
        background: #252b30 !important;
        color: #e5e6e4 !important;
        border: 1px solid #3d464d !important;
        border-radius: 10px !important;
    }

    div.stButton > button {
        background: #8ea7b8;
        color: #20262b;
        border: none;
        border-radius: 10px;
        height: 48px;
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        font-weight: 600;
        margin-top: 25px;
    }

    div.stButton > button:hover {
        background: #a3b8c6;
        color: #20262b;
        border: none;
    }

    .prediction-card {
        background: #2a3137;
        border: 1px solid #414a51;
        border-radius: 18px;
        padding: 30px;
        margin-top: 35px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
    }

    .prediction-label {
        color: #8ea7b8;
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .prediction-price {
        color: #f0ede6;
        font-family: 'DM Sans', sans-serif;
        font-size: 42px;
        font-weight: 600;
        line-height: 1;
    }

    .summary-card {
        background: #252c32;
        border: 1px solid #3b444b;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
    }

    .summary-title {
        color: #e1e3e1;
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 20px;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        column-gap: 50px;
        row-gap: 12px;
    }

    .summary-item {
        color: #aeb5b9;
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        line-height: 1.6;
    }

    .summary-item strong {
        color: #e1e3e1;
        font-weight: 600;
    }

    .note {
        color: #858e94;
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        text-align: center;
        margin-top: 18px;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="eyebrow">Machine Learning · Regression</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Laptop Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Configure your laptop specifications and get an estimated market price using a deep learning model.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Laptop Specifications</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    company = st.selectbox(
        "Brand",
        dropdowns["Company"]
    )

    type_name = st.selectbox(
        "Laptop Type",
        dropdowns["TypeName"]
    )

    cpu = st.selectbox(
        "CPU Brand",
        dropdowns["Cpu_brand"]
    )

    gpu = st.selectbox(
        "GPU Brand",
        dropdowns["Gpu_brand"]
    )

    os = st.selectbox(
        "Operating System",
        dropdowns["OpSys"]
    )

with col2:

    ram = st.selectbox(
        "RAM (GB)",
        dropdowns["Ram"]
    )

    inches = st.number_input(
        "Screen Size (Inches)",
        min_value=10.0,
        max_value=20.0,
        value=15.6,
        step=0.1
    )

    ssd = st.number_input(
        "SSD (GB)",
        min_value=0,
        max_value=2000,
        value=512,
        step=128
    )

    hdd = st.number_input(
        "HDD (GB)",
        min_value=0,
        max_value=2000,
        value=0,
        step=256
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1
    )

if st.button("Estimate Price", use_container_width=True):

    input_data = {
        "Company": company,
        "TypeName": type_name,
        "Cpu_brand": cpu,
        "Gpu_brand": gpu,
        "OpSys": os,
        "Ram": ram,
        "Inches": inches,
        "SSD": ssd,
        "HDD": hdd,
        "Weight": weight
    }

    input_df = pd.DataFrame([input_data])

    encoded_df = pd.get_dummies(input_df)

    encoded_df = encoded_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    numerical_features = ["Inches", "Ram", "Weight"]

    encoded_df[numerical_features] = scaler.transform(
        encoded_df[numerical_features]
    )

    prediction = model.predict(
        encoded_df,
        verbose=0
    )[0][0]

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">Estimated Laptop Price</div>
            <div class="prediction-price">€{prediction:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    summary_html = f"""
<div class="summary-card"><div class="summary-title">Laptop Summary</div><div class="summary-grid"><div class="summary-item"><strong>Brand:</strong> {company}</div><div class="summary-item"><strong>RAM:</strong> {ram} GB</div><div class="summary-item"><strong>Type:</strong> {type_name}</div><div class="summary-item"><strong>Screen:</strong> {inches}"</div><div class="summary-item"><strong>CPU:</strong> {cpu}</div><div class="summary-item"><strong>SSD:</strong> {ssd} GB</div><div class="summary-item"><strong>GPU:</strong> {gpu}</div><div class="summary-item"><strong>HDD:</strong> {hdd} GB</div><div class="summary-item"><strong>OS:</strong> {os}</div><div class="summary-item"><strong>Weight:</strong> {weight} kg</div></div></div>
"""

    st.markdown(
        summary_html,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="note">Prediction is based on historical data and may vary from actual market prices.</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

st.markdown(
    '<div class="note">Built with Deep Learning · Streamlit</div>',
    unsafe_allow_html=True
)