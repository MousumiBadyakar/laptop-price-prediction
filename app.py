import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model

# ----------------------------------
# Load model and preprocessing files
# ----------------------------------

model = load_model("laptop_price_model.keras")

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

with open("dropdowns.pkl", "rb") as f:
    dropdowns = pickle.load(f)

scaler = joblib.load("laptop_price_scaler.pkl")


# ----------------------------------
# Streamlit Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

st.title("💻 Laptop Price Prediction")
st.write("Enter laptop specifications to estimate its price.")


# ----------------------------------
# User Inputs
# ----------------------------------

st.subheader("🔧 Laptop Specifications")

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


# ----------------------------------
# Prediction
# ----------------------------------

if st.button("🔮 Predict Price"):

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

    # One-hot encoding
    encoded_df = pd.get_dummies(input_df)

    # Make columns exactly the same as training data
    encoded_df = encoded_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Scale numerical features
    numerical_features = ["Inches", "Ram", "Weight"]

    encoded_df[numerical_features] = scaler.transform(
        encoded_df[numerical_features]
    )

    # Prediction
    prediction = model.predict(encoded_df, verbose=0)[0][0]

    st.success(
        f"💰 Estimated Laptop Price: €{prediction:,.2f}"
    )

    st.caption(
        "⚠️ Prediction is based on historical data and may vary."
    )



# ----------------------------------
# Footer
# ----------------------------------

st.markdown("---")

st.markdown(
    "Built with ❤️ using Deep Learning"
)