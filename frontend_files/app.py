import os

import pandas as pd
import requests
import streamlit as st

# The backend is reached through the Docker network by its container name.
# Override with the BACKEND_URL environment variable if it runs somewhere else.
BACKEND_URL = os.getenv("BACKEND_URL", "http://superkart-backend:7860").rstrip("/")
PREDICT_URL = f"{BACKEND_URL}/v1/predict"
BATCH_URL = f"{BACKEND_URL}/v1/predictbatch"

FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
    "Store_Age_Years", "Product_Type_Category",
]

st.set_page_config(page_title="SuperKart Sales Forecaster", page_icon="🛒", layout="wide")
st.title("🛒 SuperKart Sales Forecaster")
st.caption("Predict the total sales of a product in a SuperKart store, one at a time or in bulk.")

with st.sidebar:
    st.header("Backend status")
    try:
        requests.get(BACKEND_URL, timeout=3).raise_for_status()
        st.success("Prediction API is reachable")
    except requests.RequestException:
        st.error("Prediction API is not reachable")
    st.caption(f"Endpoint: {BACKEND_URL}")

single_tab, batch_tab = st.tabs(["Single prediction", "Batch prediction"])

# ------------------------------------------------------------------ single prediction
with single_tab:
    with st.form("single_prediction"):
        product_col, store_col = st.columns(2)

        with product_col:
            st.subheader("Product")
            product_weight = st.number_input("Product weight", 1.0, 30.0, 12.66, 0.01)
            sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
            area = st.number_input("Allocated display area (ratio)", 0.001, 0.5, 0.056, 0.001, format="%.3f")
            mrp = st.number_input("Maximum retail price (MRP)", 20.0, 300.0, 146.74, 0.01)
            id_char = st.selectbox("Product ID prefix", ["FD", "DR", "NC"],
                                   help="FD = food, DR = drinks, NC = non-consumable")
            category = st.selectbox("Product category", ["Perishables", "Non Perishables"])

        with store_col:
            st.subheader("Store")
            store_type = st.selectbox("Store type", ["Supermarket Type1", "Supermarket Type2",
                                                     "Departmental Store", "Food Mart"], index=1)
            store_size = st.selectbox("Store size", ["Small", "Medium", "High"], index=1)
            city_type = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
            store_age = st.number_input("Store age (years)", 0, 60, 16, 1)

        submitted = st.form_submit_button("Predict sales", type="primary")

    if submitted:
        payload = {
            "Product_Weight": product_weight, "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": area, "Product_MRP": mrp, "Store_Size": store_size,
            "Store_Location_City_Type": city_type, "Store_Type": store_type,
            "Product_Id_char": id_char, "Store_Age_Years": int(store_age),
            "Product_Type_Category": category,
        }
        try:
            response = requests.post(PREDICT_URL, json=payload, timeout=30)
            if response.ok:
                result = response.json()
                st.metric("Predicted sales for this product in this store", f"{result['Predicted_Sales']:,.2f}")
                for note in result.get("Warnings", []):
                    st.warning(note)
            else:
                st.error(response.json().get("error", f"Request failed ({response.status_code})"))
        except requests.RequestException as exc:
            st.error(f"Could not reach the prediction API: {exc}")

# ------------------------------------------------------------------ batch prediction
with batch_tab:
    st.write("Upload a CSV with one row per product-store combination. Required columns:")
    st.code(", ".join(FEATURES))

    sample = pd.DataFrame([{
        "Product_Weight": 12.66, "Product_Sugar_Content": "Low Sugar", "Product_Allocated_Area": 0.056,
        "Product_MRP": 146.74, "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2",
        "Store_Type": "Supermarket Type2", "Product_Id_char": "FD", "Store_Age_Years": 16,
        "Product_Type_Category": "Perishables",
    }])
    st.download_button("Download CSV template", sample.to_csv(index=False), "superkart_batch_template.csv", "text/csv")

    uploaded = st.file_uploader("Batch file (CSV)", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Preview ({len(batch_df)} rows)")
        st.dataframe(batch_df.head(10), width="stretch")

        if st.button("Run batch prediction", type="primary"):
            try:
                response = requests.post(
                    BATCH_URL, files={"file": ("batch.csv", batch_df.to_csv(index=False).encode("utf-8"), "text/csv")},
                    timeout=120,
                )
                if response.ok:
                    predictions = pd.Series(response.json())
                    predictions.index = predictions.index.astype(int)
                    output = batch_df.copy()
                    output["Predicted_Sales"] = predictions.reindex(output.index)

                    st.success("Predictions ready")
                    left, right = st.columns(2)
                    left.metric("Rows scored", f"{len(output):,}")
                    right.metric("Total predicted sales", f"{output['Predicted_Sales'].sum():,.2f}")
                    for note in filter(None, response.headers.get("X-Warnings", "").split(" | ")):
                        st.warning(note)
                    st.dataframe(output, width="stretch")
                    st.download_button("Download predictions", output.to_csv(index=False),
                                       "superkart_predictions.csv", "text/csv")
                else:
                    st.error(response.json().get("error", f"Request failed ({response.status_code})"))
            except requests.RequestException as exc:
                st.error(f"Could not reach the prediction API: {exc}")
