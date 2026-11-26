import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Flask application for the SuperKart sales forecaster
superkart_api = Flask("SuperKart Sales Forecaster")

# Load the serialized pipeline (preprocessing + tuned model) once, at start-up
model = joblib.load("superkart_model.joblib")

NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
    "Store_Age_Years", "Product_Type_Category",
]

# Categories seen during training, used to warn about values the model has never seen
_column_transformer = model.named_steps["columntransformer"]
_cat_columns = _column_transformer.transformers_[1][2]
_encoder = _column_transformer.named_transformers_["onehotencoder"]
KNOWN_CATEGORIES = {col: set(cats) for col, cats in zip(_cat_columns, _encoder.categories_)}


def prepare_input(df):
    # Validates the incoming data and returns (clean_dataframe, error_message)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {missing}"
    df = df[FEATURES].copy()
    if df.empty:
        return None, "No rows to predict."
    try:
        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col])
    except (ValueError, TypeError) as exc:
        return None, f"Numeric columns must contain numbers only ({exc})."
    if df.isnull().any().any():
        return None, "Input contains missing values; please fill every feature."
    return df, None


def unseen_categories(df):
    # Lists category values the model was not trained on (their store/product effect is ignored)
    notes = []
    for col, known in KNOWN_CATEGORIES.items():
        for value in sorted(set(df[col].astype(str)) - known):
            notes.append(f"{col}='{value}' was not seen in training")
    return notes


@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Forecasting API!"


@superkart_api.post("/v1/predict")
def predict_sales():
    # Online inference: one JSON object with the 10 model features
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    input_df, error = prepare_input(pd.DataFrame([payload]))
    if error:
        return jsonify({"error": error}), 400

    prediction = float(model.predict(input_df)[0])
    response = {"Predicted_Sales": round(prediction, 2)}
    notes = unseen_categories(input_df)
    if notes:
        response["Warnings"] = notes
    return jsonify(response)


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    # Batch inference: a CSV uploaded under the form key "file"
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "Upload a CSV file using the form key 'file'."}), 400
    try:
        batch_df = pd.read_csv(uploaded)
    except Exception:
        return jsonify({"error": "The uploaded file could not be read as a CSV."}), 400

    input_df, error = prepare_input(batch_df)
    if error:
        return jsonify({"error": error}), 400

    predictions = model.predict(input_df)
    # {row_index: predicted_sales}
    response = jsonify({str(idx): round(float(p), 2) for idx, p in zip(input_df.index, predictions)})
    notes = unseen_categories(input_df)
    if notes:
        response.headers["X-Warnings"] = " | ".join(notes)
    return response


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
