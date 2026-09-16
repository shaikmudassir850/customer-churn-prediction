from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ==========================================
# LOAD SAVED MACHINE LEARNING FILES
# ==========================================

model = joblib.load("best_ada_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
contract_mapping = joblib.load("contract_mapping.pkl")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# CUSTOMER CHURN PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # --------------------------------------
        # Convert contract type using mapping
        # --------------------------------------

        contract_type = contract_mapping[data["contract_type"]]

        # --------------------------------------
        # Create input dataframe
        # --------------------------------------

        input_data = pd.DataFrame([{

            "tenure_months": float(data["tenure_months"]),

            "contract_type": contract_type,

            "monthly_charges": float(data["monthly_charges"]),

            "total_charges": float(data["total_charges"]),

            "payment_method": int(data["payment_method"]),

            "avg_monthly_usage_gb": float(
                data["avg_monthly_usage_gb"]
            ),

            "satisfaction_score": float(
                data["satisfaction_score"]
            ),

            "autopay_enabled": int(
                data["autopay_enabled"]
            )

        }])

        # --------------------------------------
        # EXACT FEATURE ORDER
        # --------------------------------------

        input_data = input_data[
            [
                "tenure_months",
                "contract_type",
                "monthly_charges",
                "total_charges",
                "payment_method",
                "avg_monthly_usage_gb",
                "satisfaction_score",
                "autopay_enabled"
            ]
        ]

        # --------------------------------------
        # SCALE INPUT
        # --------------------------------------

        input_scaled = scaler.transform(input_data)

        # --------------------------------------
        # PREDICTION
        # --------------------------------------

        prediction = model.predict(input_scaled)

        # Convert encoded prediction
        # back to original label
        result = label_encoder.inverse_transform(prediction)[0]

        return jsonify({
            "prediction": str(result)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)