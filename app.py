from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ============================================================
# LOAD SAVED MODEL AND SCALER
# ============================================================

model = joblib.load("best_ada_model.pkl")
scaler = joblib.load("scaler.pkl")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# CUSTOMER CHURN PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # ----------------------------------------------------
        # GET INPUT VALUES
        # HTML ALREADY SENDS ENCODED VALUES
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "tenure_months": float(
                data["tenure_months"]
            ),

            "contract_type": int(
                data["contract_type"]
            ),

            "monthly_charges": float(
                data["monthly_charges"]
            ),

            "total_charges": float(
                data["total_charges"]
            ),

            "payment_method": int(
                data["payment_method"]
            ),

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


        # ----------------------------------------------------
        # EXACT FEATURE ORDER USED FOR TRAINING
        # ----------------------------------------------------

        feature_order = [
            "tenure_months",
            "contract_type",
            "monthly_charges",
            "total_charges",
            "payment_method",
            "avg_monthly_usage_gb",
            "satisfaction_score",
            "autopay_enabled"
        ]

        input_data = input_data[feature_order]


        # ----------------------------------------------------
        # SCALE INPUT
        # ----------------------------------------------------

        input_scaled = scaler.transform(input_data)


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = int(
            model.predict(input_scaled)[0]
        )


        # ----------------------------------------------------
        # CHURN PROBABILITY
        # ----------------------------------------------------

        churn_probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_scaled
            )[0]

            classes = list(model.classes_)

            if 1 in classes:

                churn_index = classes.index(1)

                churn_probability = (
                    probabilities[churn_index] * 100
                )


        # ----------------------------------------------------
        # FINAL RESULT
        #
        # 1 = CUSTOMER WILL LEAVE
        # 0 = CUSTOMER WILL STAY
        # ----------------------------------------------------

        if prediction == 1:

            message = "Customer likely to leave"

        else:

            message = "Customer likely to stay"


        # ----------------------------------------------------
        # SEND RESPONSE TO HTML
        # ----------------------------------------------------

        return jsonify({

            "prediction": prediction,

            "message": message,

            "churn_probability":
                round(churn_probability, 2)
                if churn_probability is not None
                else None

        })


    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 400


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
