from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


# ============================================================
# LOAD MODEL AND SCALER
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

        # ----------------------------------------------------
        # GET DATA FROM HTML
        # ----------------------------------------------------

        data = request.get_json()


        # ----------------------------------------------------
        # NUMERIC FEATURES
        # ----------------------------------------------------

        numeric_data = pd.DataFrame([{

            "tenure_months":
                float(data["tenure_months"]),

            "monthly_charges":
                float(data["monthly_charges"]),

            "total_charges":
                float(data["total_charges"]),

            "avg_monthly_usage_gb":
                float(data["avg_monthly_usage_gb"])

        }])


        # ----------------------------------------------------
        # SCALE ONLY THE NUMERIC FEATURES
        # ----------------------------------------------------

        numeric_scaled = scaler.transform(numeric_data)


        # Convert scaled values back to DataFrame
        numeric_scaled = pd.DataFrame(
            numeric_scaled,
            columns=[
                "tenure_months",
                "monthly_charges",
                "total_charges",
                "avg_monthly_usage_gb"
            ]
        )


        # ----------------------------------------------------
        # CATEGORICAL / ALREADY ENCODED FEATURES
        # ----------------------------------------------------

        categorical_data = pd.DataFrame([{

            "contract_type":
                int(data["contract_type"]),

            "payment_method":
                int(data["payment_method"]),

            "satisfaction_score":
                float(data["satisfaction_score"]),

            "autopay_enabled":
                int(data["autopay_enabled"])

        }])


        # ----------------------------------------------------
        # COMBINE FEATURES
        # ----------------------------------------------------

        input_data = pd.concat(
            [
                numeric_scaled,
                categorical_data
            ],
            axis=1
        )


        # ----------------------------------------------------
        # EXACT FEATURE ORDER
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = int(
            model.predict(input_data)[0]
        )


        # ----------------------------------------------------
        # CHURN PROBABILITY
        # ----------------------------------------------------

        churn_probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            classes = list(model.classes_)

            if 1 in classes:

                churn_index = classes.index(1)

                churn_probability = (
                    probabilities[churn_index] * 100
                )


        # ----------------------------------------------------
        # RESULT
        #
        # 1 = CHURN / LEAVE
        # 0 = STAY
        # ----------------------------------------------------

        if prediction == 1:

            message = "Customer likely to leave"

        else:

            message = "Customer likely to stay"


        # ----------------------------------------------------
        # SEND RESULT TO HTML
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
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
