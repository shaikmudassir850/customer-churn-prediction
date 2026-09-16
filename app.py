from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ============================================================
# LOAD MODEL
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
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "tenure_months":
                float(data["tenure_months"]),

            "contract_type":
                int(data["contract_type"]),

            "monthly_charges":
                float(data["monthly_charges"]),

            "total_charges":
                float(data["total_charges"]),

            "payment_method":
                int(data["payment_method"]),

            "avg_monthly_usage_gb":
                float(data["avg_monthly_usage_gb"]),

            "satisfaction_score":
                float(data["satisfaction_score"]),

            "autopay_enabled":
                int(data["autopay_enabled"])

        }])


        # ----------------------------------------------------
        # SCALE ONLY THE 4 FEATURES USED BY SCALER
        # ----------------------------------------------------

        numeric_columns = [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "avg_monthly_usage_gb"
        ]

        scaled_numeric = pd.DataFrame(
            scaler.transform(
                input_data[numeric_columns]
            ),
            columns=numeric_columns
        )


        # ----------------------------------------------------
        # ADD NON-SCALED FEATURES
        # ----------------------------------------------------

        final_input = pd.concat(
            [
                scaled_numeric,
                input_data[
                    [
                        "contract_type",
                        "payment_method",
                        "satisfaction_score",
                        "autopay_enabled"
                    ]
                ]
            ],
            axis=1
        )


        # ----------------------------------------------------
        # EXACT MODEL FEATURE ORDER
        # ----------------------------------------------------

        final_input = final_input[
            model.feature_names_in_
        ]


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = int(
            model.predict(final_input)[0]
        )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probabilities = model.predict_proba(final_input)[0]

        classes = list(model.classes_)

        churn_probability = 0.0

        if 1 in classes:

            churn_index = classes.index(1)

            churn_probability = (
                float(probabilities[churn_index]) * 100
            )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            message = "Customer likely to leave"

        else:

            message = "Customer likely to stay"


        # ----------------------------------------------------
        # SEND RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "prediction": prediction,

            "message": message,

            "churn_probability":
                round(churn_probability, 2)

        })


    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 400


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
