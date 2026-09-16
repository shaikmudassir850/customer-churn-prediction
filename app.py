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
# ENCODING MAPPINGS USED DURING TRAINING
# ============================================================

contract_mapping = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_mapping = {
    "Bank transfer": 0,
    "Credit card": 1,
    "Electronic check": 2,
    "Mailed check": 3
}


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
        # RECEIVE DATA
        # ----------------------------------------------------

        data = request.get_json()

        print("====================================")
        print("RECEIVED DATA:")
        print(data)
        print("====================================")

        if data is None:
            return jsonify({
                "error": "No JSON data received"
            }), 400


        # ----------------------------------------------------
        # CONTRACT TYPE
        # Accept text OR numeric value
        # ----------------------------------------------------

        contract_value = data["contract_type"]

        if isinstance(contract_value, str):

            if contract_value in contract_mapping:
                contract_type = contract_mapping[contract_value]
            else:
                contract_type = int(contract_value)

        else:
            contract_type = int(contract_value)


        # ----------------------------------------------------
        # PAYMENT METHOD
        # Accept text OR numeric value
        # ----------------------------------------------------

        payment_value = data["payment_method"]

        if isinstance(payment_value, str):

            if payment_value in payment_mapping:
                payment_method = payment_mapping[payment_value]
            else:
                payment_method = int(payment_value)

        else:
            payment_method = int(payment_value)


        # ----------------------------------------------------
        # CREATE RAW INPUT
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "tenure_months":
                float(data["tenure_months"]),

            "contract_type":
                contract_type,

            "monthly_charges":
                float(data["monthly_charges"]),

            "total_charges":
                float(data["total_charges"]),

            "payment_method":
                payment_method,

            "avg_monthly_usage_gb":
                float(data["avg_monthly_usage_gb"]),

            "satisfaction_score":
                float(data["satisfaction_score"]),

            "autopay_enabled":
                int(data["autopay_enabled"])

        }])


        print("RAW INPUT:")
        print(input_data)


        # ----------------------------------------------------
        # NUMERIC COLUMNS THAT WERE SCALED DURING TRAINING
        # ----------------------------------------------------

        numeric_columns = [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "avg_monthly_usage_gb"
        ]


        # ----------------------------------------------------
        # SCALE ONLY THOSE 4 COLUMNS
        # ----------------------------------------------------

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
        # EXACT FEATURE ORDER USED BY ADA BOOST
        # ----------------------------------------------------

        final_input = final_input[
            model.feature_names_in_
        ]


        print("FINAL INPUT:")
        print(final_input)


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = int(
            model.predict(final_input)[0]
        )


        # ----------------------------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            final_input
        )[0]

        classes = list(model.classes_)


        # Probability of class 1 = customer leaving

        if 1 in classes:

            churn_probability = (
                float(
                    probabilities[
                        classes.index(1)
                    ]
                ) * 100
            )

        else:

            churn_probability = 0.0


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            message = "Customer likely to leave"

        else:

            message = "Customer likely to stay"


        # ----------------------------------------------------
        # PRINT RESULT TO RENDER LOGS
        # ----------------------------------------------------

        print("====================================")
        print("PREDICTION:", prediction)
        print(
            "CHURN PROBABILITY:",
            round(churn_probability, 2),
            "%"
        )
        print("MESSAGE:", message)
        print("====================================")


        # ----------------------------------------------------
        # SEND RESULT TO HTML
        # ----------------------------------------------------

        return jsonify({

            "prediction": prediction,

            "message": message,

            "churn_probability":
                round(churn_probability, 2)

        })


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print("====================================")
        print("ERROR:")
        print(repr(e))
        print("====================================")

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
