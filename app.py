from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ============================================================
# LOAD SAVED MACHINE LEARNING FILES
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
        # GET JSON DATA FROM HTML
        # ----------------------------------------------------

        data = request.get_json()

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        #
        # HTML ALREADY SENDS ENCODED NUMBERS
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
        # EXACT FEATURE ORDER USED FOR TRAINING
        # ----------------------------------------------------

        features = [
            "tenure_months",
            "contract_type",
            "monthly_charges",
            "total_charges",
            "payment_method",
            "avg_monthly_usage_gb",
            "satisfaction_score",
            "autopay_enabled"
        ]

        input_data = input_data[features]


        # ----------------------------------------------------
        # SCALE INPUT DATA
        # ----------------------------------------------------

        input_scaled = scaler.transform(input_data)


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_scaled)[0]


        # ----------------------------------------------------
        # CONVERT NUMPY INTEGER TO NORMAL PYTHON INTEGER
        # ----------------------------------------------------

        prediction = int(prediction)


        # ----------------------------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------------------------

        churn_probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_scaled)[0]

            classes = list(model.classes_)

            if 1 in classes:

                churn_index = classes.index(1)

                churn_probability = (
                    float(probabilities[churn_index]) * 100
                )


        # ----------------------------------------------------
        # RETURN NUMERIC PREDICTION
        #
        # 0 = STAY
        # 1 = CHURN / LEAVE
        # ----------------------------------------------------

        return jsonify({

            "prediction": prediction,

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
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    ) 
