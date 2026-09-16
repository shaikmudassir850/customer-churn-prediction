from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("best_ada_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        print("RECEIVED DATA:", data)

        # Create raw input
        input_data = pd.DataFrame([{
            "tenure_months": float(data["tenure_months"]),
            "contract_type": int(data["contract_type"]),
            "monthly_charges": float(data["monthly_charges"]),
            "total_charges": float(data["total_charges"]),
            "payment_method": int(data["payment_method"]),
            "avg_monthly_usage_gb": float(data["avg_monthly_usage_gb"]),
            "satisfaction_score": float(data["satisfaction_score"]),
            "autopay_enabled": int(data["autopay_enabled"])
        }])

        print("RAW INPUT:")
        print(input_data)

        # Columns scaled during training
        numeric_columns = [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "avg_monthly_usage_gb"
        ]

        # Scale only numeric columns
        scaled_numeric = pd.DataFrame(
            scaler.transform(input_data[numeric_columns]),
            columns=numeric_columns
        )

        # Add categorical columns
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

        # Exact model order
        final_input = final_input[
            model.feature_names_in_
        ]

        print("FINAL INPUT:")
        print(final_input)

        # Prediction
        prediction = int(model.predict(final_input)[0])

        probability = model.predict_proba(final_input)[0]

        classes = list(model.classes_)

        churn_probability = (
            float(probability[classes.index(1)]) * 100
        )

        if prediction == 1:
            message = "Customer likely to leave"
        else:
            message = "Customer likely to stay"

        print("PREDICTION:", prediction)
        print("CHURN PROBABILITY:", churn_probability)

        return jsonify({
            "prediction": prediction,
            "message": message,
            "churn_probability": round(churn_probability, 2)
        })

    except Exception as e:

        print("ERROR:", repr(e))

        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
