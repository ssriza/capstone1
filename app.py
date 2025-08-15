from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and feature list
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

@app.route("/")
def home():
    return render_template("index.html", features=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data from HTML
        input_data = [float(request.form[feature]) for feature in feature_names]

        # Create DataFrame
        df = pd.DataFrame([input_data], columns=feature_names)

        # Scale data
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]

        return render_template(
            "index.html",
            features=feature_names,
            prediction_text=f"Churn: {prediction}, Probability: {probability:.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            features=feature_names,
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
