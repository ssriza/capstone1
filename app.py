from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

yes_no_map = {"Yes": 1, "No": 0}

feature_order = [
    "SeniorCitizen",
    "partner",
    "dependents",
    "phone_service",
    "internet_service",
    "online_security",
    "tech_support",
    "contract"
]

def preprocess_input(form_data):
    processed_data = {}
    for feature in feature_order:
        value = form_data.get(feature)
        if value in yes_no_map:
            processed_data[feature] = yes_no_map[value]
        else:
            processed_data[feature] = float(value)
    return [[processed_data[f] for f in feature_order]]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form
    X = preprocess_input(form_data)
    prediction = model.predict(X)
    prediction_label = "Yes" if prediction[0] == 1 else "No"
    return render_template("result.html", prediction=prediction_label)

if __name__ == "__main__":
    app.run(debug=True)
