from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# Load the model
model = joblib.load('model/xgboost_model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')
features = joblib.load('model/features.pkl')

# Setup Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    top_diseases = []

    if request.method == "POST":
        input_data = [0] * len(features)
        selected_symptoms = request.form.getlist("symptoms")

        for symptom in selected_symptoms:
            if symptom in features:
                idx = features.index(symptom)
                input_data[idx] = 1

        probabilities = model.predict_proba([input_data])[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = [(label_encoder.inverse_transform([i])[0], round(probabilities[i] * 100, 2)) for i in top_indices]

    return render_template("index.html", features=features, top_diseases=top_diseases)

if __name__ == "__main__":
    app.run(debug=True)
