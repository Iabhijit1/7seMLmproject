from flask import Flask, render_template, request
import joblib
import numpy as np
import os  # <--- Added for Render compatibility

app = Flask(__name__)

# Load the trained ensemble model and scaler
loaded_ensemble_model = joblib.load('ensemble_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        input_features = [float(x) for x in request.form.values()]
        # Scale the input features
        scaled_features = scaler.transform([input_features])
        # Make prediction using the loaded model
        prediction = loaded_ensemble_model.predict(scaled_features)
        result = 'Malignant' if prediction[0] == 1 else 'Benign'
        return render_template('index.html', result=result)

if __name__ == '__main__':
    # Render assigns a dynamic port via environment variables. 
    # If it's not found (like when running locally), it defaults to 5000.
    port = int(os.environ.get("PORT", 5000))
    # '0.0.0.0' is required to make the server accessible externally on Render
    app.run(host='0.0.0.0', port=port)
