from pathlib import Path

from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import numpy as np
import pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__, template_folder=str(BASE_DIR))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/style.css')
def styles():
    return send_from_directory(BASE_DIR, 'style.css')

@app.route('/img.jpg')
def image():
    return send_from_directory(BASE_DIR, 'img.jpg')

@app.route('/predict', methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosphorus'])
    K = float(request.form['Potassium'])
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    ph = float(request.form['ph'])
    Rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, Temperature, Humidity, ph, Rainfall]
    single_pred = pd.DataFrame([feature_list], columns=[
        'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
    ])
    prediction = model.predict(single_pred)[0]

    result = f"{prediction} is the best crop to be cultivated right there"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)