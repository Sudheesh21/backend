from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load('mutual_fund_tree.pkl')  # Your PKL file

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']  # Flutter sends 7 numbers
    features = pd.DataFrame([data], columns=model.feature_names_in_)
    prediction = model.predict(features)[0]
    actions = ['Pause', 'Continue', 'Shift']
    return jsonify({'action': actions[int(prediction)]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
