'''
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Example data extraction from form
            data = [float(request.form['feature1']), float(request.form['feature2'])]  # adjust for your form fields
            data = np.array(data).reshape(1, -1)
            
            # Scale the data
            scaled_data = standard_scaler.transform(data)
            
            # Predict using the ridge model
            prediction = ridge_model.predict(scaled_data)
            
            # Return the prediction
            return jsonify({'prediction': prediction[0]})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
'''