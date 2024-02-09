# app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS  # Import the CORS extension

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the machine learning model
with open('random_forest_regressor_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.json['data']

    # Convert input data to a numpy array
    input_array = np.array(input_data).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(input_array)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
