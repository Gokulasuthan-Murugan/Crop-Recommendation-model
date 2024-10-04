import pickle 
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open('normalize.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])


def predict_api():
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        # Safely get the data from request
        request_data = request.json
        if 'data' not in request_data:
            return jsonify({"error": "Missing 'data' key in request"}), 400

        data = request_data['data']
        print("Received data:", data)

        # Ensure data is in correct format (dictionary)
        if not isinstance(data, dict):
            return jsonify({"error": "Data must be a dictionary"}), 400

        # Convert values to float with validation
        try:
            numeric_data = []
            for key, value in data.items():
                # Skip if value is a function or method
                if callable(value):
                    return jsonify({"error": f"Value for {key} is a function, not a number"}), 400
                
                # Try to convert to float
                try:
                    float_value = float(value)
                    numeric_data.append(float_value)
                except (ValueError, TypeError):
                    return jsonify({"error": f"Could not convert value for {key} to number"}), 400

            # Reshape for model prediction
            new_data = np.array(numeric_data).reshape(1, -1)
            
            # Make prediction
            output = model.predict(new_data)
            
            return jsonify({"prediction": float(output[0])})

        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return jsonify({"error": f"Error processing data: {str(e)}"}), 400

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
        

if __name__=="main":
    app.run(debug=True)
