from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained model (ensure the path to your model is correct)
model = pickle.load(open('fraud_detection_model.pkl', 'rb'))

# Define the /predict route with POST method
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the POST request
        data = request.get_json()
        print(f"Received data: {data}")  # This is for debugging
        
        # Make sure the "features" key exists in the data
        if 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400

        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)

        return jsonify({'prediction': 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# Define the home route (optional, for testing)
@app.route('/')
def home():
    return "Credit Card Fraud Detection API is Running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    app.run(debug=True)



  
