import pickle

try:
    with open('fraud_detection_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
