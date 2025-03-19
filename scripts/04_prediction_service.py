import requests
import joblib
import json
import os
import pandas as pd
from datetime import datetime

def fetch_new_penguin():
    """Fetch new penguin data from the API."""
    url = "http://130.225.39.127:8000/new_penguin/"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching penguin data: {e}")
        return None

def predict_species(penguin_data):
    """Predict the species of a penguin using the trained model."""
    try:
        # Load the trained model
        model_pipeline = joblib.load('models/penguin_classifier.joblib')
        
        # Extract model components
        scaler = model_pipeline['scaler']
        model = model_pipeline['model']
        features = model_pipeline['features']
        
        # Convert the input data into a DataFrame
        df = pd.DataFrame([{feature: penguin_data.get(feature, None) for feature in features}])
        
        # Scale the features
        df_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)[0]
        
        # Get prediction probabilities
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df_scaled)[0]
            classes = model.classes_
            probabilities = {int(cls): float(prob) for cls, prob in zip(classes, proba)}
        else:
            probabilities = {int(prediction): 1.0}  # Assign full probability to predicted class
        
        # Create a structured result
        result = {
            'predicted_species': int(prediction),
            'probabilities': probabilities,
            'penguin_data': penguin_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def save_prediction(prediction):
    """Save the prediction to JSON files for GitHub Pages."""
    try:
        os.makedirs('docs', exist_ok=True)
        
        # Save the latest prediction
        with open('docs/latest_prediction.json', 'w') as f:
            json.dump(prediction, f, indent=2)
        
        # Append to prediction history
        history_file = 'docs/prediction_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(prediction)
        
        # Keep only the last 30 predictions
        history = history[-30:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Prediction saved: {prediction['predicted_species']}")
    
    except Exception as e:
        print(f"Error saving prediction: {e}")

def main():
    """Main function to fetch new penguin data and make a prediction."""
    print(f"Starting prediction service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    penguin_data = fetch_new_penguin()
    
    if penguin_data:
        print(f"New penguin data received: {penguin_data}")
        
        prediction = predict_species(penguin_data)
        
        if prediction:
            save_prediction(prediction)
            print(f"Prediction: {prediction['predicted_species']}")
            print(f"Probabilities: {prediction['probabilities']}")
        else:
            print("Failed to make prediction")
    else:
        print("Failed to fetch penguin data")

if __name__ == "__main__":
    main()
