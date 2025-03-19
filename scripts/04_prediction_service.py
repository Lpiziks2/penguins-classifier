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
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching penguin data: {e}")
        return None

def predict_species(penguin_data):
    """Predict the species of a penguin using the trained model."""
    try:
        # Load the model pipeline
        model_pipeline = joblib.load('models/penguin_classifier.joblib')
        
        # Extract components
        scaler = model_pipeline['scaler']
        model = model_pipeline['model']
        features = model_pipeline['features']
        
        # Create a DataFrame with the penguin data
        df = pd.DataFrame([{
            feature: penguin_data[feature] 
            for feature in features
            if feature in penguin_data
        }])
        
        # Scale the features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        decision_scores = model.decision_function(scaled_features)
        
        # Get class names
        classes = model.classes_
        
        # Create results dictionary
        result = {
            'predicted_species': prediction,
            'probabilities': {cls: float(prob) for cls, prob in zip(classes, probabilities)},
            'penguin_data': penguin_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def save_prediction(prediction):
    """Save the prediction to a JSON file for GitHub Pages."""
    try:
        # Create docs directory if it doesn't exist
        os.makedirs('docs', exist_ok=True)
        
        # Save the current prediction
        with open('docs/latest_prediction.json', 'w') as f:
            json.dump(prediction, f, indent=2)
        
        # Load prediction history or create new
        history_file = 'docs/prediction_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Add new prediction to history
        history.append(prediction)
        
        # Keep only the last 30 predictions
        if len(history) > 30:
            history = history[-30:]
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Prediction saved: {prediction['predicted_species']}")
        
        # Update HTML file
        update_html_page(prediction, history)
        
    except Exception as e:
        print(f"Error saving prediction: {e}")

def update_html_page(prediction, history):
    """Update the HTML page with the latest prediction and history."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Penguins of Madagascar Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .prediction-card {{
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .history-item {{
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }}
        .adelie {{
            background-color: #d4edda;
        }}
        .chinstrap {{
            background-color: #d1ecf1;
        }}
        .gentoo {{
            background-color: #fff3cd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center my-4">Penguins of Madagascar Classifier</h1>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card prediction-card">
                    <div class="card-header bg-primary text-white">
                        <h3>Latest Prediction</h3>
                    </div>
                    <div class="card-body">
                        <h4>Species: <span class="badge bg-success">{prediction['predicted_species']}</span></h4>
                        <p>Timestamp: {prediction['timestamp']}</p>
                        
                        <h5>Measurement Data:</h5>
                        <ul class="list-group mb-3">
                            <li class="list-group-item">Bill Length: {prediction['penguin_data'].get('bill_length_mm', 'N/A')} mm</li>
                            <li class="list-group-item">Bill Depth: {prediction['penguin_data'].get('bill_depth_mm', 'N/A')} mm</li>
                            <li class="list-group-item">Flipper Length: {prediction['penguin_data'].get('flipper_length_mm', 'N/A')} mm</li>
                            <li class="list-group-item">Body Mass: {prediction['penguin_data'].get('body_mass_g', 'N/A')} g</li>
                        </ul>
                        
                        <h5>Prediction Confidence:</h5>
                        <div class="progress mb-1">
                            <div class="progress-bar bg-success" role="progressbar" 
                                 style="width: {prediction['probabilities'].get('Adelie', 0)*100}%" 
                                 aria-valuenow="{prediction['probabilities'].get('Adelie', 0)*100}" aria-valuemin="0" aria-valuemax="100">
                                Adelie: {prediction['probabilities'].get('Adelie', 0)*100:.1f}%
                            </div>
                        </div>
                        <div class="progress mb-1">
                            <div class="progress-bar bg-info" role="progressbar" 
                                 style="width: {prediction['probabilities'].get('Chinstrap', 0)*100}%" 
                                 aria-valuenow="{prediction['probabilities'].get('Chinstrap', 0)*100}" aria-valuemin="0" aria-valuemax="100">
                                Chinstrap: {prediction['probabilities'].get('Chinstrap', 0)*100:.1f}%
                            </div>
                        </div>
                        <div class="progress mb-1">
                            <div class="progress-bar bg-warning" role="progressbar" 
                                 style="width: {prediction['probabilities'].get('Gentoo', 0)*100}%" 
                                 aria-valuenow="{prediction['probabilities'].get('Gentoo', 0)*100}" aria-valuemin="0" aria-valuemax="100">
                                Gentoo: {prediction['probabilities'].get('Gentoo', 0)*100:.1f}%
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-secondary text-white">
                        <h3>Recent Predictions</h3>
                    </div>
                    <div class="card-body">
                        <div class="history-container">
                            {''.join([
                                f'''<div class="history-item {pred['predicted_species'].lower()}">
                                <strong>Species:</strong> {pred['predicted_species']} 
                                <strong>Time:</strong> {pred['timestamp']}
                                <br>
                                <small>Bill: {pred['penguin_data'].get('bill_length_mm', 'N/A')}mm × {pred['penguin_data'].get('bill_depth_mm', 'N/A')}mm, 
                                Flipper: {pred['penguin_data'].get('flipper_length_mm', 'N/A')}mm, 
                                Mass: {pred['penguin_data'].get('body_mass_g', 'N/A')}g</small>
                            </div>''' 
                            for pred in history[-10:][::-1]])}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-dark text-white">
                        <h3>About the Project</h3>
                    </div>
                    <div class="card-body">
                        <p>This is a classifier that predicts penguin species based on physical measurements. The goal is to find Skipper, Private, Rico, and Kowalski from the Penguins of Madagascar, who are Adelie penguins.</p>
                        <p>New penguin data is fetched daily at 7:30 AM from the API, and predictions are updated on this page.</p>
                        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    
    with open('docs/index.html', 'w') as f:
        f.write(html_content)

def main():
    """Main function to fetch new penguin data and make a prediction."""
    print(f"Starting prediction service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch new penguin data
    penguin_data = fetch_new_penguin()
    
    if penguin_data:
        print(f"New penguin data received: {penguin_data}")
        
        # Make prediction
        prediction = predict_species(penguin_data)
        
        if prediction:
            # Save prediction for GitHub Pages
            save_prediction(prediction)
            
            # Print result
            print(f"Prediction: {prediction['predicted_species']}")
            print(f"Probabilities: {prediction['probabilities']}")
        else:
            print("Failed to make prediction")
    else:
        print("Failed to fetch penguin data")

if __name__ == "__main__":
    main()
