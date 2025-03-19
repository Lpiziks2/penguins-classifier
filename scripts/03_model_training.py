import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data_from_db():
    """Load data from SQLite database."""
    conn = sqlite3.connect('data/penguins.db')
    query = """
    SELECT species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
    FROM PENGUINS
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def train_model():
    """Train and evaluate a penguin species classifier."""
    print("Loading data from database...")
    df = load_data_from_db()
    
    # Define features
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = df[features]
    y = df['species']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to compare
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    # Hyperparameters
    param_grid = {
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],  
            'min_samples_split': [5, 10]  
        },
        'SVM': {
            'C': [0.1, 1, 5], 
            'kernel': ['linear', 'rbf']
        }
    }
    
    # Stratified K-Fold for better generalization
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_score = 0
    best_model = None
    best_model_name = ''
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        grid_search = GridSearchCV(model, param_grid[name], cv=cv, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = grid_search.predict(X_test_scaled)
        score = accuracy_score(y_test, y_pred)
        
        print(f"{name} best parameters: {grid_search.best_params_}")
        print(f"{name} accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with accuracy: {best_score:.4f}")
    
    # Evaluation
    y_pred = best_model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump({'scaler': scaler, 'model': best_model, 'features': features}, 'models/penguin_classifier.joblib')
    print("\nModel saved to models/penguin_classifier.joblib")
    
    return {'model_type': best_model_name, 'accuracy': best_score, 'features': features}

if __name__ == "__main__":
    model_info = train_model()
    print(f"\nTrained {model_info['model_type']} with {model_info['accuracy']:.4f} accuracy")
    print(f"Using features: {model_info['features']}")
