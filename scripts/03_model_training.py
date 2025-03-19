import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance

def load_data_from_db():
    """Load data from SQLite database."""
    conn = sqlite3.connect('data/penguins.db')
    query = """
    SELECT p.species, p.bill_length_mm, p.bill_depth_mm, p.flipper_length_mm, 
           p.body_mass_g
    FROM PENGUINS p
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def preprocess_data(df):
    """Preprocess data: encode target variable and scale numerical features."""
    df['species'] = LabelEncoder().fit_transform(df['species'])
    X = df.drop(columns=['species'])
    y = df['species']
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y, scaler

def train_and_save_model():
    """Train model, evaluate features, and save the trained model."""
    print("Loading and preprocessing data...")
    df = load_data_from_db()
    X, y, scaler = preprocess_data(df)
    
    print("\nTraining Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')
    
    print(f"Cross-validation Accuracy: {np.mean(accuracy):.4f} ± {np.std(accuracy):.4f}")
    rf.fit(X, y)
    
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
    print(feature_importance.sort_values('Importance', ascending=False))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': rf, 'scaler': scaler, 'features': X.columns.tolist()}, 'models/penguin_classifier.joblib')
    print("Model saved successfully in 'models/penguin_classifier.joblib'")

if __name__ == "__main__":
    train_and_save_model()
