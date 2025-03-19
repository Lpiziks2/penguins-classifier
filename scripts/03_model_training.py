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
    """Preprocess data: encode categorical features, handle missing values, and scale."""
    
    # Encode target variable (species)
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    
    # Separate features and target
    X = df.drop(columns=['species'])
    y = df['species']
    
    # Scale numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y, scaler

def analyze_features():
    """Analyze and select the most important features using multiple methods with cross-validation."""
    print("Loading and preprocessing data...")
    df = load_data_from_db()
    X, y, scaler = preprocess_data(df)
    
    # 1. Filter Method - ANOVA F-test
    print("\n1. ANOVA F-test for Feature Selection")
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_
    }).sort_values('F_Score', ascending=False)
    
    print(feature_scores)

    # 2. Embedded Method - Random Forest Feature Importance (Cross-validated)
    print("\n2. Random Forest Feature Importance (Cross-validated)")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')
    
    print(f"Random Forest Cross-validation Accuracy: {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
    
    rf.fit(X, y)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance)
    
    # 3. Permutation Importance
    print("\n3. Permutation Importance")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    rf.fit(X_train, y_train)
    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    
    perm_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': result.importances_mean
    }).sort_values('Importance', ascending=False)
    
    print(perm_importance)
    
    # Save the trained model
    os.makedirs('models', exist_ok=True)
    model_pipeline = {
        'model': rf,
        'scaler': scaler,
        'features': X.columns.tolist()
    }
    joblib.dump(model_pipeline, 'models/penguin_classifier.joblib')
    print("Model saved successfully in 'models/penguin_classifier.joblib'")
    
    # Final feature selection based on importance
    selected_features = feature_importance.head(4)['Feature'].tolist()
    
    print(f"\nFinal selected features for model training: {selected_features}")
    
    return selected_features

if __name__ == "__main__":
    selected_features = analyze_features()
    print(f"\nSelected features for model training: {selected_features}")
