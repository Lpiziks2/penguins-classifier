import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

def load_data_from_db():
    """Load data from SQLite database."""
    conn = sqlite3.connect('data/penguins.db')
    query = """
    SELECT p.species, p.bill_length_mm, p.bill_depth_mm, p.flipper_length_mm, 
           p.body_mass_g, p.sex, i.name as island
    FROM PENGUINS p
    JOIN ISLANDS i ON p.island_id = i.island_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def analyze_features():
    """Analyze and select the most important features for penguin classification."""
    # Load data
    print("Loading data from database...")
    df = load_data_from_db()
    
    # Basic exploration
    print("\nDataset shape:", df.shape)
    print("\nFeature statistics:")
    print(df.describe())
    
    # Prepare data for feature selection
    # Encoding target variable
    le = LabelEncoder()
    y = le.fit_transform(df['species'])
    
    # We'll focus on the numerical features for this analysis
    numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = df[numeric_features]
    
    # 1. Filter Method - ANOVA F-value
    print("\n1. Filter Method - ANOVA F-value")
    selector = SelectKBest(f_classif, k=len(numeric_features))
    selector.fit(X, y)
    
    # Get scores and p-values
    feature_scores = pd.DataFrame({
        'Feature': numeric_features,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_
    })
    print(feature_scores.sort_values('F_Score', ascending=False))
    
    # 2. Embedded Method - Random Forest Feature Importance
    print("\n2. Embedded Method - Random Forest Feature Importance")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': rf.feature_importances_
    })
    print(feature_importance.sort_values('Importance', ascending=False))
    
    # 3. Permutation Importance
    print("\n3. Permutation Importance")
    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    
    perm_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': result.importances_mean
    })
    print(perm_importance.sort_values('Importance', ascending=False))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    
    # Plot Random Forest feature importance
    plt.subplot(1, 2, 1)
    sns.barplot(x='Importance', y='Feature', data=feature_importance.sort_values('Importance'))
    plt.title('Random Forest Feature Importance')
    
    # Plot Permutation importance
    plt.subplot(1, 2, 2)
    sns.barplot(x='Importance', y='Feature', data=perm_importance.sort_values('Importance'))
    plt.title('Permutation Importance')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Feature correlation with target
    print("\nFeature correlation analysis:")
    # One-hot encode species for correlation analysis
    species_dummies = pd.get_dummies(df['species'], prefix='species')
    correlation_data = pd.concat([df[numeric_features], species_dummies], axis=1)
    
    # Get correlations
    correlation_matrix = correlation_data.corr()
    
    # Extract correlations with species
    species_cols = [col for col in correlation_matrix.columns if col.startswith('species_')]
    feature_correlations = correlation_matrix.loc[numeric_features, species_cols]
    
    print("\nFeature correlations with species:")
    print(feature_correlations)
    
    # Based on all methods, determine the final selected features
    print("\nFinal selected features:")
    selected_features = numeric_features  # In this case, all features are valuable
    print(selected_features)
    
    return selected_features

if __name__ == "__main__":
    selected_features = analyze_features()
    print(f"\nSelected features for model training: {selected_features}")
