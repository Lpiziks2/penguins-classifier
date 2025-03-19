# Penguins of Madagascar Classifier

A machine learning project that classifies penguin species based on physical measurements. This project demonstrates the full machine learning pipeline from data preparation to model deployment and prediction visualization.

## Project Overview

This project uses the Palmer Penguins dataset to build a classifier that can identify penguin species (Adelie, Chinstrap, or Gentoo) based on physical measurements such as bill dimensions, flipper length, and body mass. The project includes:

- Data preparation and storage in SQLite database
- Feature selection and analysis
- Model training and evaluation
- Prediction service that fetches new penguin data from an API
- Web visualization of predictions

## Installation

```bash
# Clone the repository
git clone 
cd penguins-classifier

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── .github/workflows/  # GitHub Actions workflows for automation
├── data/               # Data storage
│   └── penguins.db     # SQLite database with penguin data
├── docs/               # Web visualization files
│   ├── index.html      # Main visualization page
│   ├── latest_prediction.json  # Latest prediction data
│   └── prediction_history.json # History of predictions
├── models/             # Trained models
│   └── penguin_classifier.joblib  # Saved model pipeline
├── scripts/            # Python scripts
│   ├── 01_data_preparation.py    # Data loading and preprocessing
│   ├── 02_feature_selection.py   # Feature analysis and selection
│   ├── 03_model_training.py      # Model training and evaluation
│   └── 04_prediction_service.py  # Prediction service and visualization
└── requirements.txt    # Project dependencies
```

## Usage

### 1. Data Preparation

The data preparation script loads penguin data and stores it in an SQLite database:

```bash
python scripts/01_data_preparation.py
```

### 2. Feature Selection

Analyze and select the most important features for classification:

```bash
python scripts/02_feature_selection.py
```

This script performs feature importance analysis using multiple methods (ANOVA F-value, Random Forest importance, and Permutation importance) and generates visualizations.

### 3. Model Training

Train and evaluate classification models:

```bash
python scripts/03_model_training.py
```

This script compares Random Forest and SVM models with hyperparameter tuning, selects the best model, and saves it for later use.

### 4. Prediction Service

Fetch new penguin data from the API and make predictions:

```bash
python scripts/04_prediction_service.py
```

This script:
- Fetches new penguin measurements from an external API
- Uses the trained model to predict the species
- Saves prediction results and updates the visualization

## Web Visualization

The project includes a web visualization component that displays the latest prediction and prediction history. After running the prediction service, you can view the results by opening `docs/index.html` in a web browser.

## Model Performance

The model selection process compares different algorithms (Random Forest and SVM) with various hyperparameters. The best model is selected based on accuracy and other metrics such as precision, recall, and F1-score.

